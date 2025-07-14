import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange, repeat
from torchdiffeq import odeint, odeint_adjoint
from scene.cameras import Camera

import math


class CoMoKernel(nn.Module):
    def __init__(self, 
                 num_views: int = None,
                 view_dim: int = 64,
                 num_warp: int = 9,
                 method: str = 'rk4',
                 adjoint: bool = False,
                 iteration: int = None,
                 ) -> None:
        super(CoMoKernel, self).__init__()

        self.num_warp = num_warp
        self.model = CoMoModule(num_views=num_views,
                          view_dim=view_dim,
                          num_warp=num_warp,
                          method=method,
                          adjoint=adjoint)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.lr_factor = (1e-4 - 1e-6) / iteration
    
    def get_warped_cams(self,
                        cam : Camera = None
                        ):
        
        Rt = self.get_Rt(cam)
        idx_view = cam.uid
        warped_Rt, ortho_loss = self.model(Rt, idx_view)
        warped_cams = [self.get_cam(cam, warped_Rt[i]) for i in range(self.num_warp)]

        return warped_cams, ortho_loss
    
    def get_cam(self,
                cam: Camera = None,
                Rt: torch.Tensor = None
                ) -> Camera:
        
        return Camera(colmap_id=cam.colmap_id, R=Rt[:3, :3], T=Rt[:3, 3], FoVx=cam.FoVx, FoVy=cam.FoVy,
                      image=cam.image, gt_alpha_mask=cam.gt_alpha_mask, image_name=cam.image_name,
                      uid=cam.uid, data_device=cam.data_device)

    def get_Rt(self, 
               cam: Camera = None
               ) -> torch.Tensor:
        R, T = cam.R, cam.T
        Rt = np.concatenate([R, T[:, None]], axis=-1)
        Rt_fill = np.array([0, 0, 0, 1])[None]
        Rt = np.concatenate([Rt, Rt_fill], axis=0)
        Rt = torch.tensor(Rt, dtype=torch.float32).cuda()
        return Rt
    
    def get_weight_and_mask(self,
                            img: torch.Tensor = None,
                            idx_view: int = None,
                            ):
        weight, mask = self.model.get_weight_and_mask(img, idx_view)
        return weight, mask

    def adjust_lr(self) -> None:
        for param_group in self.optimizer.param_groups:
            param_group['lr'] -= self.lr_factor


class NeuralDerivative(nn.Module):
    def __init__(self,
                 view_dim: int = 32,
                 num_views: int = 29,
                 num_warp: int = 5,
                 time_dim: int = 8,
                 ) -> None:
        super(NeuralDerivative, self).__init__()

        self.view_dim = view_dim
        self.num_views = num_views
        self.num_warp = num_warp

        self.time_embedder = nn.Parameter(
            torch.zeros(num_warp, time_dim).type(torch.float32), 
            requires_grad=True
        )
        self.linear_p = nn.Linear(view_dim // 2 + time_dim, view_dim // 2)
        self.linear_q = nn.Linear(view_dim // 2 + time_dim, view_dim // 2)

        self.relu = nn.ReLU()

    def forward(self,
                t: float = 0,
                x: torch.Tensor = None,
                ) -> torch.Tensor:
        
        t_embed = self.time_embedder[int(t)]
        x = self.relu(x)

        p, q = torch.chunk(x, 2, dim=-1)

        p = torch.cat([p, t_embed], dim=-1)
        q = torch.cat([q, t_embed], dim=-1)

        p, q = self.linear_p(p), self.linear_q(q)

        return torch.cat([p, q], dim=-1)
    

class DiffEqSolver(nn.Module):
    def __init__(self, 
                 odefunc: nn.Module = None,
                 method: str = 'euler',
                 odeint_rtol: float = 1e-4,
                 odeint_atol: float = 1e-5,
                 num_warp: int = 9,
                 adjoint: bool = False,
                 ) -> None:
        super(DiffEqSolver, self).__init__()
        
        self.ode_func = odefunc
        self.ode_method = method
        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol
        self.integration_time = torch.arange(0, num_warp, dtype=torch.long)
        self.t_mid = num_warp // 2
        self.solver = odeint_adjoint if adjoint else odeint
            
    def forward(self, 
                x: torch.Tensor = None,
                ) -> torch.Tensor:
        
        t = self.integration_time.type_as(x)
        forward_t = t[t >= self.t_mid]                   # [ 4, 5, 6, 7, 8 ]
        backward_t = t[t <= self.t_mid].flip(dims=[0])   # [ 4, 3, 2, 1, 0 ]
        out_forward = self.solver(self.ode_func, x, forward_t.cuda(x.get_device()), 
                                  rtol=self.odeint_rtol, atol=self.odeint_atol, 
                                  method=self.ode_method)
        out_backward = self.solver(self.ode_func, x, backward_t.cuda(x.get_device()), 
                                   rtol=self.odeint_rtol, atol=self.odeint_atol, 
                                   method=self.ode_method)
        out = torch.cat([out_backward.flip(dims=[0]), out_forward[1:]], dim=0)
        
        return out


class CoMoModule(nn.Module):
    def __init__(self,
                 num_views: int = 29,
                 view_dim: int = 32,
                 num_warp: int = 9,
                 method: str = 'euler',
                 adjoint: bool = False,
                 ) -> None:
        super(CoMoModule, self).__init__()

        self.num_warp = num_warp

        self.view_embedder = nn.Parameter(
            torch.zeros(num_views, view_dim).type(torch.float32), 
            requires_grad=True
        )

        self.linear_Rt = nn.ModuleList()
        
        self.encoder_rigid = nn.ModuleList()
        self.encoder_cmr = nn.ModuleList()
        
        self.diffeq_solver_rigid = nn.ModuleList()
        self.diffeq_solver_cmr = nn.ModuleList()
        
        self.decoder_rigid_w = nn.ModuleList()
        self.decoder_rigid_v = nn.ModuleList()
        self.decoder_rigid_theta = nn.ModuleList()
        
        self.decoder_cmr_rot = nn.ModuleList()
        self.decoder_cmr_trans = nn.ModuleList()

        self.mlpWeight = nn.ModuleList()
        self.mlpMask = nn.ModuleList()


        for i in range(num_views):
            
            self.linear_Rt.append(nn.Linear(12, view_dim))
            
            self.encoder_rigid.append(nn.Linear(view_dim + view_dim, view_dim))
            self.encoder_cmr.append(nn.Linear(view_dim + view_dim, view_dim))
            
            self.diffeq_solver_rigid.append(DiffEqSolver(
                odefunc=NeuralDerivative(view_dim=view_dim, num_views=num_views, num_warp=num_warp),
                method=method, num_warp=num_warp, adjoint=adjoint))
            
            self.diffeq_solver_cmr.append(DiffEqSolver(
                odefunc=NeuralDerivative(view_dim=view_dim, num_views=num_views, num_warp=num_warp),
                method=method, num_warp=num_warp, adjoint=adjoint))
            
            self.decoder_rigid_w.append(nn.Linear(view_dim // 2, 3))
            self.decoder_rigid_v.append(nn.Linear(view_dim // 2, 3))
            self.decoder_rigid_theta.append(nn.Linear(view_dim // 2, 1))
            
            self.decoder_cmr_rot.append(nn.Linear(view_dim // 2, 9))
            self.decoder_cmr_trans.append(nn.Linear(view_dim // 2, 3))
            
            gain = 0.00001 / (math.sqrt((view_dim // 2 + 3) / 6))
            self._init(self.decoder_rigid_w[i], gain=gain)
            self._init(self.decoder_rigid_v[i], gain=gain)
            self._init(self.decoder_rigid_theta[i], gain=gain)
            self._init(self.decoder_cmr_rot[i], gain=gain)
            self._init(self.decoder_cmr_trans[i], gain=gain)


        # conv, mlp_weight, mlp_mask from BAGS (https://github.com/snldmt/BAGS/)
        channels = 32
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.InstanceNorm2d(channels),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.InstanceNorm2d(channels),
        )

        self.mlp_weight = nn.Conv2d(channels, 1, 1, bias=False)
        self.mlp_mask = nn.Conv2d(channels * num_warp, 1, 1, bias=False)
    
    
    def _init(self, 
              layer: nn.Module, 
              gain: float = 0.00001):
        
        nn.init.xavier_uniform_(layer.weight, gain=gain)
        if layer.bias is not None:
            layer.bias.data.fill_(0)


    def get_weight_and_mask(self, 
                            img: torch.Tensor = None,
                            idx_view: int = None,
                            ):
        
        feat = self.conv(img)
        weight = self.mlp_weight(feat)
        weight = F.softmax(weight, dim=0)
        
        feat_mask = rearrange(feat, 't c h w -> 1 (t c ) h w')
        mask = torch.sigmoid(self.mlp_mask(feat_mask))[0]

        return weight, mask
        
        
    def forward(self,
                Rt: torch.Tensor = None,
                idx_view: int = None,
                ) -> torch.Tensor:

        Rt_encoded = self.linear_Rt[idx_view](Rt[:3, :].reshape(-1))
        view_embed = self.view_embedder[idx_view]
        view_embed = torch.cat([view_embed, Rt_encoded], dim=-1)

        z_rigid = self.encoder_rigid[idx_view](view_embed)
        Z_rigid = self.diffeq_solver_rigid[idx_view](z_rigid)
        T_rigid = self.get_rigid_transformation(Z_rigid, idx_view)
        
        z_cmr = self.encoder_cmr[idx_view](view_embed)
        Z_cmr = self.diffeq_solver_cmr[idx_view](z_cmr)
        T_cmr, R_cmr = self.get_cmr_transformation(Z_cmr, idx_view)
        
        T_transform = torch.matmul(T_rigid, T_cmr)
        Rt_new = torch.einsum('ij, tjk -> tik', Rt, T_transform)
        
        w_loss = (torch.matmul(R_cmr, R_cmr.transpose(1, 2)) \
            - torch.eye(3)[None].repeat(self.num_warp, 1, 1).to(R_cmr)).abs().mean()
        
        return Rt_new, w_loss
    
    
    def get_rigid_transformation(self, 
                                 latent_rigid: torch.Tensor = None,
                                 idx_view: int = None
                                 ) -> torch.Tensor:
        
        Z_RIGID_w, Z_RIGID_v = torch.chunk(latent_rigid, 2, dim=-1)
        
        w = self.decoder_rigid_w[idx_view](Z_RIGID_w)
        theta = self.decoder_rigid_theta[idx_view](Z_RIGID_w)[..., None]
        v = self.decoder_rigid_v[idx_view](Z_RIGID_v)
        
        w = self.exp_map(w)
        w_skew = self.skew_symmetric(w)
        R_exp = self.rodrigues_formula(w_skew, theta)
        G = self.G_formula(w_skew, theta)
        p = torch.matmul(G, v[..., None])
        T_rigid = self.transform_SE3(R_exp, p)
        
        return T_rigid

    def transform_SE3(self, 
                      exp_w_skew: torch.Tensor, 
                      p: torch.Tensor
                      ) -> torch.Tensor:
        
        delta_Rt = torch.cat([exp_w_skew, p], dim=-1)
        delta_Rt_fill = torch.tensor([0, 0, 0, 1])[None].repeat(delta_Rt.size(0), 1, 1).to(delta_Rt)
        delta_Rt = torch.cat([delta_Rt, delta_Rt_fill], dim=1)
        return delta_Rt
    
    def rodrigues_formula(self, 
                          w: torch.Tensor, 
                          theta: torch.Tensor,
                          ) -> torch.Tensor:
        
        term1 = torch.eye(3).to(w)
        term2 = torch.sin(theta) * w
        term3 = (1 - torch.cos(theta)) * torch.matmul(w, w)
        return term1 + term2 + term3
    
    def G_formula(self,
                  w: torch.Tensor, 
                  theta: torch.Tensor,
                  ) -> torch.Tensor:
        
        term1 = torch.eye(3)[None].to(w) * theta
        term2 = (1 - torch.cos(theta)) * w
        term3 = (theta - torch.sin(theta)) * torch.matmul(w, w)
        return term1 + term2 + term3

    def exp_map(self, 
                w: torch.Tensor,
                ) -> torch.Tensor:
        norm = torch.norm(w, dim=-1)[..., None] + 1e-10
        w = w / norm
        return w

    def skew_symmetric(self, 
                       w : torch.Tensor,
                       ) -> torch.Tensor:
        
        w1, w2, w3 = torch.chunk(w, 3, dim=-1)

        w_skew =  torch.cat([torch.zeros_like(w1), -w3, w2,
                             w3, torch.zeros_like(w1), -w1,
                             -w2, w1, torch.zeros_like(w1)], dim=-1)
        w_skew = w_skew.reshape(-1, 3, 3)
        return w_skew
    
    def get_cmr_transformation(self, 
                               latent_cmr: torch.Tensor = None,
                               idx_view: int = None
                               ) -> torch.Tensor:
        
        Z_CMR_rot, Z_CMR_trans = torch.chunk(latent_cmr, 2, dim=-1)
        
        R_cmr = self.decoder_cmr_rot[idx_view](Z_CMR_rot).reshape(-1, 3, 3) \
            + torch.eye(3)[None].repeat(self.num_warp, 1, 1).to(Z_CMR_rot)
        t_cmr = self.decoder_cmr_trans[idx_view](Z_CMR_trans)[..., None]
        T_cmr = self.transform_SE3(R_cmr, t_cmr)
        
        return T_cmr, R_cmr