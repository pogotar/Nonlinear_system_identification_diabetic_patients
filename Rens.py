import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np  # linear algebra


class ContractiveREN(nn.Module):
    """
    Acyclic contractive recurrent equilibrium network, following the paper:
    "Recurrent equilibrium networks: Flexible dynamic models with guaranteed
    stability and robustness, Revay M et al. ."

    The mathematical model of RENs relies on an implicit layer embedded in a recurrent layer.
    The model is described as,

                    [  E . x_t+1 ]  =  [ F    B_1  B_2   ]   [  x_t ]   +   [  b_x ]
                    [  Λ . v_t   ]  =  [ C_1  D_11  D_12 ]   [  w_t ]   +   [  b_w ]
                    [  y_t       ]  =  [ C_2  D_21  D_22 ]   [  u_t ]   +   [  b_u ]

    where E is an invertible matrix and Λ is a positive-definite diagonal matrix. The model parameters
    are then {E, Λ , F, B_i, C_i, D_ij, b} which form a convex set according to the paper.

    NOTE: REN has input "u", output "y", and internal state "x". When used in closed-loop,
          the REN input "u" would be the noise reconstruction ("w") and the REN output ("y")
          would be the input to the plant. The internal state of the REN ("x") should not be mistaken
          with the internal state of the plant.
    """

    def __init__(
        self, dim_in: int, dim_out: int, dim_internal: int,
        dim_nl: int, internal_state_init = None, y_init = None,
        initialization_std: float = 0.5, pos_def_tol: float = 0.001, contraction_rate_lb: float = 1.0
    ):
        """
        Args:
            dim_in (int): Input (u) dimension.
            dim_out (int): Output (y) dimension.
            dim_internal (int): Internal state (x) dimension. This state evolves with contraction properties.
            dim_nl (int): Dimension of the input ("v") and ouput ("w") of the nonlinear static block.
            initialization_std (float, optional): Weight initialization. Set to 0.1 by default.
            internal_state_init (torch.Tensor or None, optional): Initial condition for the internal state. Defaults to 0 when set to None.
            epsilon (float, optional): Positive and negligible scalar to force positive definite matrices.
            contraction_rate_lb (float, optional): Lower bound on the contraction rate. Defaults to 1.
        """
        super().__init__()

        # set dimensions
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_internal = dim_internal
        self.dim_nl = dim_nl

        # set functionalities
        self.contraction_rate_lb = contraction_rate_lb

        # auxiliary elements
        self.epsilon = pos_def_tol

        # # # free parameters
        # define matrices shapes
        # auxiliary matrices
        self.X_shape = (2 * self.dim_internal + self.dim_nl, 2 * self.dim_internal + self.dim_nl)
        self.Y_shape = (self.dim_internal, self.dim_internal)
        # nn state dynamics
        self.B2_shape = (self.dim_internal, self.dim_in)
        # nn output
        self.C2_shape = (self.dim_out, self.dim_internal)
        self.D21_shape = (self.dim_out, self.dim_nl)
        self.D22_shape = (self.dim_out, self.dim_in)
        # v signal
        self.D12_shape = (self.dim_nl, self.dim_in)

        # define trainable params
        self.training_param_names = ['X', 'Y', 'B2', 'C2', 'D21', 'D22', 'D12']
        self._init_trainable_params(initialization_std)

        # mask
        self.register_buffer('eye_mask_H', torch.eye(2 * self.dim_internal + self.dim_nl))
        self.register_buffer('eye_mask_w', torch.eye(self.dim_nl))

        # initialize internal state
        if internal_state_init is None:
            if y_init is None:
                self.x = torch.zeros(1, 1, self.dim_internal)
            else:
                y_init = y_init.reshape(1, -1)
                self.x = torch.linalg.lstsq(self.C2, y_init.to(self.C2.device).squeeze(1).T)[0].T
        else:
            assert isinstance(internal_state_init, torch.Tensor)
            self.x = internal_state_init.reshape(1, 1, self.dim_internal)
        self.register_buffer('x_init', self.x.detach().clone())
        self.register_buffer('y_init', F.linear(self.x_init, self.C2))

    def _update_model_param(self):
        """
        Update non-trainable matrices according to the REN formulation to preserve contraction.
        """
        # dependent params
        H = torch.matmul(self.X.T, self.X) + self.epsilon * self.eye_mask_H
        h1, h2, h3 = torch.split(H, [self.dim_internal, self.dim_nl, self.dim_internal], dim=0)
        H11, H12, H13 = torch.split(h1, [self.dim_internal, self.dim_nl, self.dim_internal], dim=1)
        H21, H22, _ = torch.split(h2, [self.dim_internal, self.dim_nl, self.dim_internal], dim=1)
        H31, H32, H33 = torch.split(h3, [self.dim_internal, self.dim_nl, self.dim_internal], dim=1)
        P = H33

        # nn state dynamics
        self.F = H31
        self.B1 = H32

        # nn output
        self.E = 0.5 * (H11 + self.contraction_rate_lb * P + self.Y - self.Y.T)
        self.E_inv = self.E.inverse()

        # v signal for strictly acyclic REN
        self.Lambda = 0.5 * torch.diag(H22)
        self.D11 = -torch.tril(H22, diagonal=-1)
        self.C1 = -H21

    def forward(self, u_in):
        """
        Forward pass of REN.

        Args:
            u_in (torch.Tensor): Input with the size of (batch_size, 1, self.dim_in).

        Return:
            y_out (torch.Tensor): Output with (batch_size, 1, self.dim_out).
        """
        # update non-trainable model params
        self._update_model_param()

        batch_size = u_in.shape[0]

        w = torch.zeros(batch_size, 1, self.dim_nl, device=u_in.device)

        # update each row of w using Eq. (8) with a lower triangular D11
        for i in range(self.dim_nl):
            #  v is element i of v with dim (batch_size, 1)
            v = F.linear(self.x, self.C1[i, :]) + F.linear(w, self.D11[i, :]) + F.linear(u_in, self.D12[i,:])
            w = w + (self.eye_mask_w[i, :] * torch.tanh(v / self.Lambda[i])).reshape(batch_size, 1, self.dim_nl)

        # compute next state using Eq. 18
        self.x = F.linear(F.linear(self.x, self.F) + F.linear(w, self.B1) + F.linear(u_in, self.B2), self.E_inv)

        # compute output
        y_out = F.linear(self.x, self.C2) + F.linear(w, self.D21) + F.linear(u_in, self.D22)
        return y_out

    def reset(self):
        self.x = self.x_init  # reset the REN state to the initial value


    def run(self, u_in):
        """
        Runs the forward pass of REN for a whole input sequence of length horizon.

        Args:
            u_in (torch.Tensor): Input with the size of (batch_size, horizon, self.dim_in).

        Return:
            y_out (torch.Tensor): Output with (batch_size, horizon, self.dim_out).
        """

        self.reset()
        y_log = self.y_init.detach().clone().repeat(u_in.shape[0], 1, 1)
        for t in range(u_in.shape[1] - 1):
            y_log = torch.cat((y_log, self.forward(u_in[:, t:t + 1, :])), 1)
        # note that the last input is not used
        return y_log

    # init trainable params
    def _init_trainable_params(self, initialization_std):
        for training_param_name in self.training_param_names:  # name of one of the training params, e.g., X
            # read the defined shapes of the selected training param, e.g., X_shape
            shape = getattr(self, training_param_name + '_shape')
            # define the selected param (e.g., self.X) as nn.Parameter
            setattr(self, training_param_name, nn.Parameter((torch.randn(*shape) * initialization_std)))

    # setters and getters
    def get_parameter_shapes(self):
        param_dict = OrderedDict(
            (name, getattr(self, name).shape) for name in self.training_param_names
        )
        return param_dict

    def get_named_parameters(self):
        param_dict = OrderedDict(
            (name, getattr(self, name)) for name in self.training_param_names
        )
        return param_dict

    def __call__(self, u_in):
        return self.run(u_in)


# REN WITH GIVEN GAMMA
class REN_IQC_gamma(nn.Module):
    """
    Acyclic contractive recurrent equilibrium network, following the paper:
    "Recurrent equilibrium networks: Flexible dynamic models with guaranteed
    stability and robustness, Revay M et al. ."

    The mathematical model of RENs relies on an implicit layer embedded in a recurrent layer.
    The model is described as,

                    [  E . x_t+1 ]  =  [ F    B_1  B_2   ]   [  x_t ]   +   [  b_x ]
                    [  Λ . v_t   ]  =  [ C_1  D_11  D_12 ]   [  w_t ]   +   [  b_w ]
                    [  y_t       ]  =  [ C_2  D_21  D_22 ]   [  u_t ]   +   [  b_u ]

    where E is an invertible matrix and Λ is a positive-definite diagonal matrix. The model parameters
    are then {E, Λ , F, B_i, C_i, D_ij, b} which form a convex set according to the paper.

    NOTE: REN has input "u", output "y", and internal state "x". When used in closed-loop,
          the REN input "u" would be the noise reconstruction ("w") and the REN output ("y")
          would be the input to the plant. The internal state of the REN ("x") should not be mistaken
          with the internal state of the plant.
    """

    def __init__(
        self, dim_in: int, dim_out: int, dim_internal: int,
        dim_nl: int, internal_state_init = None, y_init = None,
        initialization_std: float = 0.5, pos_def_tol: float = 0.001, gammat=None, QR_fun=None, IQC_type='l2_gain',
        Q=None, R=None, S=None, device='cpu'
    ):
        """
        Args:
            dim_in (int): Input (u) dimension.
            dim_out (int): Output (y) dimension.
            dim_internal (int): Internal state (x) dimension. This state evolves with contraction properties.
            dim_nl (int): Dimension of the input ("v") and ouput ("w") of the nonlinear static block.
            initialization_std (float, optional): Weight initialization. Set to 0.1 by default.
            internal_state_init (torch.Tensor or None, optional): Initial condition for the internal state. Defaults to 0 when set to None.
            epsilon (float, optional): Positive and negligible scalar to force positive definite matrices.
            gamma (float, optional):  Defaults to 1.
        """
        super().__init__()
        
        self.device = device

        # set dimensions
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_internal = dim_internal
        self.dim_nl = dim_nl

        # set functionalities

        self.IQC_type = IQC_type
        self.QR_fun = QR_fun
        self.Q = Q
        self.R = R
        self.S = S
        

        # auxiliary elements
        self.epsilon = pos_def_tol

        # # # free parameters except for D22
        # define matrices shapes
        # auxiliary matrices
        self.X_shape = (2 * dim_internal + dim_nl, 2 * dim_internal + dim_nl)
        self.Y_shape = (dim_internal, dim_internal)
        # nn state dynamics
        self.B2_shape = (dim_internal, dim_in)
        # nn output
        self.C2_shape = (dim_out, dim_internal)
        self.D21_shape = (dim_out, dim_nl)
        self.D22_shape = (dim_out, dim_in)


        # v signal
        self.D12_shape = (dim_nl, dim_in)

        self.s = np.max((dim_in, dim_out))
        # self.X3_shape = (self.s, self.s)
        # self.Y3_shape = (self.s, self.s)
        # self.Z3_shape = (abs(dim_out - dim_in), min(dim_out, dim_in))

        self.gamma_shape = (1, 1)
        self.gammat = gammat
        self.device = device

        # define trainable params
        self.training_param_names = ['X', 'Y', 'B2', 'C2', 'D21', 'D12']
        if self.gammat is None:
            self.training_param_names.append('gamma')
        else:
            self.gamma = gammat

        # initialize trainable parameters
        self._init_trainable_params(initialization_std)

        # register buffers (non-trainable masks)
        self.register_buffer('eye_mask_min', torch.eye(min(dim_in, dim_out), device=device))
        self.register_buffer('eye_mask_dim_in', torch.eye(dim_in, device=device))
        self.register_buffer('eye_mask_dim_out', torch.eye(dim_out, device=device))
        self.register_buffer('eye_mask_dim_state', torch.eye(dim_internal, device=device))
        self.register_buffer('eye_mask_H', torch.eye(2 * dim_internal + dim_nl, device=device))
        self.register_buffer('zeros_mask_S', torch.zeros(dim_in, dim_out, device=device))
        self.register_buffer('zeros_mask_Q', torch.zeros(dim_out, dim_out, device=device))
        self.register_buffer('zeros_mask_R', torch.zeros(dim_in, dim_in, device=device))
        self.register_buffer('zeros_mask_so', torch.zeros(dim_internal, dim_out, device=device))
        self.register_buffer('eye_mask_w', torch.eye(dim_nl, device=device))

        # initialize internal state
        if internal_state_init is not None:
            assert isinstance(internal_state_init, torch.Tensor)
            self.x = internal_state_init.reshape(1, 1, dim_internal)
        elif y_init is not None:
            y_init = y_init.reshape(1, -1)
            self.C2 = nn.Parameter(self.C2.data)  # C2 already inizializzato come Parameter
            self.x = torch.linalg.lstsq(self.C2, y_init.squeeze(1).T)[0].T.unsqueeze(0).unsqueeze(0)
        else:
            self.x = torch.zeros(1, 1, dim_internal)

        # register initial state buffers
        self.register_buffer('x_init', self.x.detach().clone())
        y_init_calc = F.linear(self.x_init, self.C2)
        y_init_calc = y_init_calc.view(1, 1, self.dim_out)
        self.register_buffer('y_init', y_init_calc)

        # move everything to device
        self.to(device)

        # esplicito solo stato interno (già incluso in buffers), ma per sicurezza
        self.x = self.x.to(device)
        

    def _update_model_param(self):
        """
        Update non-trainable matrices according to the REN formulation to preserve IQC.
        """
        dim_internal, dim_nl, dim_in, dim_out = self.dim_internal, self.dim_nl, self.dim_in, self.dim_out

        if self.QR_fun is not None:
            self.Q, self.R, self.S = self.QR_fun(self.gamma, dim_in, dim_out, self.IQC_type)
        else:
            # incremental L2 gain constraints (default)
            self.Q = (-1 / self.gamma) * torch.eye(dim_out, dim_out)  # -1/gamma * I
            self.R = self.gamma * torch.eye(dim_in, dim_in)  # gamma * I
            self.S = torch.zeros(dim_out, dim_in)  # 0


        # M = F.linear(self.X3.T, self.X3.T) + self.Y3 - self.Y3.T + F.linear(self.Z3.T,
        #                                                                     self.Z3.T) + self.epsilon * self.eye_mask_min
        # if self.dim_out >= self.dim_in:
        #     N = torch.vstack((F.linear(self.eye_mask_dim_in - M,
        #                                torch.inverse(self.eye_mask_dim_in + M).T),
        #                       -2 * F.linear(self.Z3, torch.inverse(self.eye_mask_dim_in + M).T)))
        # else:
        #     N = torch.hstack((F.linear(torch.inverse(self.eye_mask_dim_out + M),
        #                                (self.eye_mask_dim_out - M).T),
        #                       -2 * F.linear(torch.inverse(self.eye_mask_dim_out + M), self.Z3)))

        # Lq = torch.linalg.cholesky(-self.Q).T
        #Lr = torch.linalg.cholesky(self.R - torch.matmul(self.S, torch.matmul(torch.inverse(self.Q), self.S.T))).T

        # self.D22 = -torch.matmul(torch.inverse(self.Q), self.S.T) + torch.matmul(torch.inverse(Lq), torch.matmul(N, Lr))

        # Calculate psi_r:
        # R_cal = self.R + torch.matmul(self.S, self.D22) + torch.matmul(self.S, self.D22).T + torch.matmul(self.D22.T,
        #                                                                                                   torch.matmul(
        #                                                                                                       self.Q,
        #                                                                                                       self.D22))
        # R_cal_inv = torch.inverse(R_cal)
        # C2_cal = torch.matmul(torch.matmul(self.D22.T, self.Q) + self.S, self.C2)
        # D21_cal = torch.matmul(torch.matmul(self.D22.T, self.Q) + self.S, self.D21) - self.D12.T
        # vec_r = torch.cat((C2_cal.T, D21_cal.T, self.B2), dim=0)
        # psi_r = torch.matmul(vec_r, torch.matmul(R_cal_inv, vec_r.T))
        # Calculate psi_q:
        # vec_q = torch.cat((self.C2.T, self.D21.T, self.zeros_mask_so), dim=0)
        # psi_q = torch.matmul(vec_q, torch.matmul(self.Q, vec_q.T))
        # Create H matrix:
        H = torch.matmul(self.X.T, self.X) + self.epsilon * self.eye_mask_H # + psi_r - psi_q
        h1, h2, h3 = torch.split(H, [dim_internal, dim_nl, dim_internal], dim=0)
        H11, H12, H13 = torch.split(h1, [dim_internal, dim_nl, dim_internal], dim=1)
        H21, H22, _ = torch.split(h2, [dim_internal, dim_nl, dim_internal], dim=1)
        H31, H32, H33 = torch.split(h3, [dim_internal, dim_nl, dim_internal], dim=1)
        self.P_cal = H33
        # NN state dynamics:
        self.F = H31
        self.B1 = H32
        # NN output:
        self.E = 0.5 * (H11 + self.P_cal + self.Y - self.Y.T)
        self.E_inv = self.E.inverse()
        # v signal:  [Change the following 2 lines if we don't want a strictly acyclic REN!]
        self.Lambda = 0.5 * torch.diag(H22)
        self.D11 = -torch.tril(H22, diagonal=-1)
        self.C1 = -H21
        # Matrix P
        #self.P = torch.matmul(self.E.T, torch.matmul(torch.inverse(self.P_cal), self.E))

    def forward(self, u_in):
        """
        Forward pass of REN.

        Args:
            u_in (torch.Tensor): Input with the size of (batch_size, 1, self.dim_in).

        Return:
            y_out (torch.Tensor): Output with (batch_size, 1, self.dim_out).
        """
        # update non-trainable model params
        self._update_model_param()

        batch_size = u_in.shape[0]

        w = torch.zeros(batch_size, 1, self.dim_nl, device=u_in.device)

        # update each row of w using Eq. (8) with a lower triangular D11
        for i in range(self.dim_nl): # does the non linear one by one but all together the multiple batches
            #  v is element i of v with dim (batch_size, 1)
            v = F.linear(self.x, self.C1[i, :]) + F.linear(w, self.D11[i, :]) + F.linear(u_in, self.D12[i, :])
            w = w + (self.eye_mask_w[i, :] * torch.tanh(v / self.Lambda[i])).reshape(batch_size, 1, self.dim_nl) # w dim [batch 1 dim_nl]


        # compute next state using Eq. 18
        self.x = F.linear(F.linear(self.x, self.F) + F.linear(w, self.B1) + F.linear(u_in, self.B2), self.E_inv)

        # compute output
        y_out = F.linear(self.x, self.C2) + F.linear(w, self.D21) # + F.linear(u_in, self.D22)
        return y_out

    def reset(self, x0=None, batch_size=None):
        """
        Reset compatibile con ClosedLoopSystem.
        Se x0 è fornito (shape (batch, 1, dim_internal) o (1, 1, dim_internal)),
        lo utilizza come stato; altrimenti usa self.x_init (replicato per batch se necessario).
        """
        if x0 is not None:
            # accetta sia (1,1,dim) sia (batch,1,dim)
            self.x = x0.clone().to(self.device)
        else:
            # default behaviour: reset to x_init, replicate to batch_size se richiesto
            if batch_size is None:
                self.x = self.x_init.clone().to(self.device)
            else:
                self.x = self.x_init.detach().clone().repeat(batch_size, 1, 1).to(self.device)

    def y0_from_x0(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Given an internal state x0 (batch,1,dim_internal) returns the corresponding y0.
        We assume initial w = 0 and u = 0, so y0 = C2 @ x0 (linear part).
        """
        return F.linear(x0, self.C2)


    def run(self, u_in):
        """
        Runs the forward pass of REN for a whole input sequence of length horizon.

        Args:
            u_in (torch.Tensor): Input with the size of (batch_size, horizon, self.dim_in).

        Return:
            y_out (torch.Tensor): Output with (batch_size, horizon, self.dim_out).
        """

        self.reset()
        y_log = self.y_init.detach().clone().repeat(u_in.shape[0], 1, 1)
        for t in range(u_in.shape[1] - 1):
            y_log = torch.cat((y_log, self.forward(u_in[:, t:t + 1, :])), 1)
        # note that the last input is not used
        return y_log

    # init trainable params
    def _init_trainable_params(self, initialization_std):
        for training_param_name in self.training_param_names:  # name of one of the training params, e.g., X
            # read the defined shapes of the selected training param, e.g., X_shape
            shape = getattr(self, training_param_name + '_shape')
            # define the selected param (e.g., self.X) as nn.Parameter
            if training_param_name == 'gamma':
                initialization_std = 3
            setattr(self, training_param_name, nn.Parameter((torch.randn(*shape) * initialization_std)))

    # setters and getters
    def get_parameter_shapes(self):
        param_dict = OrderedDict(
            (name, getattr(self, name).shape) for name in self.training_param_names
        )
        return param_dict

    def get_named_parameters(self):
        param_dict = OrderedDict(
            (name, getattr(self, name)) for name in self.training_param_names
        )
        return param_dict

    def __call__(self, u_in):
        return self.run(u_in)
    
    
    
class DualREN(nn.Module):
    """
    A wrapper module combining two REN models.
    It takes two separate inputs and outputs the difference:
        y_hat = REN_0(u0) - REN_1(u1)
    """

    def __init__(self, REN_0: nn.Module, REN_1: nn.Module, device='cpu'):
        super().__init__()
        
        # buffer for the cobined internal state
        self.x = None
        self.y = None

        self.REN_0 = REN_0.to(device)
        self.REN_1 = REN_1.to(device)
        self.device = device
        self.to(device)
        


    def forward(self, u0_in: torch.Tensor, u1_in: torch.Tensor) -> torch.Tensor:
        """
        Esegue UN SOLO passo temporale di entrambe le REN e restituisce la differenza tra i loro output.
        Args:
            u0_in: input per REN_0 (batch, 1, dim_in_0)
            u1_in: input per REN_1 (batch, 1, dim_in_1)
        Returns:
            y_out: output combinato (batch, 1, dim_out)
        """
        y0 = self.REN_0.forward(u0_in)
        y1 = self.REN_1.forward(u1_in)
        y_out = y0 - y1

        # Aggiorna lo stato combinato
        self.x = torch.cat((self.REN_0.x, self.REN_1.x), dim=-1)
        return y_out
    
    def run(self, u0_seq: torch.Tensor, u1_seq: torch.Tensor) -> torch.Tensor:
        """
        Esegue il forward delle due REN su tutta una sequenza temporale.
        Usa il metodo forward() interno passo per passo.
        Args:
            u0_seq: input sequence per REN_0 (batch, time, dim_in_0)
            u1_seq: input sequence per REN_1 (batch, time, dim_in_1)
        Returns:
            y_seq: output sequence combinato (batch, time, dim_out)
        """
        self.reset()  # reset degli stati iniziali

        # il primo output parte da y_init
        y_init = self.REN_0.y_init.detach().clone().repeat(u0_seq.shape[0], 1, 1) \
                - self.REN_1.y_init.detach().clone().repeat(u1_seq.shape[0], 1, 1)
        y_seq = [y_init]

        # loop temporale
        for t in range(u0_seq.shape[1] - 1):
            y_t = self.forward(u0_seq[:, t:t+1, :], u1_seq[:, t:t+1, :])
            y_seq.append(y_t)

        y_seq = torch.cat(y_seq, dim=1)
        return y_seq
    
    def y0_from_x0(self, x0: torch.Tensor) -> torch.Tensor:
        """Converte lo stato interno combinato in y0, come differenza delle due REN."""
        # suddividi x0 nelle due parti corrispondenti
        d0 = self.REN_0.x.shape[-1]
        x0_0 = x0[..., :d0]
        x0_1 = x0[..., d0:]

        y0_0 = self.REN_0.y0_from_x0(x0_0.to(self.REN_0.x.device))
        y0_1 = self.REN_1.y0_from_x0(x0_1.to(self.REN_1.x.device))
        return y0_0 - y0_1
    
    def __call__(self, u0_seq: torch.Tensor, u1_seq: torch.Tensor) -> torch.Tensor:
        """Alias per run(), per coerenza con le singole REN"""
        return self.run(u0_seq, u1_seq)

    def reset(self):
        """Reset both REN internal states."""
        self.REN_0.reset()
        self.REN_1.reset()
        self.x = torch.cat((self.REN_0.x.to(self.device), self.REN_1.x.to(self.device)), dim=-1)
        self.y = torch.cat((self.REN_0.y_init.to(self.device), self.REN_1.y_init.to(self.device)), dim=-1)


