import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np  # linear algebra
    
class DualSSM(nn.Module):
    """
    A wrapper module combining two REN models.
    It takes two separate inputs and outputs the difference:
        y_hat = REN_0(u0) - REN_1(u1)
    """

    def __init__(self, SSM_0: nn.Module, SSM_1: nn.Module, device='cpu'):
        super().__init__()
        
        # buffer for the cobined internal state
        self.SSN_0_x = None
        self.SSN_1_x = None

        self.SSM_0 = SSM_0.to(device)
        self.SSM_1 = SSM_1.to(device)
        self.device = device
        self.to(device)
        
        self.SSM_0.dim_in = SSM_0.d_input
        self.SSM_1.dim_in = SSM_1.d_input
        
        


    def forward(self, u0_in: torch.Tensor, u1_in: torch.Tensor) -> torch.Tensor:
        """
        Esegue UN SOLO passo temporale di entrambe le REN e restituisce la differenza tra i loro output.
        Args:
            u0_in: input per REN_0 (batch, 1, dim_in_0)
            u1_in: input per REN_1 (batch, 1, dim_in_1)
        Returns:
            y_out: output combinato (batch, 1, dim_out)
        """
        y0, self.SSM_0_x = self.SSM_0.forward(u = u0_in, state = self.SSM_0_x)
        y1, self.SSM_1_x = self.SSM_1.forward(u = u1_in, state = self.SSM_1_x)
        y_out = y0 - y1

        # Aggiorna lo stato combinato
        # self.x = torch.cat((self.SSM_0.x, self.SSM_1.x), dim=-1)
        
        return y_out, y0, y1
    
    def run(self, u0_seq: torch.Tensor, u1_seq: torch.Tensor, y0 = None) -> torch.Tensor:
        """
        Esegue il forward delle due REN su tutta una sequenza temporale.
        Usa il metodo forward() interno passo per passo.
        Args:
            u0_seq: input sequence per REN_0 (batch, time, dim_in_0)
            u1_seq: input sequence per REN_1 (batch, time, dim_in_1)
        Returns:
            y_seq: output sequence combinato (batch, time, dim_out)
        """
        # self.reset(y0 = y0)  # reset degli stati iniziali
        
        y_SSM_0, _ = self.SSM_0(u0_seq)
        y_SSM_1, _ = self.SSM_1(u1_seq)
 
        y_seq = y_SSM_0 - y_SSM_1

        return y_seq, y_SSM_0, y_SSM_1
    
    
    def __call__(self, u0_seq: torch.Tensor, u1_seq: torch.Tensor, y0 = None) -> torch.Tensor:
        """Alias per run(), per coerenza con le singole REN"""
        return self.run(u0_seq, u1_seq, y0 = y0)

    def reset(self):
        """Reset both REN internal states."""
        # in qualche modo è da separare e mandare separato
        
        
        self.SSM_0_x = None
        self.SSM_1_x = None
        
        
        
        
        

