import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from torch.utils.data import Dataset

class Parameter:
    
    torch.set_default_dtype(torch.float32) 
    def __init__(self, patient):
        self.patient = patient
        self.PID_par = self.PID_par()
        self.Patient_par = self.Patient_par(patient)
        self.pumpParameter = self.pumpParameter()

    class PID_par:
        def __init__(self):
            self.K_p = -0.0665
            self.K_i = -1.9342e-4
            self.K_d = -2.0922
            self.tsController = 5.0
            self.ref = 110
            self.intSatLower = 30
            self.intSatPerc = 1.5
            self.conversion_index = 0.007
            self.integral_duration = 12 * 2
            self.ts_measurement = 5.0
            self.conversion_index = 0.007

    class Patient_par:
        def __init__(self, patient):
            self.CR_tuned = torch.tensor([0.5, 1.8, 1.3, 4, 1, 0.8, 0.5, 1.2, 0.9, 2.2], dtype=torch.float64)
            CRtv_data = sio.loadmat('./data/CRtv.mat')
            self.CR_values_not_norm = torch.from_numpy(CRtv_data['CRtv']['values'][0, patient - 1]).double()

    class pumpParameter:
        quantum = torch.tensor([0.05])
        saturationMax = torch.tensor([12.0])


class PID_functions:

    @staticmethod
    def saturation_of_pump_and_trasformation(bolus, basal, ts_measurement,
                                             pumpParameter, saturation_error):
        """
        Saturation of pump and transformation function

        Parameters:
        -----------
        bolus : float
            Bolus insulin amount
        basal : float
            Basal insulin rate
        ts_measurement : float
            Measurement time step
        pumpParameter : object
            Pump parameters (quantum, saturationMax)
        saturation_error : float
            Previous saturation error

        Returns:
        --------
        bolus : float
            Saturated bolus
        basal : float
            Saturated basal (always 0)
        saturation_error : float
            Updated saturation error
        """
        
        torch.set_default_dtype(torch.float32) 
        # Total amount
        amount = bolus + basal / 60 * ts_measurement  # UI

        # Pump params
        pumpSaturation = pumpParameter.saturationMax  # UI
        quanto = pumpParameter.quantum  # UI

        # Rounding to closest quanto taking previous error into account
        if torch.all(quanto == 0):
            # Quanto=0 --> Ideal pump (for software test)
            bolus = amount
        else:
            bolus = quanto * torch.round((amount + saturation_error) / quanto)  # UI

        # Store remaining bolus if bolus > saturationMax
        
        # Saturation check
        bolus = torch.minimum(bolus, pumpSaturation)

        saturation_error = amount + saturation_error - bolus  # UI

        basal = 0.0
        
        bolus = bolus.to(torch.float32)
        basal = torch.tensor(basal, dtype=torch.float32)        
        saturation_error = saturation_error.to(torch.float32)

        return bolus, basal, saturation_error
    
    @staticmethod
    def saturation_of_pump_and_trasformation_p(bolus, basal, ts_measurement,
                                             pumpParameter, saturation_error):
        """
        Saturation of pump and transformation function

        Parameters:
        -----------
        bolus : float
            Bolus insulin amount
        basal : float
            Basal insulin rate
        ts_measurement : float
            Measurement time step
        pumpParameter : object
            Pump parameters (quantum, saturationMax)
        saturation_error : float
            Previous saturation error

        Returns:
        --------
        bolus : float
            Saturated bolus
        basal : float
            Saturated basal (always 0)
        saturation_error : float
            Updated saturation error
        """
        
        torch.set_default_dtype(torch.float32) 
        
                
        if torch.isnan(bolus).any() or torch.isnan(saturation_error).any():
            print("Errore: rwgn_instantaneous contiene NaN")
            
        if torch.isinf(bolus).any() or torch.isinf(saturation_error).any():
            print("Errore: rwgn_instantaneous contiene Inf")
            
        # Total amount
        amount = bolus + basal / 60 * ts_measurement  # UI

        # Pump params
        pumpSaturation = pumpParameter.saturationMax  # UI
        quanto = pumpParameter.quantum  # UI
        
        if torch.isnan(bolus).any() or torch.isnan(saturation_error).any():
            print("Errore: rwgn_instantaneous contiene NaN")

        # Rounding to closest quanto taking previous error into account
        bolus = quanto * torch.round((amount + saturation_error) / quanto)  # UI

        # Store remaining bolus if bolus > saturationMax
        
        # Saturation check
        bolus = torch.minimum(bolus, pumpSaturation)

        saturation_error = amount + saturation_error - bolus  # UI

        
        bolus = bolus.to(torch.float32)
        # basal = basal.detach().clone().float() 
        saturation_error = saturation_error.to(torch.float32)
        
        if torch.isnan(bolus).any() or torch.isnan(saturation_error).any():
            print("Errore: rwgn_instantaneous contiene NaN")

        return bolus, basal, saturation_error

    @staticmethod
    def function_booster_d(glucose_PID):

        # Assicurati che ci siano almeno 4 elementi
        if len(glucose_PID) < 4:
            raise ValueError("glucose_PID deve contenere almeno 4 valori")

        d1 = glucose_PID[-1] - glucose_PID[-2]
        d2 = glucose_PID[-1] - glucose_PID[-3]
        d3 = glucose_PID[-1] - glucose_PID[-4]

        # ---- correttore_d_neg ----
        if d1 < 0 and d2 < 0 and d3 < 0:
            booster_d_neg = 0
        else:
            booster_d_neg = 1

        # ---- correttore_d_pos ----
        if d1 > 0 and d2 > 0 and d3 > 0:
            booster_d_pos = 1.1
        else:
            booster_d_pos = 1

        if d1 > 10 and d2 > 20:
            booster_d_pos = 2.5

        if d1 > 15 and d2 > 30:
            booster_d_pos = 3

        if d1 > 20 and d2 > 40:
            booster_d_pos = 4

        return booster_d_neg, booster_d_pos
    
    @staticmethod
    def function_booster_d_p(glucose_PID_batch):
        """
        glucose_PID_batch: torch.Tensor shape (batch, 25)
        return: booster_d_neg, booster_d_pos shape (batch, 1)
        """

        if glucose_PID_batch.shape[1] < 4:
            raise ValueError("Ogni batch deve avere almeno 4 valori")

        # Calcolo differenze sugli ultimi 4 valori
        d1 = glucose_PID_batch[:, -1] - glucose_PID_batch[:, -2]
        d2 = glucose_PID_batch[:, -1] - glucose_PID_batch[:, -3]
        d3 = glucose_PID_batch[:, -1] - glucose_PID_batch[:, -4]

        # ---- booster_d_neg ----
        booster_d_neg = torch.ones_like(d1)
        mask_neg = (d1 < 0) & (d2 < 0) & (d3 < 0)
        booster_d_neg[mask_neg] = 0

        # ---- booster_d_pos ----
        booster_d_pos = torch.ones_like(d1)
        mask_pos = (d1 > 0) & (d2 > 0) & (d3 > 0)
        booster_d_pos[mask_pos] = 1.1

        # Aggiornamento valori più alti
        booster_d_pos = torch.where((d1 > 10) & (d2 > 20), torch.tensor(2.5, device=glucose_PID_batch.device), booster_d_pos)
        booster_d_pos = torch.where((d1 > 15) & (d2 > 30), torch.tensor(3.0, device=glucose_PID_batch.device), booster_d_pos)
        booster_d_pos = torch.where((d1 > 20) & (d2 > 40), torch.tensor(4.0, device=glucose_PID_batch.device), booster_d_pos)

        # Aggiungi dimensione finale per shape (batch,1)
        return booster_d_neg.unsqueeze(1), booster_d_pos.unsqueeze(1)

    @staticmethod
    def rwgn_at_time(t_index, seed, mu, sigma):
        def fract(x): return x - np.floor(x)
        
        if isinstance(t_index, torch.Tensor):
            t_index = t_index.detach().cpu().numpy().astype(np.float64)

        u1 = fract(np.sin(12.9898 * (seed + t_index)) * 43758.5453)
        u2 = fract(np.sin(78.233 * (seed + t_index)) * 43758.5453)

        z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        r = mu + sigma * z
        
        r_np = r.numpy()
        if np.isinf(r_np).any() or np.isnan(r_np).any():
            print("Errore: rwgn_at_time contiene Inf o NaN")
        return r

    @staticmethod
    def calculate_basal(basal_vec, ToD):
        basal_values = basal_vec.values
        basal_time = basal_vec.time

        indices = np.where(basal_time <= ToD)[0]
        if len(indices) > 0:
            currentBasal = basal_values[indices[-1]]  # ultimo valore valido
        else:
            currentBasal = basal_values[-1]

        return currentBasal
    
    @staticmethod
    def calculate_basal_p(basal_time, basal_values, ToD):
        """
        basal_time: tensor (num_basal,)  -> valori dei tempi basal
        basal_values: tensor (num_basal,) -> valori basal
        ToD: tensor (batch, 1)           -> time of day per batch
        return: tensor (batch, 1)        -> current basal per batch
        """

        batch = ToD.shape[0]
        num_basal = basal_time.shape[0]

        # espandiamo per broadcasting: (batch, num_basal)
        ToD_exp = ToD.repeat(1, num_basal)       # (batch, num_basal)
        basal_time_exp = basal_time.unsqueeze(0).repeat(batch, 1)  # (batch, num_basal)

        # creiamo maschera: True se basal_time <= ToD
        mask = basal_time_exp <= ToD_exp         # (batch, num_basal)

        # indici dell'ultimo True per ogni batch
        mask_flipped = torch.flip(mask, dims=[1])
        last_idx = num_basal - 1 - mask_flipped.float().argmax(dim=1)  # (batch,)

        # otteniamo i valori
        currentBasal = basal_values[last_idx]      # (batch,)
        return currentBasal.unsqueeze(1)           # (batch,1)


class MinMaxScalerTorch:
    def __init__(self):
        """
        Min-Max normalizer for PyTorch tensors.
        Scales data to [0, 1] based on min and max of the input tensor.
        """
        self.params = {}

    def compute_norm_indexes(self, x: torch.Tensor):
        """
        Compute min and max values for normalization.

        Args:
            x (torch.Tensor): input tensor
        """
        self.params['low'] = torch.min(x)
        self.params['high'] = torch.max(x)



    def normalize(self, x: torch.Tensor):
        """
        Normalize tensor x to [0, 1].

        Args:
            x (torch.Tensor): input tensor

        Returns:
            x_norm (torch.Tensor): normalized tensor
        """
        low = self.params['low']
        high = self.params['high']
        return (x - low) / (high - low)

    def denormalize(self, x_norm: torch.Tensor):
        """
        Denormalize tensor from [0,1] back to original scale.

        Args:
            x_norm (torch.Tensor): normalized tensor

        Returns:
            x_orig (torch.Tensor): original scale tensor
        """
        low = self.params['low']
        high = self.params['high']
        return x_norm * (high - low) + low

