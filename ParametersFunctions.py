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
    def rwgn_at_time(t_index, seed, mu, sigma):
        def fract(x): return x - np.floor(x)

        u1 = fract(np.sin(12.9898 * (seed + t_index)) * 43758.5453)
        u2 = fract(np.sin(78.233 * (seed + t_index)) * 43758.5453)

        z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        r = mu + sigma * z
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

