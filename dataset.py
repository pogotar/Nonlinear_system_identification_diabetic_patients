import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from torch.utils.data import Dataset
from ParametersFunctions import MinMaxScalerTorch




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
        
    the output data is not normalized
    """
    # Total amount
    amount = bolus + basal / 60 * ts_measurement  # UI

    # Pump params
    pumpSaturation = pumpParameter.saturationMax  # UI
    quanto = pumpParameter.quantum  # UI

    # Rounding to closest quanto taking previous error into account
    if quanto == 0:
        # Quanto=0 --> Ideal pump (for software test)
        bolus = amount
    else:
        bolus = quanto * round((amount + saturation_error) / quanto)  # UI

    # Store remaining bolus if bolus > saturationMax
    if bolus > pumpSaturation:
        bolus = pumpSaturation

    saturation_error = amount + saturation_error - bolus  # UI

    basal = 0

    return bolus, basal, saturation_error


def round_to_5min(dt):
    # Get total minutes since midnight
    total_minutes = dt.hour * 60 + dt.minute
    # Round to nearest 5
    rounded_minutes = round(total_minutes / 5) * 5
    # Create new datetime with rounded time
    return dt.replace(hour=rounded_minutes // 60, minute=rounded_minutes % 60, second=0, microsecond=0)

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

def rwgn_at_time(t_index, seed, mu, sigma):
    def fract(x): return x - np.floor(x)
    u1 = fract(np.sin(12.9898*(seed + t_index))*43758.5453)
    u2 = fract(np.sin(78.233*(seed + t_index))*43758.5453)

    z = np.sqrt(-2*np.log(u1)) * np.cos(2*np.pi*u2)
    r = mu + sigma*z
    return r



class LoadData(Dataset):
    def __init__(self, patient, data_path, use_noise, train_size):

        self.patient = patient
        self.data_path = data_path

        grams_of_hypo = 15
        duration_of_hypo = 15

        # Load data
        loaded = sio.loadmat(f'{data_path}/s#adult#{patient:03d}.mat')

        # Insulin suggestions saturated (effectively given by the pump)
        I_sat_real = loaded['injection']['signals'][0, 0]['values'][0, 0][1::5] / 6000  # now in U

        # Glucose from PID
        CGM_real = loaded['CGM']['signals'][0, 0]['values'][0, 0][1::5].flatten()
        G_real = loaded['G']['signals'][0, 0]['values'][0, 0][1::5].flatten()


        true_carbs_hypo_redone_05 = loaded['carb_intake']['signals'][0, 0]['values'][0, 0][1:-1:2].flatten()
        MH_real = true_carbs_hypo_redone_05.reshape(-1, 5).sum(axis=1)
        M_real = np.where(
            MH_real == grams_of_hypo * 1000,
            0,
            MH_real
        )




        hypoTreatment = loaded['hypoTreatment'][0]

        ########################################################################################

        # Initialize arrays
        I_rec = np.zeros(len(CGM_real))
        basal_tot = np.zeros(len(CGM_real))
        basal_tot_p_noise = np.zeros(len(CGM_real))
        I_sat_rec_p_noise = np.zeros(len(CGM_real))
        noise_tot = np.zeros(len(CGM_real))
        I_sat_rec = np.zeros(len(CGM_real))
        saturation_error = np.zeros(len(CGM_real))
        saturation_error_2 = np.zeros(len(CGM_real))
        I_rec_p_noise = np.zeros(len(CGM_real))

        # PID parameters
        class PID_par:
            K_p = -0.0665
            K_i = -1.9342e-4
            K_d = -2.0922
            tsController = 5
            ref = 110
            intSatLower = 30
            intSatPerc = 1.5

        class patient_par:
            CR_tuned = np.array([0.5, 1.8, 1.3, 4, 1, 0.8, 0.5, 1.2, 0.9, 2.2])

        conversion_index = 0.007

        # Load CR values
        CRtv_data = sio.loadmat('./data/CRtv.mat')
        CR_values_not_norm = CRtv_data['CRtv']['values'][0, patient - 1]

        # Pump parameters
        class pumpParameter:
            quantum = 0.05
            saturationMax = 12

        integral_duration = 12 * 2
        ts_measurement = 5


        for i in range(0, len(CGM_real)):

            # Time of Day calculation
            ToD = i * 5 % 1440

            ########################################################

            if i < 5:
                bolus = 0
            else:
                glucose_PID = CGM_real[0:i + 1]

                # Calculate candidate error with capping
                start_idx = max(0, i - integral_duration)
                candidate_error = PID_par.ref - glucose_PID[start_idx:i + 1]
                candidate_error_capped = np.maximum(candidate_error, PID_par.ref - 140)
                e_sum = np.sum(candidate_error_capped)

                CR_index = np.where(np.arange(0, 289) <= round((ToD % 1) * 1440 / 5))[0][-1]
                CR_now = CR_values_not_norm[0, CR_index] / np.max(CR_values_not_norm)

                # Compute Kp, Kd, and Ki
                K_p = PID_par.K_p / CR_now
                K_d = PID_par.K_d / PID_par.tsController / CR_now
                K_i = PID_par.K_i * PID_par.tsController / CR_now

                # Compute errors
                e = PID_par.ref - glucose_PID[i]
                e_m1 = PID_par.ref - glucose_PID[i - 1]

                # Compute PID control action (delta_P + delta_I + delta_D)
                delta_P = K_p * e
                delta_D = K_d * (e - e_m1)
                delta_I = K_i * e_sum

                # Inizializza le variabili
                booster_d_neg = 1
                booster_d_pos = 1

                if len(glucose_PID) > 3:
                    booster_d_neg, booster_d_pos = function_booster_d(glucose_PID)

                basal_PID = delta_P + delta_D * booster_d_pos + delta_I

                CR_tuned = patient_par.CR_tuned[patient - 1]

                bolus = basal_PID * conversion_index * booster_d_neg * CR_tuned


            ###############################calculate basal##########################################################
            indexCurrTbr = 0
            basal_values = loaded['basal_pattern_original']['values'][0, 0].flatten()
            basal_time = loaded['basal_pattern_original']['time'][0, 0].flatten()
            # Trova l’ultimo indice dove time <= ToD
            indices = np.where(basal_time <= ToD)[0]
            if len(indices) > 0:
                currentBasal = basal_values[indices[-1]]  # ultimo valore valido
            else:
                currentBasal = basal_values[-1]  # se nessun valore valido, prendi l’ultimo
            basal = currentBasal

            I_rec[i] = basal / 60 * ts_measurement + bolus

            ##################################################################################################
            if use_noise:
                mu = 0
                sigma = np.min(basal_values) / 60 * ts_measurement * 0.4

                if i < 5:
                    rwgn_instantaneous = 0
                else:
                    rwgn_instantaneous = rwgn_at_time(ToD, 42, mu, sigma)

                I_rec_p_noise[i] = I_rec[i] + rwgn_instantaneous


            #####################################################################

            if i - 1 < 0:
                current_saturation_error = 0
                current_saturation_error_2 = 0
            else:
                current_saturation_error = saturation_error[i - 1]
                current_saturation_error_2 = saturation_error_2[i - 1]
            # Saturation of pump and transformation
            bolus_sat, basal_sat, saturation_error[i] = saturation_of_pump_and_trasformation(
                I_rec[i], 0, ts_measurement, pumpParameter, current_saturation_error
            )
            bolus_sat_2, basal_sat_2, saturation_error_2[i] = saturation_of_pump_and_trasformation(
                I_rec_p_noise[i], 0, ts_measurement, pumpParameter, current_saturation_error_2
            )


            I_sat_rec[i] = basal_sat + bolus_sat
            I_sat_rec_p_noise[i] = basal_sat_2 + bolus_sat_2


        # Ensure non-negative values
        # I_sat_rec = np.maximum(I_sat_rec, 0) # questi maximum non sono sicuro che non facciano ninete ma controlare

        #### true hypo  ##############################################################################################

        hypoTreatment = loaded['hypoTreatment'][0]

        if len(hypoTreatment) < 2:
            hypo_time = np.array([])
            true_hypo_amount = np.array([])
        else:
            hypo_time = np.concatenate([hypoTreatment[i]['time'] for i in range(1, len(hypoTreatment))]).flatten()
            true_hypo_amount = np.concatenate([np.asarray(hypoTreatment[i]['amount']).astype(np.float64).ravel()for i in range(1, len(hypoTreatment))])

            # Convert to relative time
            timeCL = loaded['timeCL'][0, 0]
            # Convert MATLAB datenum to Python datetime
            # MATLAB datenum: days since January 0, 0000
            # Python datetime: needs conversion
            matlab_epoch = datetime(1, 1, 1)

            datetime_hypo = [round_to_5min(matlab_epoch + timedelta(days=float(t) - 367)) for t in hypo_time.flatten()]

            datetime_today_base = matlab_epoch + timedelta(days=float(timeCL) - 367)
            datetime_today = datetime_today_base.replace(hour=0, minute=0, second=0, microsecond=0)

            hypo_rel_time_single = np.array([(dt - datetime_today).total_seconds() / 60 / 5 for dt in datetime_hypo])
            hypo_rel_time_single = np.floor(hypo_rel_time_single / 5) * 5

            hypo_rel_time = np.concatenate([np.arange(x, x + 3) for x in hypo_rel_time_single])

        # MH_real = M_real.copy()
        H_real = np.zeros(len(M_real))

        if len(true_hypo_amount) > 0:
            H_real[hypo_rel_time.astype(int)] = true_hypo_amount[0] * 1000 / duration_of_hypo
        #     MH_real[hypo_rel_time.astype(int)] = true_hypo_amount[0] * 1000 / duration_of_hypo

        #### hypo from data ##############################################################################################
        hypo_amount_from_data = grams_of_hypo * 1000 / duration_of_hypo

        index_G_under_70 = np.where(G_real <= 70)[0] - 1  # lo zero elimina la tupla

        H_rec = np.zeros(len(M_real))
        MH_rec = M_real.copy()

        if len(index_G_under_70) > 2:
            H_rec = np.zeros(len(M_real))
            index_ipo_from_data_sinle = []
            start = 0  # indice di inizio blocco
            step = 5
            for i in range(1, len(index_G_under_70)):
                # se c'è una discontinuità, finisce un blocco
                if index_G_under_70[i] != index_G_under_70[i - 1] + 1:
                    # salva indici ogni 6 elementi a partire dall'inizio
                    for j in range(start, i, step):
                        index_ipo_from_data_sinle.append(index_G_under_70[j])
                    start = i  # nuovo blocco inizia qui

            # non dimenticare l'ultimo blocco
            for j in range(start, len(index_G_under_70), step):
                index_ipo_from_data_sinle.append(index_G_under_70[j])

            index_ipo_from_data = np.concatenate([np.arange(x, x + 3) for x in index_ipo_from_data_sinle])

            H_rec[index_ipo_from_data] = hypo_amount_from_data  # ipoglicemia trattata con 15g di carboidrati
            MH_rec [index_ipo_from_data] = hypo_amount_from_data # !!!

        I_sat_real = I_sat_real[:,0]
        
        #---------normalization-----------------
        scaler_meal       = MinMaxScalerTorch()
        scaler_insulin  = MinMaxScalerTorch() # For R and I_sat
        scaler_glucose      = MinMaxScalerTorch()

        scaler_meal.compute_norm_indexes(torch.from_numpy(MH_real[:train_size]).float())
        scaler_insulin.compute_norm_indexes(torch.from_numpy(I_sat_real[:train_size]).float())
        scaler_glucose.compute_norm_indexes(torch.from_numpy(CGM_real[:train_size]).float())
        
        
        #---------- store normalized data------------------
        self.CGM = scaler_glucose.normalize(torch.from_numpy(CGM_real).float())
        self.G = scaler_glucose.normalize(torch.from_numpy(G_real).float())


        self.M = scaler_meal.normalize(torch.from_numpy(M_real).float())

        self.H = scaler_meal.normalize(torch.from_numpy(H_real).float())
        self.H_rec = scaler_meal.normalize(torch.from_numpy(H_rec).float())

        self.MH = scaler_meal.normalize(torch.from_numpy(MH_real).float())
        self.MH_rec = scaler_meal.normalize(torch.from_numpy(MH_rec).float())


        self.I_rec = scaler_insulin.normalize(torch.from_numpy(I_rec).float())
        if use_noise:
            self.I_sat_rec = scaler_insulin.normalize(torch.from_numpy(I_sat_rec_p_noise).float())
        else:
            self.I_sat_rec = scaler_insulin.normalize(torch.from_numpy(I_sat_rec).float())

        self.I_sat = scaler_insulin.normalize(torch.from_numpy(I_sat_real).float())


        self.R = scaler_insulin.normalize(torch.from_numpy(I_sat_real - I_rec).float()) # I_rec is before saturation and noise

        self.time = torch.arange(0, len(CGM_real))

        class Data:
            def __init__(self, time, values):
                self.time = time
                self.values = values

        basal_vec = Data(torch.from_numpy(basal_time).float(), torch.from_numpy(basal_values).float())

        self.basal_vec = basal_vec
        
        self.scaler_meal = scaler_meal
        self.scaler_insulin = scaler_insulin
        self.scaler_glucose = scaler_glucose


        # how many samples are present
    def __len__(self):
            return len(self.CGM)

        # how to pick the indexes during the training
    def __getitem__(self, idx):

        MH = self.MH[idx]
        I_rec = self.I_rec[idx]
        R = self.R[idx]
        I_sat = self.I_sat[idx]
        CGM = self.CGM[idx]
        time = self.time[idx]
        return MH.float(), I_rec.float(), R.float(),I_sat.float(), CGM.float(), time.float()







