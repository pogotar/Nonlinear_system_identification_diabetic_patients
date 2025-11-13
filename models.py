from timeit import repeat
import torch
import torch.nn as nn
import numpy as np

torch.set_printoptions(precision=17)

class NonLinearController(nn.Module):


    torch.set_default_dtype(torch.float32)

    """Simulates the closed-loop system (Plant + Controller)."""

    def __init__(self, P, PF, basal_vec,  scaler_glucose, scaler_insulin, scaler_meal, use_noise=False):
        super().__init__()
        self.P = P
        self.PF = PF
        self.basal_vec = basal_vec
        self.use_noise = use_noise

        self.register_buffer("glucose_PID", P.PID_par.ref*torch.ones(P.PID_par.integral_duration+1, dtype=torch.float32))
        self.register_buffer("saturation_error", torch.zeros(1, dtype=torch.float32))

        self.scaler_glucose = scaler_glucose
        self.scaler_insulin = scaler_insulin
        self.scaler_meal = scaler_meal

    def update(self, vec, new_value):
        return torch.cat([vec[1:], new_value.unsqueeze(0)])


    def forward(self, CGM_i, i):
        """
        Compute the next state and output of the system.

        Args:
            u_ext (torch.Tensor, normalized): external input at t. shape = (batch_size, 1, input_dim)
            y (torch.Tensor, normalized): plant's output at t. shape = (batch_size, 1, output_dim)

        Returns:
            torch.Tensor, torch.Tensor: Input of plant and next output at t+1. shape = (batch_size, 1, state_dim), shape = (batch_size, 1, output_dim)
        """
 
  
        # update the previous informations



        ###############################################################
        #Compute next state and output
        i = i.item()

        ToD = i * 5 % 1440
        P = self.P
        PF = self.PF
        use_noise = self.use_noise
        
        basal_vec = self.basal_vec # it is not normalized
        CGM_i = self.scaler_glucose.denormalize(CGM_i)

        ########################################################
        self.glucose_PID = self.update(self.glucose_PID, CGM_i)
        


        if i < 5:
            bolus = torch.tensor(0.0, dtype=torch.float32)
        else:
            # candidate_error deve essere passato di volta in volta , lungo [start_idx:i + 1]
            candidate_error = P.PID_par.ref - self.glucose_PID
            candidate_error_capped = torch.maximum(candidate_error,torch.tensor(P.PID_par.ref - 140, dtype=torch.float32))
            e_sum = torch.sum(candidate_error_capped)

            CR_index = np.where(np.arange(0, 289) <= round((ToD % 1) * 1440 / 5))[0][-1]
            CR_now = P.Patient_par.CR_values_not_norm[0, CR_index] / torch.max(P.Patient_par.CR_values_not_norm)
            # Compute Kp, Kd, and Ki
            K_p = P.PID_par.K_p / CR_now
            K_d = P.PID_par.K_d / P.PID_par.tsController / CR_now
            K_i = P.PID_par.K_i * P.PID_par.tsController / CR_now

            e = P.PID_par.ref - self.glucose_PID[-1]
            e_m1 = P.PID_par.ref - self.glucose_PID[-2]

            # Compute PID control action (delta_P + delta_I + delta_D)
            delta_P = K_p * e
            delta_D = K_d * (e - e_m1)
            delta_I = K_i * e_sum

            # Inizializza le variabili
            booster_d_neg = 1
            booster_d_pos = 1

            if len(self.glucose_PID) > 3:
                booster_d_neg, booster_d_pos = PF.function_booster_d(self.glucose_PID)

            basal_PID = delta_P + delta_D * booster_d_pos + delta_I

            CR_tuned = P.Patient_par.CR_tuned[P.patient - 1]

            bolus = basal_PID * P.PID_par.conversion_index * booster_d_neg * CR_tuned

        basal = PF.calculate_basal(basal_vec, ToD)
        basal = torch.tensor(basal, dtype=torch.float32)
        bolus = torch.tensor(bolus, dtype=torch.float32)

        I = basal / 60 * P.PID_par.ts_measurement + bolus

        rwgn_instantaneous = 0
        if use_noise:
            mu = 0
            sigma = torch.min(basal_vec.values) / 60 * P.PID_par.ts_measurement * 0.4

            if i < 5:
                rwgn_instantaneous = 0
            else:
                rwgn_instantaneous = PF.rwgn_at_time(ToD, 42, mu, sigma)

        I_rwgn = I + rwgn_instantaneous

        ######################################

        bolus_sat, basal_sat, self.saturation_error = PF.saturation_of_pump_and_trasformation(
            I_rwgn, 0, P.PID_par.ts_measurement, P.pumpParameter, self.saturation_error
        )

        I_sat = bolus_sat + basal_sat

        R = I_sat - I
        
        if I.dtype != torch.float32 or I_rwgn.dtype != torch.float32 or I_sat.dtype != torch.float32 or R.dtype != torch.float32:
            print(f"Errore: uno dei tensori non è float32")
        
        
        

        # return I, I_sat.squeeze()  # entrambi scalari
        return self.scaler_insulin.normalize(I), self.scaler_insulin.normalize(I_rwgn), self.scaler_insulin.normalize(I_sat.squeeze()), self.scaler_insulin.normalize(R.squeeze())   # entrambi scalari





    def run(self, CGM, time, saturation_error_init = None, glucose_PID_init = None):
        """
        Simulates the closed-loop system for a given initial condition.

        Args:
            x0 (torch.Tensor): Initial state. Shape = (batch_size, 1, state_dim)
            u_ext (torch.Tensor): External input signal. Shape = (batch_size, horizon, input_dim) normalized
            output_noise_std: standard deviation of output noise

        Returns:
            torch.Tensor, torch.Tensor: Trajectories of outputs and inputs normalized
        """

        if saturation_error_init is not None:
            self.saturation_error[:] = saturation_error_init

        if glucose_PID_init is not None:
            self.glucose_PID[:] = glucose_PID_init

        u_pid_traj = []
        u_pid_rwgn_traj = []
        u_pid_rwgn_sat_traj = []
        r_traj = []
        for CGM_t, t in zip(CGM, time):
            # if t == 646:
            #     print("Debug point at time step 648")
            u_pid, u_pid_rwgn, u_pid_rwgn_sat, r = self.forward(CGM_t, t)
            u_pid_traj.append(u_pid)
            u_pid_rwgn_traj.append(u_pid_rwgn)
            u_pid_rwgn_sat_traj.append(u_pid_rwgn_sat)
            r_traj.append(r)

        u_pid_traj = torch.stack(u_pid_traj).to(torch.float32)
        u_pid_rwgn_traj = torch.stack(u_pid_rwgn_traj).to(torch.float32)
        u_pid_rwgn_sat_traj = torch.stack(u_pid_rwgn_sat_traj).to(torch.float32)
        r_traj = torch.stack(r_traj).to(torch.float32)


        return u_pid_traj, u_pid_rwgn_traj, u_pid_rwgn_sat_traj, r_traj

    def __call__(self, CGM, time, saturation_error_init = None, glucose_PID_init = None):
        """

        Args:
            x0 (torch.Tensor): Initial state. Shape = (batch_size, 1, state_dim)
            u_ext (torch.Tensor): External input signal. Shape = (batch_size, 1, input_dim)

        Returns:
            torch.Tensor, torch.Tensor: Trajectories of outputs and inputs
        """
        return self.run(CGM, time, saturation_error_init, glucose_PID_init)



class NonLinearController_p(nn.Module):
    torch.set_default_dtype(torch.float32)

    """Simulates the closed-loop system (Plant + Controller)."""

    def __init__(self, P, PF, basal_vec, scaler_glucose, scaler_insulin, scaler_meal, batch_size=1, use_noise=False):
        super().__init__()
        self.P = P
        self.PF = PF
        self.basal_vec = basal_vec
        self.use_noise = use_noise
        self.batch_size = batch_size  # MODIFICATO PER BATCH

        # MODIFICATO PER BATCH: aggiunta dimensione batch
        self.register_buffer(
            "glucose_PID",
            P.PID_par.ref * torch.ones((batch_size, P.PID_par.integral_duration + 1), dtype=torch.float32)  # MODIFICATO PER BATCH
        )
        self.register_buffer(
            "saturation_error",
            torch.zeros(batch_size, 1, dtype=torch.float32)  # MODIFICATO PER BATCH
        )

        self.scaler_glucose = scaler_glucose
        self.scaler_insulin = scaler_insulin
        self.scaler_meal = scaler_meal

    # MODIFICATO PER BATCH: update supporta batch
    def update(self, vec, new_value):
        # vec: (batch_size, vec_len), new_value: (batch_size,)
        return torch.cat([vec[:, 1:], new_value.unsqueeze(1)], dim=1)  # MODIFICATO PER BATCH

    def forward(self, CGM_i, i):
        """
        Compute the next state and output of the system in parallel for batch.
        CGM_i: (batch_size, ) or (batch_size, 1)
        i: (batch_size, ) or scalar
        """
        

        # MODIFICATO PER BATCH: gestisco batch dimension
        batch_size = CGM_i.shape[0]
        CGM_i = CGM_i.squeeze(1) if CGM_i.dim() == 2 else CGM_i  # MODIFICATO PER BATCH
        CGM_i = self.scaler_glucose.denormalize(CGM_i)

        self.glucose_PID = self.update(self.glucose_PID, CGM_i)  # MODIFICATO PER BATCH

        P = self.P
        PF = self.PF
        basal_vec = self.basal_vec
        use_noise = self.use_noise

        # MODIFICATO PER BATCH: ToD calcolato batch-wise
        if isinstance(i, torch.Tensor):
            ToD = (i*5) % 1440  # MODIFICATO PER BATCH
        else:
            ToD = i*5 % 1440

        # Inizializza bolus batch-wise
        bolus = torch.zeros(batch_size, dtype=torch.float32)  # MODIFICATO PER BATCH

        # PID calculation
        # MODIFICATO PER BATCH: calcolo solo per i>=5
        mask = i >= 5 if isinstance(i, torch.Tensor) else i >= 5
        if mask.any() if isinstance(mask, torch.Tensor) else mask:
            candidate_error = P.PID_par.ref - self.glucose_PID  # MODIFICATO PER BATCH
            candidate_error_capped = torch.maximum(candidate_error, torch.tensor(P.PID_par.ref - 140, dtype=torch.float32))  # MODIFICATO PER BATCH
            e_sum = torch.sum(candidate_error_capped, dim=1).unsqueeze(1)  # MODIFICATO PER BATCH

            # Calcolo CR_now batch-wise (semplice broadcasting)
            CR_index = ((ToD % 1) * 1440 / 5).long() # !!! (ToD % 1) percentuale su 1 ma ToD è intero, verificare che non ci siano stranezze # MODIFICATO PER BATCH
            
            
            CR_now = P.Patient_par.CR_values_not_norm[0, CR_index] / torch.max(P.Patient_par.CR_values_not_norm)  # MODIFICATO PER BATCH

            K_p = P.PID_par.K_p / CR_now  # MODIFICATO PER BATCH
            K_d = P.PID_par.K_d / P.PID_par.tsController / CR_now  # MODIFICATO PER BATCH
            K_i = P.PID_par.K_i * P.PID_par.tsController / CR_now  # MODIFICATO PER BATCH

            e = P.PID_par.ref - self.glucose_PID[:, -1]  # MODIFICATO PER BATCH
            e_m1 = P.PID_par.ref - self.glucose_PID[:, -2]  # MODIFICATO PER BATCH

            delta_P = K_p * e.unsqueeze(1)  # MODIFICATO PER BATCH
            delta_D = K_d * (e.unsqueeze(1) - e_m1.unsqueeze(1))  # MODIFICATO PER BATCH
            delta_I = K_i* e_sum  # MODIFICATO PER BATCH

            booster_d_neg = torch.ones(batch_size,1, dtype=torch.float32)  # MODIFICATO PER BATCH
            booster_d_pos = torch.ones(batch_size,1, dtype=torch.float32)  # MODIFICATO PER BATCH

            if self.glucose_PID.shape[1] > 3:
                booster_d_neg, booster_d_pos = PF.function_booster_d_p(self.glucose_PID)  # MODIFICATO PER BATCH

            basal_PID = delta_P + delta_D * booster_d_pos + delta_I  # MODIFICATO PER BATCH
            CR_tuned = P.Patient_par.CR_tuned[P.patient - 1]  # MODIFICATO PER BATCH
            bolus = basal_PID * P.PID_par.conversion_index * booster_d_neg * CR_tuned  # MODIFICATO PER BATCH
            
            if torch.isinf(bolus).any():
                print("Errore: bolus contiene NaN")

        # Basal e I batch-wise
        basal = PF.calculate_basal_p(basal_vec.time, basal_vec.values, ToD)  # MODIFICATO PER BATCH
        basal = torch.tensor(basal, dtype=torch.float32) # MODIFICATO PER BATCH
        bolus = bolus.detach().clone().float() 
        I = basal / 60 * P.PID_par.ts_measurement + bolus  # MODIFICATO PER BATCH

        # Rumore batch-wise
        rwgn_instantaneous = torch.zeros(batch_size, dtype=torch.float32)  # MODIFICATO PER BATCH
        if use_noise:
            mu = 0
            sigma = torch.min(basal_vec.values) / 60 * P.PID_par.ts_measurement * 0.4
            rwgn_instantaneous = PF.rwgn_at_time(ToD, 42, mu, sigma)  # MODIFICATO PER BATCH
            


        I_rwgn = I + rwgn_instantaneous  # MODIFICATO PER BATCH
        
        if torch.isnan(self.saturation_error).any() or torch.isnan(I_rwgn).any():
            print("Errore: saturation_error o I_rwgn contengono NaN")
        
        if torch.isinf(I_rwgn).any():
            print("Errore: I_rwgn contiene Inf")

        # Saturation batch-wise
        bolus_sat, basal_sat, self.saturation_error = PF.saturation_of_pump_and_trasformation_p(
            I_rwgn, torch.zeros(batch_size,1), P.PID_par.ts_measurement, P.pumpParameter, self.saturation_error  # MODIFICATO PER BATCH
        )
        
        if torch.isnan(self.saturation_error).any() or torch.isnan(I_rwgn).any():
            print("Errore: saturation_error o I_rwgn contengono NaN")

        I_sat = bolus_sat + basal_sat  # MODIFICATO PER BATCH
        R = I_sat - I  # MODIFICATO PER BATCH
        

        # Normalizzazione batch-wise
        return self.scaler_insulin.normalize(I), \
               self.scaler_insulin.normalize(I_rwgn), \
               self.scaler_insulin.normalize(I_sat), \
               self.scaler_insulin.normalize(R)  # MODIFICATO PER BATCH


    def __call__(self, CGM, time, saturation_error_init=None, glucose_PID_init=None):
        return self.run(CGM, time, saturation_error_init, glucose_PID_init)




    def run(self, CGM, time, saturation_error_init = None, glucose_PID_init = None):
        """
        Simulates the closed-loop system for a given initial condition.

        Args:
            x0 (torch.Tensor): Initial state. Shape = (batch_size, 1, state_dim)
            u_ext (torch.Tensor): External input signal. Shape = (batch_size, horizon, input_dim) normalized
            output_noise_std: standard deviation of output noise

        Returns:
            torch.Tensor, torch.Tensor: Trajectories of outputs and inputs normalized
        """

        if saturation_error_init is not None:
            self.saturation_error[:] = saturation_error_init

        if glucose_PID_init is not None:
            self.glucose_PID[:] = glucose_PID_init
            
        if torch.isnan(self.saturation_error).any():
            print("Errore: saturation_error o I_rwgn contengono NaN")

        u_pid_traj = []
        u_pid_rwgn_traj = []
        u_pid_rwgn_sat_traj = []
        r_traj = []
        for CGM_t, t in zip(CGM.T, time.T):
            # if t == 646:
            #     print("Debug point at time step 648")
            CGM_t = CGM_t.unsqueeze(1)
            t = t.unsqueeze(1)
            u_pid, u_pid_rwgn, u_pid_rwgn_sat, r = self.forward(CGM_t, t)
            u_pid_traj.append(u_pid)
            u_pid_rwgn_traj.append(u_pid_rwgn)
            u_pid_rwgn_sat_traj.append(u_pid_rwgn_sat)
            r_traj.append(r)

        u_pid_traj = torch.cat(u_pid_traj, dim=1).to(torch.float32)
        u_pid_rwgn_traj = torch.cat(u_pid_rwgn_traj, dim=1).to(torch.float32)
        u_pid_rwgn_sat_traj = torch.cat(u_pid_rwgn_sat_traj, dim=1).to(torch.float32)
        r_traj = torch.cat(r_traj, dim=1).to(torch.float32)


        return u_pid_traj, u_pid_rwgn_traj, u_pid_rwgn_sat_traj, r_traj

    def __call__(self, CGM, time, saturation_error_init = None, glucose_PID_init = None):
        """

        Args:
            x0 (torch.Tensor): Initial state. Shape = (batch_size, 1, state_dim)
            u_ext (torch.Tensor): External input signal. Shape = (batch_size, 1, input_dim)

        Returns:
            torch.Tensor, torch.Tensor: Trajectories of outputs and inputs
        """
        return self.run(CGM, time, saturation_error_init, glucose_PID_init)
    
    
class NonLinearController_p2(nn.Module):
    torch.set_default_dtype(torch.float32)

    """Simulates the closed-loop system (Plant + Controller)."""

    def __init__(self, P, PF, basal_vec, scaler_glucose, scaler_insulin, scaler_meal, batch_size=1, use_noise=False, saturation_error_init = None, glucose_PID_init = None):
        super().__init__()
        self.P = P
        self.PF = PF
        self.basal_vec = basal_vec
        self.use_noise = use_noise
        self.batch_size = batch_size  # MODIFICATO PER BATCH

        # Se non fornisci inizializzazioni, usa i default
        if saturation_error_init is None:
            saturation_error_init = torch.zeros(batch_size, 1, dtype=torch.float32)
        if glucose_PID_init is None:
            glucose_PID_init = P.PID_par.ref * torch.ones(
                (batch_size, P.PID_par.integral_duration + 1), dtype=torch.float32
            )

        # Registrazione buffer con i valori iniziali forniti
        self.register_buffer("glucose_PID", glucose_PID_init.clone())
        self.register_buffer("saturation_error", saturation_error_init.clone())

        self.scaler_glucose = scaler_glucose
        self.scaler_insulin = scaler_insulin
        self.scaler_meal = scaler_meal

    # MODIFICATO PER BATCH: update supporta batch
    def update(self, vec, new_value):
        # vec: (batch_size, vec_len), new_value: (batch_size,)
        return torch.cat([vec[:, 1:], new_value.unsqueeze(1)], dim=1)  # MODIFICATO PER BATCH

    def forward(self, CGM_i, i):
        """
        Compute the next state and output of the system in parallel for batch.
        CGM_i: (batch_size, ) or (batch_size, 1)
        i: (batch_size, ) or scalar
        """
        

        # MODIFICATO PER BATCH: gestisco batch dimension
        batch_size = CGM_i.shape[0]
        CGM_i = CGM_i.squeeze(1) if CGM_i.dim() == 2 else CGM_i  # MODIFICATO PER BATCH
        CGM_i = self.scaler_glucose.denormalize(CGM_i)

        self.glucose_PID = self.update(self.glucose_PID, CGM_i)  # MODIFICATO PER BATCH

        P = self.P
        PF = self.PF
        basal_vec = self.basal_vec
        use_noise = self.use_noise

        # MODIFICATO PER BATCH: ToD calcolato batch-wise
        if isinstance(i, torch.Tensor):
            ToD = (i*5) % 1440  # MODIFICATO PER BATCH
        else:
            ToD = i*5 % 1440

        # Inizializza bolus batch-wise
        bolus = torch.zeros(batch_size, dtype=torch.float32)  # MODIFICATO PER BATCH

        # PID calculation
        # MODIFICATO PER BATCH: calcolo solo per i>=5
        mask = i >= 5 if isinstance(i, torch.Tensor) else i >= 5
        if mask.any() if isinstance(mask, torch.Tensor) else mask:
            candidate_error = P.PID_par.ref - self.glucose_PID  # MODIFICATO PER BATCH
            candidate_error_capped = torch.maximum(candidate_error, torch.tensor(P.PID_par.ref - 140, dtype=torch.float32))  # MODIFICATO PER BATCH
            e_sum = torch.sum(candidate_error_capped, dim=1).unsqueeze(1)  # MODIFICATO PER BATCH

            # Calcolo CR_now batch-wise (semplice broadcasting)
            CR_index = ((ToD % 1) * 1440 / 5).long() # !!! (ToD % 1) percentuale su 1 ma ToD è intero, verificare che non ci siano stranezze # MODIFICATO PER BATCH
            
            
            CR_now = P.Patient_par.CR_values_not_norm[0, CR_index] / torch.max(P.Patient_par.CR_values_not_norm)  # MODIFICATO PER BATCH

            K_p = P.PID_par.K_p / CR_now  # MODIFICATO PER BATCH
            K_d = P.PID_par.K_d / P.PID_par.tsController / CR_now  # MODIFICATO PER BATCH
            K_i = P.PID_par.K_i * P.PID_par.tsController / CR_now  # MODIFICATO PER BATCH

            e = P.PID_par.ref - self.glucose_PID[:, -1]  # MODIFICATO PER BATCH
            e_m1 = P.PID_par.ref - self.glucose_PID[:, -2]  # MODIFICATO PER BATCH

            delta_P = K_p * e.unsqueeze(1)  # MODIFICATO PER BATCH
            delta_D = K_d * (e.unsqueeze(1) - e_m1.unsqueeze(1))  # MODIFICATO PER BATCH
            delta_I = K_i* e_sum  # MODIFICATO PER BATCH

            booster_d_neg = torch.ones(batch_size,1, dtype=torch.float32)  # MODIFICATO PER BATCH
            booster_d_pos = torch.ones(batch_size,1, dtype=torch.float32)  # MODIFICATO PER BATCH

            if self.glucose_PID.shape[1] > 3:
                booster_d_neg, booster_d_pos = PF.function_booster_d_p(self.glucose_PID)  # MODIFICATO PER BATCH

            basal_PID = delta_P + delta_D * booster_d_pos + delta_I  # MODIFICATO PER BATCH
            CR_tuned = P.Patient_par.CR_tuned[P.patient - 1]  # MODIFICATO PER BATCH
            bolus = basal_PID * P.PID_par.conversion_index * booster_d_neg * CR_tuned  # MODIFICATO PER BATCH
            
            if torch.isinf(bolus).any():
                print("Errore: bolus contiene NaN")

        # Basal e I batch-wise
        basal = PF.calculate_basal_p(basal_vec.time, basal_vec.values, ToD)  # MODIFICATO PER BATCH
        basal = torch.tensor(basal, dtype=torch.float32) # MODIFICATO PER BATCH
        bolus = bolus.detach().clone().float() 
        I = basal / 60 * P.PID_par.ts_measurement + bolus  # MODIFICATO PER BATCH

        # Rumore batch-wise
        rwgn_instantaneous = torch.zeros(batch_size, dtype=torch.float32)  # MODIFICATO PER BATCH
        if use_noise:
            mu = 0
            sigma = torch.min(basal_vec.values) / 60 * P.PID_par.ts_measurement * 0.4
            rwgn_instantaneous = PF.rwgn_at_time(ToD, 42, mu, sigma)  # MODIFICATO PER BATCH
            


        I_rwgn = I + rwgn_instantaneous  # MODIFICATO PER BATCH
        
        if torch.isnan(self.saturation_error).any() or torch.isnan(I_rwgn).any():
            print("Errore: saturation_error o I_rwgn contengono NaN")
        
        if torch.isinf(I_rwgn).any():
            print("Errore: I_rwgn contiene Inf")

        # Saturation batch-wise
        bolus_sat, basal_sat, self.saturation_error = PF.saturation_of_pump_and_trasformation_p(
            I_rwgn, torch.zeros(batch_size,1), P.PID_par.ts_measurement, P.pumpParameter, self.saturation_error  # MODIFICATO PER BATCH
        )
        
        if torch.isnan(self.saturation_error).any() or torch.isnan(I_rwgn).any():
            print("Errore: saturation_error o I_rwgn contengono NaN")

        I_sat = bolus_sat + basal_sat  # MODIFICATO PER BATCH
        R = I_sat - I  # MODIFICATO PER BATCH
        

        # Normalizzazione batch-wise
        return self.scaler_insulin.normalize(I), \
               self.scaler_insulin.normalize(I_rwgn), \
               self.scaler_insulin.normalize(I_sat), \
               self.scaler_insulin.normalize(R)  # MODIFICATO PER BATCH


    def __call__(self, CGM, time, saturation_error_init=None, glucose_PID_init=None):
        return self.run(CGM, time, saturation_error_init, glucose_PID_init)




    def run(self, CGM, time):
        """
        Simulates the closed-loop system for a given initial condition.

        Args:
            x0 (torch.Tensor): Initial state. Shape = (batch_size, 1, state_dim)
            u_ext (torch.Tensor): External input signal. Shape = (batch_size, horizon, input_dim) normalized
            output_noise_std: standard deviation of output noise

        Returns:
            torch.Tensor, torch.Tensor: Trajectories of outputs and inputs normalized
        """


        u_pid_traj = []
        u_pid_rwgn_traj = []
        u_pid_rwgn_sat_traj = []
        r_traj = []
        for CGM_t, t in zip(CGM.T, time.T):
            # if t == 646:
            #     print("Debug point at time step 648")
            CGM_t = CGM_t.unsqueeze(1)
            t = t.unsqueeze(1)
            u_pid, u_pid_rwgn, u_pid_rwgn_sat, r = self.forward(CGM_t, t)
            u_pid_traj.append(u_pid)
            u_pid_rwgn_traj.append(u_pid_rwgn)
            u_pid_rwgn_sat_traj.append(u_pid_rwgn_sat)
            r_traj.append(r)

        u_pid_traj = torch.cat(u_pid_traj, dim=1).to(torch.float32)
        u_pid_rwgn_traj = torch.cat(u_pid_rwgn_traj, dim=1).to(torch.float32)
        u_pid_rwgn_sat_traj = torch.cat(u_pid_rwgn_sat_traj, dim=1).to(torch.float32)
        r_traj = torch.cat(r_traj, dim=1).to(torch.float32)


        return u_pid_traj, u_pid_rwgn_traj, u_pid_rwgn_sat_traj, r_traj

    def __call__(self, CGM, time):
        """

        Args:
            x0 (torch.Tensor): Initial state. Shape = (batch_size, 1, state_dim)
            u_ext (torch.Tensor): External input signal. Shape = (batch_size, 1, input_dim)

        Returns:
            torch.Tensor, torch.Tensor: Trajectories of outputs and inputs
        """
        return self.run(CGM, time)
    
    
    
    
class ClosedLoopSystem(nn.Module):
    """Simulates the closed-loop system (Plant + Controller)."""

    def __init__(self, system_model, controller, negative: bool = False):
        super().__init__()
        self.system_model = system_model
        self.controller = controller
        self.negative = negative
        # self.output_dim = self.system_model.output_dim
        # self.input_dim = self.system_model.input_dim
        # self.state_dim = self.system_model.state_dim

        # just self does not register anything in the state dict of the model, it doesn't go to the specified device
        # TODO: maybe the controller should have its state in the register buffer
        self.register_buffer('x', None)
        self.register_buffer('y_prev', None)

    def reset(self):
        """
        Reset the internal state.

        Args:
            x0 (torch.Tensor, optional): Initial state, shape = (batch_size, 1, state_dim).
            batch_size (int): Batch size for initialization.
        """

        self.system_model.reset()
        self.x = self.system_model.x
        y0 = self.y0_from_x0(self.system_model.x)
        self.y_prev = y0

    def y0_from_x0(self, x0):
        y0 = self.system_model.y0_from_x0(x0)
        return y0

    def forward(self, u_ext):
        """
        Compute the next output of the system.

        Args:
            u_ext (torch.Tensor): external input at t. shape = (batch_size, 1, input_dim)

        Returns:
            torch.Tensor: Next output at t+1. shape = (batch_size, 1, output_dim)
        """

        #Compute next state and output
        control_u = self.controller.forward(self.y_prev)  # Compute control input
        # minus sign for the control input
        if self.negative:
            control_u = -control_u
        u = control_u + u_ext
        y = self.system_model.forward(u)
        self.y_prev = y
        return y


    def run(self, x0, u_ext, time):
        """
        Simulates the closed-loop system for a given initial condition.

        Args:
            x0 (torch.Tensor): Initial state. Shape = (batch_size, 1, state_dim)
            u_ext (torch.Tensor): External input signal. Shape = (batch_size, horizon, input_dim)
            output_noise: realizations of output noise

        Returns:
            torch.Tensor: Trajectories of outputs (batch_size, horizon, output_dim) [y0, ..., y_T-1]
        """

        batch_size = u_ext.shape[0]
        horizon = u_ext.shape[1]

        # Storage for trajectories
        y_traj = []
        u_traj = []

        self.system_model.reset()
        # Use pre-generated noise
        y0 = self.y0_from_x0(self.system_model.x)
        y = y0   # First noise realization
        self.y_prev = y
        
        # dimensioni input delle due REN
        dim_in_0 = self.system_model.REN_0.dim_in
        dim_in_1 = self.system_model.REN_1.dim_in

        for i, t in enumerate(time[0,:,0]):
            y_traj.append(y)  # Store output
            control_u, _, _,_ = self.controller.forward(self.y_prev.squeeze(), t)  # Compute control input
            if self.negative:
                control_u = -control_u

            # Prendo u0 dalla sequenza u_ext e sommo il controllo
            u0_t = u_ext[:, i:i+1, :dim_in_0] 

            # Prendo u1 dalla sequenza u_ext
            u1_t = u_ext[:, i:i+1, dim_in_0:dim_in_0+dim_in_1]+ control_u

            # Forward DualREN passo per passo
            y = self.system_model.forward(u0_t, u1_t)

            # Concateno per salvare la traiettoria degli input
            u_traj.append(torch.cat((u0_t, u1_t), dim=-1))

            self.y_prev = y

        # Convert lists to tensors
        y_traj = torch.cat(y_traj, dim=1)  # Shape: (batch_size, horizon, output_dim)
        u_traj = torch.cat(u_traj, dim=1)  # Shape: (batch_size, horizon, input_dim)

        return u_traj, y_traj

    def __call__(self, x0, u_ext, time):
        """
        Args:
            x0 (torch.Tensor): Initial state. Shape = (batch_size, 1, state_dim)
            u_ext (torch.Tensor): External input signal. Shape = (batch_size, horizon, input_dim)
            output_noise_std: standard deviation of output noise

        Returns:
            torch.Tensor: Trajectories of outputs (batch_size, horizon, output_dim) [y0, ..., y_T-1]
        """
        return self.run(x0, u_ext, time)