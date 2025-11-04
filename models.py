from timeit import repeat
import torch
import torch.nn as nn
import numpy as np

torch.set_printoptions(precision=17)

class NonLinearController(nn.Module):
    torch.set_default_dtype(torch.float64)

    """Simulates the closed-loop system (Plant + Controller)."""

    def __init__(self, P, PF, basal_vec,  scaler_glucose, scaler_insulin, scaler_meal, use_noise=False):
        super().__init__()
        self.P = P
        self.PF = PF
        self.basal_vec = basal_vec
        self.use_noise = use_noise

        self.register_buffer("glucose_PID", P.PID_par.ref*torch.ones(P.PID_par.integral_duration+1, dtype=torch.float64))
        self.register_buffer("saturation_error", torch.zeros(1, dtype=torch.float64))
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

        #Compute next state and output


        ###############################################################

        ToD = i * 5 % 1440
        P = self.P
        PF = self.PF
        use_noise = self.use_noise
        
        basal_vec = self.basal_vec # it is not normalized
        CGM_i = self.scaler_glucose.denormalize(CGM_i)

        ########################################################
        self.glucose_PID = self.update(self.glucose_PID, CGM_i)

        if i < 5:
            bolus = torch.tensor(0.0, dtype=torch.float64)
        else:
            # candidate_error deve essere passato di volta in volta , lungo [start_idx:i + 1]
            candidate_error = P.PID_par.ref - self.glucose_PID
            candidate_error_capped = torch.maximum(candidate_error,torch.tensor(P.PID_par.ref - 140, dtype=torch.float64))
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
        basal = torch.tensor(basal, dtype=torch.float64)


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

        # return I, I_sat.squeeze()  # entrambi scalari
        return self.scaler_insulin.normalize(I), self.scaler_insulin.normalize(I_rwgn), self.scaler_insulin.normalize(I_sat.squeeze()), self.scaler_insulin.normalize(R.squeeze())   # entrambi scalari





    def run(self, CGM):
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
        for t in range(len(CGM)):
            # if t == 646:
            #     print("Debug point at time step 648")
            u_pid, u_pid_rwgn, u_pid_rwgn_sat, r = self.forward(CGM[t], t)
            u_pid_traj.append(u_pid)
            u_pid_rwgn_traj.append(u_pid_rwgn)
            u_pid_rwgn_sat_traj.append(u_pid_rwgn_sat)
            r_traj.append(r)

        u_pid_traj = torch.stack(u_pid_traj).to(torch.float64)
        u_pid_rwgn_traj = torch.stack(u_pid_rwgn_traj).to(torch.float64)
        u_pid_rwgn_sat_traj = torch.stack(u_pid_rwgn_sat_traj).to(torch.float64)
        r_traj = torch.stack(r_traj).to(torch.float64)


        return u_pid_traj, u_pid_rwgn_traj, u_pid_rwgn_sat_traj, r_traj

    def __call__(self, CGM):
        """

        Args:
            x0 (torch.Tensor): Initial state. Shape = (batch_size, 1, state_dim)
            u_ext (torch.Tensor): External input signal. Shape = (batch_size, 1, input_dim)

        Returns:
            torch.Tensor, torch.Tensor: Trajectories of outputs and inputs
        """
        return self.run(CGM)