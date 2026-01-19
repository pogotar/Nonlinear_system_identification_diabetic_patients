import torch

def set_params():
    # # # # # # # # Parameters # # # # # # # #
    
    torch.set_default_dtype(torch.float32) 

    #Model
    x0 = torch.tensor([0.01, 0.01])  # Initial state
    input_dim = [1, 1] # input dimensions
    output_dim = [1, 1] # output dimensions


    dim_internal = [8, 8] # [3, 4] # \xi dimension -- number of states of REN
    dim_nl = [8, 8] # [2, 2] # dimension of the square matrix D11 -- number of _non-linear layers_ of the REN

    y_init = torch.tensor([0.0, 0.0])

    IQC_type = ['monotone', 'monotone'] # IQC constraint type: 'l2_gain', 'monotone', 'passive'
    # gamma = torch.tensor([0.3, 0.02])  # for IQC constraints
    gamma = torch.tensor([5, 500])

    use_noise = True

    ts = 5  # Sampling time (minutes)

    # # # # # # # # Hyperparameters # # # # # # # #
    learning_rate = 1e-3
    epochs = 10 # 500

    # # # # # # # # Data path # # # # # # # #

    redo_save = True
    redo_save_101_I = True
    redo_save_101_M = True

    exp_identifier = 'no_strat_3' # train_batched
    num_days = 30  # 30 2

    string_noise = ''
    if use_noise:
        string_noise = '_rwgn'


    data_path = './data/train/sc_' + str(num_days) +  'days_identification' + string_noise + '/'
    model_folder = './models/exp' + exp_identifier + '_' + str(num_days) + 'days' + string_noise + '/'

    return x0, input_dim, output_dim, dim_internal, dim_nl, y_init, IQC_type, gamma, learning_rate, epochs, data_path, model_folder, redo_save, ts, use_noise, num_days, redo_save_101_I, redo_save_101_M, exp_identifier


def set_QR(gamma, input_dim, output_dim, IQC_type):
    # IQC constraints
    
    torch.set_default_dtype(torch.float32) 
        
    if IQC_type == 'l2_gain':
        # incremental L2 gain constraints
        Q = (-1 / gamma) * torch.eye(output_dim, output_dim)  # -1/gamma * I
        R = gamma * torch.eye(input_dim, input_dim)  # gamma * I
        S = torch.zeros(output_dim, input_dim)  # 0

    elif IQC_type == 'monotone':
        eps = 1e-4

        # monotone on l2
        Q = torch.zeros(output_dim, output_dim) -eps * torch.eye(output_dim) # 0
        R = - 2 * gamma * torch.eye(input_dim, input_dim)  # -2 nu I
        S = torch.eye(output_dim, input_dim)  # I

    elif IQC_type == 'passive':
        # incrementally strictly output passive
        Q = - 2 * gamma * torch.eye(output_dim, output_dim)  # - 2 rho I (Ho corretto da torch.ones a torch.eye come probabile intenzione)
        R = torch.zeros(input_dim, input_dim)  # 0
        S = torch.eye(output_dim, input_dim)  # I

    return Q, R, S

def ensure_3d(x):
    """ensures that tensors have dimension (batch, time, input_dim)."""
    if x.ndim == 1:
        # Case: sequence 1D -> (1, T, 1)
        x = x.unsqueeze(0).unsqueeze(-1)
    elif x.ndim == 2:
        # Case: batvh or sequence 2D -> (batch, T, 1)
        x = x.unsqueeze(-1)
    return x

def fun_start_controller(train_loader, loaded_parameters, scaler_glucose, scaler_insulin, dataset):

    CGM = dataset.CGM
    sat_e = dataset.sat_e
    
    processed = []
    for batch in train_loader:
        time_batch = batch[-1]

        # Se manca la dimensione batch, aggiungila
        if time_batch.dim() == 1:
            time_batch = time_batch.unsqueeze(0)   # (1, seq_len)

        processed.append(time_batch)

    time_batches = torch.cat(processed, dim=0)

    current_time_index = (time_batches[:, 0].int()).unsqueeze(1)
    previous_starting_index = (time_batches[:,0].int()-1).unsqueeze(1)
    previous_int_duration = previous_starting_index + torch.arange(-loaded_parameters.PID_par.integral_duration, 1) 


    saturation_error_init = scaler_insulin.denormalize(sat_e[previous_starting_index.long()].reshape_as(previous_starting_index))

    y_0 = CGM[current_time_index.long()].reshape_as(current_time_index)
    glucose_PID_init = scaler_glucose.denormalize(CGM[previous_int_duration.long()].reshape_as(previous_int_duration))

    # initial saturation error, string of previous CGM measurament, current CGM measurement
    return saturation_error_init, glucose_PID_init, y_0



def fun_start_controller_simple(train_loader, dataset):

    CGM = dataset.y
    
    processed = []
    for batch in train_loader:
        time_batch = batch[-1]

        # Se manca la dimensione batch, aggiungila
        if time_batch.dim() == 1:
            time_batch = time_batch.unsqueeze(0)   # (1, seq_len)

        processed.append(time_batch)

    time_batches = torch.cat(processed, dim=0)

    current_time_index = (time_batches[:, 0].int()).unsqueeze(1)


    # already normalized
    y_0 = CGM[current_time_index.long()].reshape_as(current_time_index)

    # initial saturation error, string of previous CGM measurament, current CGM measurement
    return y_0

