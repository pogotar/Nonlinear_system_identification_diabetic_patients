import torch

def set_params():
    # # # # # # # # Parameters # # # # # # # #
    
    torch.set_default_dtype(torch.float32) 

    #Model
    x0 = torch.tensor([0.01, 0.01])  # Initial state
    input_dim = [1, 1] # input dimensions
    output_dim = [1, 1] # output dimensions


    dim_internal = [3, 4] # \xi dimension -- number of states of REN
    dim_nl = [2, 2] # dimension of the square matrix D11 -- number of _non-linear layers_ of the REN

    y_init = torch.tensor([0.0, 0.0])

    IQC_type = ['monotone', 'monotone'] # IQC constraint type: 'l2_gain', 'monotone', 'passive'
    gamma = torch.tensor([0.3, 0.02])  # for IQC constraints

    use_noise = True

    ts = 5  # Sampling time (minutes)

    # # # # # # # # Hyperparameters # # # # # # # #
    learning_rate = 1e-3
    epochs = 1 # 500

    # # # # # # # # Data path # # # # # # # #

    redo_save = True

    exp_identifier = '1'
    num_days = 30  # 30 2

    string_noise = ''
    if use_noise:
        string_noise = '_rwgn'


    data_path = './data/train/sc_' + str(num_days) +  'days_identification' + string_noise + '/'
    model_folder = './models/exp' + exp_identifier + '_' + str(num_days) + 'days' + string_noise + '/'

    return x0, input_dim, output_dim, dim_internal, dim_nl, y_init, IQC_type, gamma, learning_rate, epochs, data_path, model_folder, redo_save, ts, use_noise, num_days


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

