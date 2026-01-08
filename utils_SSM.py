import torch
import matplotlib.pyplot as plt


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

    exp_identifier = 'SSM_no_strat_3' # train_batched
    num_days = 30  # 30 2

    string_noise = ''
    if use_noise:
        string_noise = '_rwgn'


    data_path = './data/train/sc_' + str(num_days) +  'days_identification' + string_noise + '/'
    model_folder = './models/SSM/exp' + exp_identifier + '_' + str(num_days) + 'days' + string_noise + '/'

    return x0, input_dim, output_dim, dim_internal, dim_nl, y_init, IQC_type, gamma, learning_rate, epochs, data_path, model_folder, redo_save, ts, use_noise, num_days, redo_save_101_I, redo_save_101_M, exp_identifier



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

def plot_glucose_insulin(time, insulin=None, meal=None, glucose=None, 
                         predicted_glucose=None, title='Glucose and Insulin vs Time'):
    """
    Plot Glucose/Meal (top) and Insulin (bottom, if present) with dual y-axes.
    Meal points are shown as scatter only when non-zero.
    
    Parameters:
    - time: time array
    - insulin: insulin array (optional)
    - meal: meal array (optional) - plotted as scatter for non-zero values
    - glucose: actual glucose array (optional)
    - predicted_glucose: predicted glucose array (optional)
    - title: plot title
    """
    
    # Determina numero di subplot: 2 solo se ci sono sia insulin che glucose/meal
    has_glucose_or_meal = (glucose is not None or predicted_glucose is not None or meal is not None)
    n_plots = 2 if (insulin is not None and has_glucose_or_meal) else 1
    
    # Usa height_ratios per fare il secondo subplot più stretto
    if n_plots == 2:
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3.5 + 1.75), 
                                 gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.05})
    else:
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3.5))
    
    if n_plots == 1:
        axes = [axes]
    
    # ===== PRIMO PLOT: Meal/Glucose (o solo Insulin) =====
    ax1 = axes[0]
    
    # Se c'è il secondo subplot, non mettere xticks visibili sul primo
    if n_plots == 2:
        ax1.tick_params(axis='x', labelbottom=False)
    else:
        ax1.set_xlabel('Time step')
    
    # Se solo insulin, mostra solo insulin nel primo plot
    if insulin is not None and n_plots == 1:
        ax1.set_xlabel('Time step')
        ax1.plot(time, insulin, color='tab:red', label='Insulin', zorder=1, linewidth=2)
        ax1.set_ylabel('Insulin (u1)', color='tab:red')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax1.spines['left'].set_color('red')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=10)
        
        # Ylim dinamico basato su max insulina
        insulin_max = insulin.max()
        ylim_options = [2.5, 5, 7.5, 10, 15, 20]
        ylim = next((y for y in ylim_options if y >= insulin_max), ylim_options[-1])
        ax1.set_ylim(-1, ylim)
        
        ax1.set_title(title, fontsize=14, fontweight='bold')
    else:
        # Meal come scatter (solo punti non-zero)
        if meal is not None:
            # Filtra solo i valori non-zero
            meal_nonzero_mask = meal != 0
            meal_time = time[meal_nonzero_mask]
            meal_values = meal[meal_nonzero_mask]
            
            ax1.scatter(meal_time, meal_values, color='mediumseagreen', label='Meal', 
                       zorder=2, s=80, alpha=0.8, edgecolors='darkgreen', linewidth=1.5)
            ax1.set_ylabel('Meal (mg)', color='mediumseagreen')
            ax1.tick_params(axis='y', labelcolor='mediumseagreen')
            ax1.spines['left'].set_color('mediumseagreen')
        
        # Glucose a destra
        ax2 = ax1.twinx()
        ax2.set_ylabel('Glucose (mg/dL)', color='tab:blue')
        
        if predicted_glucose is not None:
            ax2.plot(time, predicted_glucose, color='darkblue', label='Predicted Glucose', 
                    zorder=20, linewidth=2, alpha=0.7)
        
        if glucose is not None:
            ax2.plot(time, glucose, color='cornflowerblue', label='Glucose', 
                    zorder=10, linewidth=2, alpha=0.7)
        
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        ax2.spines['right'].set_color('blue')
        
        ax1.grid(True, alpha=0.3)
        
        # Legenda
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
        
        ax1.set_title(title, fontsize=14, fontweight='bold')
        if meal is not None:
            ax1.set_ylim(-0.1, max([max(meal) * 1.1, 3]))
        # Limiti asse destro: 30-300 ma adatta se i dati sforano
        if glucose is not None or predicted_glucose is not None:
            min_val = 30
            max_val = 300
            
            if glucose is not None and predicted_glucose is not None:
                data_min = min(glucose.min(), predicted_glucose.min())
                data_max = max(glucose.max(), predicted_glucose.max())
            elif glucose is not None:
                data_min = glucose.min()
                data_max = glucose.max()
            else:
                data_min = predicted_glucose.min()
                data_max = predicted_glucose.max()
            
            # Se i dati sforano, adatta i limiti
            if data_min < min_val:
                min_val = data_min * 0.95
            if data_max > max_val:
                max_val = data_max * 1.05
            
            ax2.set_ylim(min_val, max_val)
    
    # ===== SECONDO PLOT: Insulin (se presente insieme a glucose/meal) =====
    if insulin is not None and n_plots == 2:
        ax3 = axes[1]
        ax3.set_xlabel('Time step')
        ax3.plot(time, insulin, color='tab:red', label='Insulin', zorder=1, linewidth=2)
        ax3.set_ylabel('Insulin (u1)', color='tab:red')
        ax3.tick_params(axis='y', labelcolor='tab:red')
        ax3.spines['left'].set_color('red')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right', fontsize=10)
        
        # Ylim dinamico basato su max insulina
        insulin_max = insulin.max()
        ylim_options = [1.5, 2.5, 5, 7.5, 10, 15, 20]
        
        # Trova il primo valore >= insulin_max
        ylim = next((y for y in ylim_options if y >= insulin_max), ylim_options[-1])
        
        ax3.set_ylim(-0.5, ylim)
    
    fig.tight_layout()
    plt.show()