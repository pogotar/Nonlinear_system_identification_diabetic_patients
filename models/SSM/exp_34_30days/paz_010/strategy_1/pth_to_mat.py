import torch
import scipy.io

# Caricamento del file pth
checkpoint = torch.load('.\models\SSM\exp_34_30days\paz_010\strategy_1/trained_models.pth', map_location='cpu')

def extract_ssm(state_dict):
    """Estrae i pesi di una singola SSM (encoder, decoder e 7 blocchi)"""
    def extract_block(i):
        p = f'blocks.{i}'
        return {
            'rho_raw': state_dict[f'{p}.lru.rho_raw'].numpy(),
            'theta': state_dict[f'{p}.lru.theta'].numpy(),
            'K12': state_dict[f'{p}.lru.K12_raw'].numpy(),
            'K21': state_dict[f'{p}.lru.K21_raw'].numpy(),
            'K22': state_dict[f'{p}.lru.K22_raw'].numpy(),
            'log_gamma': state_dict[f'{p}.lru.log_gamma'].numpy(),
            'ff_w': state_dict[f'{p}.ff.output_linear.0.weight'].numpy(),
            'ff_b': state_dict[f'{p}.ff.output_linear.0.bias'].numpy()
        }
    
    return {
        'encoder_w': state_dict['encoder.weight'].numpy(),
        'decoder_w': state_dict['decoder.weight'].numpy(),
        'blocks': [extract_block(i) for i in range(7)]
    }

# Esportiamo entrambe le SSM rilevate nel file 
data_to_save = {
    'ssm0': extract_ssm(checkpoint['SSM_0_state_dict']),
    'ssm1': extract_ssm(checkpoint['SSM_1_state_dict'])
}

scipy.io.savemat('ssm_all_params.mat', data_to_save)
print("File 'ssm_all_params.mat' creato con ssm0 e ssm1.")