'''
May 2020 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/
'''

import torch
from datetime import datetime
from pdb import set_trace as bp

class Hyperparameters:
    ''' Hyperparameters '''
    
    ### General #########################################################
    
    gender = 'males'
    min_count = 200 # codes whose occurrence is less than min_count are encoded as OTHER
    
    # Data
    data_dir = '../../data/'
    data_pp_dir = '../../data/pp/'
    log_dir = data_dir + 'log_' + gender + '/'
    plot_dir = data_dir + 'plot/'
    
    # Seeds
    np_seed = 1234
    torch_seed = 42
    
    # Training
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    ### Model ###########################################################

    nonprop_hazards = False
    batch_size = 256
    max_epochs = 10000
    patience = 10 # early stopping
    
    # Network
    embedding_dim = 16
    num_months_hx = 60
    
    now = datetime.now() # current date and time
    model_name = 'model_' + now.strftime('%Y%m%d_%H%M%S_%f') + '.pt'

    ### Evaluation ######################################################
    
    sample_comp_bh = 10000 if nonprop_hazards else None
