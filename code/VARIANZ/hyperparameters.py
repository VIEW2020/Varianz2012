'''
May 2020 by Sebastiano Barbieri
s.barbieri@unsw.edu.au
https://www.github.com/sebbarb/
'''

import torch
import optuna
from datetime import datetime
from pdb import set_trace as bp

class Hyperparameters:
    def __init__(self, trial=None):
    
        ### General #########################################################
        
        self.gender = 'females'
        self.min_count = 200 # codes whose occurrence is less than min_count are encoded as OTHER
        
        # Data
        self.data_dir = '../../data/'
        self.data_pp_dir = '../../data/pp/'
        self.log_dir = self.data_dir + 'log_' + self.gender + '_iter3/'
        self.results_dir = self.data_dir + 'results/'
        self.plots_dir = self.results_dir + 'plots/'
        
        # Seeds
        self.np_seed = 1234
        self.torch_seed = 42
        
        # Training
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        ### Model ###########################################################

        self.batch_size = 256
        self.max_epochs = 10000
        self.patience = 10 # early stopping
        self.num_months_hx = 60
        now = datetime.now() # current date and time
        self.model_name = now.strftime('%Y%m%d_%H%M%S_%f') + '.pt'    
            
        # Network
        if trial:
            self.nonprop_hazards = trial.suggest_categorical('nonprop_hazards', [True, False])
            self.embedding_dim = trial.suggest_categorical('embedding_dim', [16, 32, 64, 128])
            self.rnn_type = trial.suggest_categorical('rnn_type', ['GRU', 'LSTM'])
            self.num_rnn_layers = trial.suggest_int('num_rnn_layers', 1, 3)
            if self.num_rnn_layers > 1:
                self.dropout = trial.suggest_discrete_uniform('dropout', 0.0, 0.5, 0.1)
            else:
                self.dropout = trial.suggest_discrete_uniform('dropout', 0.0, 0.0, 0.1)
            self.num_mlp_layers = trial.suggest_int('num_mlp_layers', 0, 2)
            self.add_diagt = trial.suggest_categorical('add_diagt', [True, False])
            self.add_month = trial.suggest_categorical('add_month', ['ignore', 'concat', 'embedding'])
            self.summarize = trial.suggest_categorical('summarize', ['hidden', 'output_max', 'output_sum', 'output_avg'])
            self.learning_rate = trial.suggest_categorical('learning_rate', [1e-4, 1e-3, 1e-2])
        else:
            self.nonprop_hazards = False
            self.embedding_dim = 32
            self.rnn_type = 'GRU'
            self.num_rnn_layers = 3
            self.dropout = 0.1
            self.num_mlp_layers = 1
            self.add_diagt = True
            self.add_month = 'concat'
            self.summarize = 'output_max'
            self.learning_rate = 1e-3
        
        ### Evaluation ######################################################
        
        self.best_model = 'final.pt' if self.gender == 'males' else 'final.pt'
