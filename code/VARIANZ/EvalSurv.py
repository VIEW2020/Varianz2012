"""
D statistic & R-squared-D
Date: 2nd August 2020
Author: C Bharat and S Barbieri

References D statistic:
Royston & Sauerbrei (2004), "A new measure of prognostic separation in survival data", Stats In Med

Notes:
D statistic requires only the prognostic index/log-partial hazard from the fitted model
Can correct D for optimisim when model is fit and D is obtained from the same data set (not yet coded, see p729 of ref)
Can also stratify the model
"""

# Libraries
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from scipy.stats import norm
import numpy as np
import pandas as pd
import math

from pdb import set_trace as bp

class EvalSurv:
    """Class for evaluating predictions.

    Arguments: df {Panda.DataFrame} -- dataframe with columns LPH (log of the partial hazard - linear predictor - from fitted model),
        TIME and EVENT (events of test set, 1 if event occured and 0 if censored), SURV (predicted survival)
    """

    def __init__(self, df, base_surv=None):
        if set(['LPH', 'TIME', 'EVENT']) <= set(df.columns):
            self.df = df
            self.base_surv = base_surv
            self.kap = math.sqrt(8 / math.pi)
            self.D = None
            self.cens_surv = None
        else:
            raise('Missing columns in df')

    ###########################################################################################################

    def D_index(self):
        """
        Order linear predictor, durations and events in ascending order based on lph rank
        """
        self.df.sort_values(by='LPH', ascending=True, inplace=True)
        num = len(self.df.index)
        self.df['PROB_LEVEL'] = (np.arange(1, num+1) - (3 / 8)) / (num + (1 / 4))
        self.df['Z_ORIG'] = norm.ppf(self.df['PROB_LEVEL']) / self.kap
        self.df['Z'] = self.df.groupby('LPH')['Z_ORIG'].transform('mean')
        
        dmod = CoxPHFitter()
        dmod.fit(self.df[['Z', 'TIME', 'EVENT']], duration_col='TIME', event_col='EVENT')
        self.D = dmod.hazard_ratios_[0]
        lCI, uCI = np.exp(dmod.confidence_intervals_.values[0][:])
        return self.D, lCI, uCI
    

    def R_squared_D(self):
        if self.D is None:
            self.D = self.D_index()[0]
        sig2 = math.pi**2 / 6
        R_squared_D = (self.D**2 / self.kap**2) / (sig2 + self.D**2 / self.kap**2)
        return R_squared_D
    
    ###########################################################################################################
    
    def get_cens_surv(self):
        # inverse KMF
        kmf = KaplanMeierFitter()
        kmf.fit(self.df['TIME'], event_observed=(1-self.df['EVENT']))
        self.cens_surv = kmf.survival_function_.reset_index().rename(columns={'timeline': 'TIME', 'KM_estimate': 'CENS_SURV'})
        self.df = self.df.merge(self.cens_surv, how='left', on='TIME')


    def brier(self, at_time):
        cens_surv_at_time = self.cens_surv.loc[self.cens_surv['TIME'] <= at_time, 'CENS_SURV'].min()
        base_surv_at_time = self.base_surv.loc[self.base_surv.index <= at_time].min()
        self.df['SURV'] = np.power(base_surv_at_time, np.exp(self.df['LPH']))
        self.df['BRIER_1'] = ((self.df['SURV']**2) * self.df['EVENT'] * (self.df['TIME'] <= at_time).astype(int)) / self.df['CENS_SURV']
        self.df['BRIER_2'] = (((1-self.df['SURV'])**2) * (self.df['TIME'] > at_time).astype(int)) / cens_surv_at_time    
        brier = (self.df['BRIER_1'].sum() + self.df['BRIER_2'].sum()) / len(self.df.index)
        return brier


    def brier_score(self, at_time):
        if self.cens_surv is None:
            self.get_cens_surv()
        return self.brier(at_time)


    def integrated_brier_score(self):
        if self.cens_surv is None:
            self.get_cens_surv()
        self.cens_surv['BRIER'] = 0.
        
        for index, row in self.cens_surv.iterrows():
            if (index != 0):
                self.cens_surv.at[index, 'BRIER'] = self.brier(row['TIME'])

        # integrate
        total_time = self.cens_surv['TIME'].max() - self.cens_surv['TIME'].min()
        ibs = (self.cens_surv['TIME'].diff() * self.cens_surv['BRIER']).sum() / total_time
        return ibs  
    
    ###########################################################################################################
    
    def concordance_index(self):    
        return concordance_index(self.df['TIME'], self.df['LPH'], self.df['EVENT'])
    
    
    
    
