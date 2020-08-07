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
from sklearn.metrics import auc, roc_curve
from scipy.stats import norm
import numpy as np
import pandas as pd
import math

from pdb import set_trace as bp

class EvalSurv:
    """Class for evaluating survival models.

    Arguments: df {Pandas.DataFrame} -- dataframe with columns LPH (log of the partial hazard - linear predictor - from fitted model),
        TIME and EVENT (events of test set, 1 if event occured and 0 if censored)
        base_surv {Pandas.Series} -- baseline survival at different times
    """

    def __init__(self, df):
        if set(['LPH', 'TIME', 'EVENT']) <= set(df.columns):
            self.df = df
            if 'PARTIAL_HAZARD' not in list(df.columns):
                self.df['PARTIAL_HAZARD'] = np.exp(self.df['LPH'])
            self.base_surv = None
            self.kap = math.sqrt(8 / math.pi)
            self.D = None
            self.cens_surv = None
        else:
            raise('Expected columns in df: LPH, TIME, EVENT')


    def compute_baseline_survival(self):
        df = self.df[['TIME', 'EVENT', 'PARTIAL_HAZARD']]
        df = df.groupby(['TIME']).sum().sort_index(ascending=False)
        df['CUM_PARTIAL_HAZARD'] = df['PARTIAL_HAZARD'].cumsum()
        df = df[df['EVENT']>0]
        df['ALPHA'] = np.exp(-df['EVENT']/df['CUM_PARTIAL_HAZARD'])
        df.sort_index(inplace=True)
        df['S0'] = df['ALPHA'].cumprod()
        self.base_surv = df['S0']
        

    def get_base_surv(self, at_time):
        if self.base_surv is None:
            self.compute_baseline_survival()
        return self.base_surv.loc[self.base_surv.index <= at_time].min()
    
    
    def get_surv(self, at_time):
        return np.power(self.get_base_surv(at_time), self.df['PARTIAL_HAZARD'])


    def get_risk(self, at_time):
        return 1 - self.get_surv(at_time)
    
    
    def get_risk_perc(self, at_time):
        return 100 * get_risk(at_time)
        

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
        self.D = dmod.params_[0]
        lCI, uCI = np.exp(dmod.confidence_intervals_.values[0][:])
        return self.D, lCI, uCI
    

    def R_squared_D(self):
        if self.D is None:
            self.D = self.D_index()[0]
        sig2 = math.pi**2 / 6
        R_squared_D = (self.D**2 / self.kap**2) / (sig2 + self.D**2 / self.kap**2)
        return R_squared_D
    
    ###########################################################################################################
    
    def compute_cens_surv(self):
        # inverse KMF
        kmf = KaplanMeierFitter()
        kmf.fit(self.df['TIME'], event_observed=(1-self.df['EVENT']))
        self.cens_surv = kmf.survival_function_.rename_axis('TIME').rename(columns={'KM_estimate': 'CENS_SURV'})
        self.df = self.df.merge(self.cens_surv.reset_index(), how='left', on='TIME')


    def get_cens_surv(self, at_time):
        if self.cens_surv is None:
            self.compute_cens_surv()
        return self.cens_surv.loc[self.cens_surv.index <= at_time, 'CENS_SURV'].min()


    def brier(self, at_time):
        cens_surv_at_time = self.get_cens_surv(at_time)
        self.df['SURV'] = self.get_surv(at_time)
        self.df['BRIER_1'] = ((self.df['SURV']**2) * self.df['EVENT'] * (self.df['TIME'] <= at_time).astype(int)) / self.df['CENS_SURV']
        self.df['BRIER_2'] = (((1-self.df['SURV'])**2) * (self.df['TIME'] > at_time).astype(int)) / cens_surv_at_time    
        brier = (self.df['BRIER_1'].sum() + self.df['BRIER_2'].sum()) / len(self.df.index)
        return brier


    def brier_score(self, at_time):
        if self.cens_surv is None:
            self.compute_cens_surv()
        return self.brier(at_time)


    def integrated_brier_score(self):
        if self.cens_surv is None:
            self.compute_cens_surv()
        self.cens_surv['BRIER'] = 0.
        for index, row in self.cens_surv.iterrows():
            self.cens_surv.at[index, 'BRIER'] = self.brier(index)
        # integrate
        total_time = self.cens_surv.index.max() - self.cens_surv.index.min()
        ibs = np.trapz(self.cens_surv['BRIER'], self.cens_surv.index) / total_time
        return ibs  
    
    ###########################################################################################################
    
    def concordance_index(self):    
        return concordance_index(self.df['TIME'], -self.df['LPH'], self.df['EVENT'])
    
    ###########################################################################################################
    
    def auc(self, at_time):
        fpr, tpr, thresholds = roc_curve(self.df['EVENT'], self.get_risk(at_time))
        return auc(fpr, tpr)
        
        
        
    
    
