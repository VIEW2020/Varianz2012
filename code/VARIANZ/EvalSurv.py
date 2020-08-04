"""
D statistic & R-squared-D
Date: 2nd August 2020
Author: C Bharat

References:
Royston & Sauerbrei (2004), "A new measure of prognostic separation in survival data", Stats In Med

Notes:
D statistic requires only the prognostic index/log-partial hazard from the fitted model
Can correct D for optimisim when model is fit and D is obtained from the same data set (not yet coded, see p729 of ref)
Can also stratify the model
"""

# Libraries
from lifelines import CoxPHFitter
from scipy.stats import norm
import numpy as np
import pandas as pd
import math

class EvalDstat:
    """Class for evaluating predictions.

    Arguments:
        lph {Panda.series} -- log of the partial hazard (linear predictor) from fitted model.
        durations {Panda.series} -- Durations of test set.
        event indicator {Panda.series} -- Events of test set, 1 if event occured and 0 if censored
    """

    def __init__(self, lph, durations, events):
        self.lph = lph
        self.durations = durations
        self.events = events
        self.kap = math.sqrt(8 / math.pi)

    def Dindex(self):
        """
        Order linear predictor, durations and events in ascending order based on lph rank
        """
        oo = np.argsort(self.lph)
        h_i = [self.lph[i] for i in oo]
        stime = [self.durations[i] for i in oo]
        sevent = [self.events[i] for i in oo]
        prob_level = []
        n = len(h_i)
        for i in range(1, n + 1):
            prob_level.append((i - (3 / 8)) / (n + (1 / 4)))
        Standard_normal_quantiles = norm.ppf(prob_level)
        z = math.pow(self.kap, -1) * Standard_normal_quantiles
        df1 = pd.DataFrame({'stime': stime, 'sevent': sevent, 'h_i': h_i, 'z_orig': z})
        df1['z'] = df1.groupby('h_i')['z_orig'].transform('mean')
        df1 = df1[['stime', 'sevent', 'z']]
        dmod = CoxPHFitter()
        dmod.fit(df1, duration_col='stime', event_col='sevent')
        Dindex = dmod.hazard_ratios_.astype(float)[0]
        lCI, uCI = np.exp(dmod.confidence_intervals_.values[0][:])
        return Dindex, lCI, uCI

    def R_squared_D(self):
        sig2 = pow(math.pi, 2) / 6
        R_squared_D = (pow(self.Dindex()[0], 2) / pow(self.kap, 2)) / (sig2 + (pow(self.Dindex()[0], 2) / pow(self.kap, 2)))
        return R_squared_D