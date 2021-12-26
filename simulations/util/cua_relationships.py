'''
Author: Qingkai Kong, kongqk@berkeley.edu
Date: 2018-06-26

'''

import numpy as np
import warnings
warnings.filterwarnings("ignore")

class envelope_amplitude(object):
    
    def __init__(self):
        pass
    
    def calc_envelope_amplitude(self, M,Rjb,evdp,IM,ZH,PS,RS,Sigma):
        '''
        # Cua and Heaton 2007 relationships
        # IM = {PGA, PGV, FD}, where FD = 3 sec high pass filtered displacement
        # ZH = {Z,H}, where Z=vertical, H=horizontal
        # PS = {P, S}, P=P-wave, S=S-wave
        # RS = {Rock, Soil}, where Rock is for sites w/ NEHRP class BC and above,
        # Soil is for sites w/ NEHRP class C and below

        # note: output units are PGA (cm/s/s), PGV (cm/s), FD (cm)
        # y is median ground motion level
        # up is median + sigma
        # low is median - sigma
        # sigma is in log10 
        
        '''

        #global IM ZH PS RS
        R1=np.sqrt(Rjb**2 + evdp**2)

        a,b,c1,c2,d,e,sigma = self.__get_coeffs_CH2007__(IM,ZH,PS,RS)
        if Sigma != 0:
            sigma=Sigma
            
        CM=c1*np.exp(c2*(M-5))*(np.arctan(M-5)+1.4)
        log10Y= a*M - b*(R1+CM) - d*np.log10(R1+CM) + e
        logup=log10Y + sigma
        loglow=log10Y - sigma

        #y=pow(10, log10Y)/100./9.81
        #up=pow(10, logup)/100./9.81
        #low=pow(10, loglow)/100./9.81

        return log10Y, sigma
    
    def __get_coeffs_CH2007__(self, IM, ZH, PS, RS):
        if IM == 'PGA' and ZH =='H' and PS == 'P' and RS == 'R':
            a = 0.72
            b=  3.3e-3
            c1= 1.6
            c2= 1.05
            d = 1.2
            e =-1.06
            sigma=0.31

        elif IM == 'PGA' and ZH =='H' and PS == 'P' and RS == 'S':
            a=0.74
            b= 3.3e-3
            c1= 2.41
            c2= 0.95
            d = 1.26
            e =-1.05
            sigma= 0.29

        elif IM == 'PGV' and ZH =='H' and PS == 'P' and RS == 'R':
            a=0.80
            b=8.4e-4
            c1=0.76
            c2=1.03
            d=1.24
            e=-3.103
            sigma=0.27

        elif IM == 'PGV' and ZH =='H' and PS == 'P' and RS == 'S':
            a=0.84
            b=5.4e-4
            c1=1.21
            c2=0.97
            d= 1.28
            e= -3.13
            sigma=0.26

        elif IM == 'FD' and ZH =='H' and PS == 'P' and RS == 'R':
            a=0.95
            b=1.7e-7
            c1=2.16
            c2=1.08
            d=1.27
            e=-4.96
            sigma=0.28

        elif IM == 'FD' and ZH =='H' and PS == 'P' and RS == 'S':
            a=0.94
            b=5.17e-7
            c1=2.26
            c2=1.02
            d=1.16
            e=-5.01
            sigma=0.3

        elif IM == 'PGA' and ZH =='H' and PS == 'S' and RS == 'R':
            a=0.733
            b=7.216e-4
            d=1.48
            c1=1.16
            c2=0.96
            e=-0.4202
            sigma=0.3069

        elif IM == 'PGA' and ZH =='H' and PS == 'S' and RS == 'S':
            a=0.709
            b=2.3878e-3
            d=1.4386
            c1=1.722
            c2=0.9560
            e=-2.4525e-2
            sigma=0.3261

        elif IM == 'PGV' and ZH =='H' and PS == 'S' and RS == 'R':
            a=0.861988
            b=5.578e-4
            d=1.36760
            c1=0.8386
            c2=0.98
            e=-2.58053
            sigma=0.2773

        elif IM == 'PGV' and ZH =='H' and PS == 'S' and RS == 'S':
            a=0.88649
            b=8.4e-4
            d=1.4729
            c1=1.39
            c2=0.95
            e=-2.2498
            sigma=0.3193

        elif IM == 'FD' and ZH =='H' and PS == 'S' and RS == 'R':
            a=1.03
            b=1.01e-7
            c1=1.09
            c2=1.13
            d=1.43
            e=-4.34
            sigma=0.27

        elif IM == 'FD' and ZH =='H' and PS == 'S' and RS == 'S':
            a=1.08
            b=1.2e-6
            c1=1.95
            c2=1.09
            d=1.56
            e=-4.1
            sigma=0.32

        elif IM == 'PGA' and ZH =='Z' and PS == 'P' and RS == 'R':
            a=0.74
            b=4.01e-3
            c1=1.75
            c2=1.09
            d=1.2
            e=-0.96
            sigma=0.29

        elif IM == 'PGA' and ZH =='Z' and PS == 'P' and RS == 'S':
            a=0.74
            b=5.17e-7
            c1=2.03
            c2=0.97
            d=1.2
            e=-0.77
            sigma=0.31

        elif IM == 'PGV' and ZH =='Z' and PS == 'P' and RS == 'R':
            a=0.82
            b=8.54e-4
            c1=1.14
            c2=1.11
            d=1.36
            e=-2.90057
            sigma=0.26

        elif IM == 'PGV' and ZH =='Z' and PS == 'P' and RS == 'S':
            a=0.81
            b=2.65e-6
            c1=1.4
            c2=1.0
            d=1.48
            e=-2.55
            sigma=0.30

        elif IM == 'FD' and ZH =='Z' and PS == 'P' and RS == 'R':
            a=0.96
            b=1.98e-6
            c1=1.66
            c2=1.16
            d=1.34
            e=-4.79
            sigma=0.28

        elif IM == 'FD' and ZH =='Z' and PS == 'P' and RS == 'S':
            a=0.93
            b=1.09e-7
            c1=1.5
            c2=1.04
            d=1.23
            e=-4.74
            sigma=0.31

        elif IM == 'PGA' and ZH =='Z' and PS == 'S' and RS == 'R':
            a= 0.78
            b=2.7e-3
            c1=1.76
            c2=1.11
            d=1.38
            e=-0.75
            sigma=0.30

        elif IM == 'PGA' and ZH =='Z' and PS == 'S' and RS == 'S':
            a=0.75
            b=2.47e-3
            c1=1.59
            c2=1.01
            d=1.47
            e=-0.36
            sigma=0.30

        elif IM == 'PGV' and ZH =='Z' and PS == 'S' and RS == 'R':
            a=0.90
            b=1.03e-3
            c1=1.39
            c2=1.09
            d= 1.51
            e=-2.78
            sigma=0.25

        elif IM == 'PGV' and ZH =='Z' and PS == 'S' and RS == 'S':
            a=0.88
            b=5.41e-4
            c1=1.53
            c2=1.04
            d=1.48
            e=-2.54
            sigma=0.27

        elif IM == 'FD' and ZH =='Z' and PS == 'S' and RS == 'R':
            a=1.04
            b=1.12e-5
            c1=1.38
            c2=1.18
            d=1.37
            e=-4.74
            sigma=0.25

        elif IM == 'FD' and ZH =='Z' and PS == 'S' and RS == 'S':
            a=1.04
            b=4.92e-6
            c1=1.55
            c2=1.08
            d=1.36
            e=-4.57
            sigma=0.28
        return a,b,c1,c2,d,e,sigma
    
class envelope_generator(envelope_amplitude):
    
    def __init__(self):
        self.coef = {}
        self.env_param = {}
        
    def calc_envelope(self, M, R, evdp, vp, vs, IM, ZH, PS, RS, sampling_rate, len_after_decay):
        
        '''
        # Cua and Heaton 2007 relationships
        # IM = {PGA, PGV, FD}, where FD = 3 sec high pass filtered displacement
        # ZH = {Z,H}, where Z=vertical, H=horizontal
        # PS = {P, S}, P=P-wave, S=S-wave
        # RS = {Rock, Soil}, where Rock is for sites w/ NEHRP class BC and above,
        # Soil is for sites w/ NEHRP class C and below
        '''
        Sigma = 0
        # get all the coef to calculate the envelope parameter
        self.__get_env_param_coef__(IM, ZH, PS, RS)
        
        # get all 4 parameters for calculate the envelope
        tr, delta_t, tau, gamma = self.__get_env_param__(M, R, IM, ZH, PS, RS)
        
        # calculate the amplitude associate with the wave
        # Note that in Cua's relationship, the depth is fixed at 9.0 km
        log10AMP, sigmaAMP = self.calc_envelope_amplitude(M,R,evdp,IM,ZH,PS,RS,Sigma)
        
        A = 10**log10AMP / 100.
        self.env_param['A'] = A
        
        # get the phase arrival time
        t0 = self.__get_arrival_time__(R, evdp, vp, vs, PS)
        
        # get the envelope, right now, we use a 0.1 s 
        t, env = self.__get_envelope__(tr, delta_t, tau, gamma, A, t0,sampling_rate, len_after_decay)
        
        return t, env
        
    def __get_envelope__(self, tr, delta_t, tau, gamma, A, t0, sampling_rate, len_after_decay):
                
        # the end of the envelope, let's add len_after_decay sec
        t_end = t0 + tr + delta_t + len_after_decay
        
        t = np.arange(0, t_end, 1./sampling_rate)
        
        envelope = np.zeros(len(t))
        
        ix = t < t0
        envelope[ix] = 0
        
        ix = (t>=t0) & (t<t0+tr)
        envelope[ix] = A/tr*(t[ix] - t0)
        
        ix = (t>=t0+tr) & (t<t0+tr+delta_t)
        envelope[ix] = A
        
        ix = t>=t0+tr+delta_t
        envelope[ix] = A/((t[ix]-t0-tr-delta_t+tau)**gamma)
        
        return t, envelope
        
        
    def __get_arrival_time__(self, R, evdp, vp, vs, PS):
        
        dist_hypo = np.sqrt(R**2+evdp**2)
        
        if PS == 'P':
            t = dist_hypo / vp
        elif PS == 'S':
            t = dist_hypo / vs
        
        return t
        
    def __get_env_param__(self, M, R, IM, ZH, PS, RS):
        # the equation
        env_param = lambda alpha, beta, eps, mu: 10**(alpha*M + beta*R + eps*np.log10(R) + mu)
        
        tr = env_param(self.coef['tr']['alpha'], self.coef['tr']['beta'], 
                     self.coef['tr']['eps'], self.coef['tr']['mu'])
        
        delta_t = env_param(self.coef['delta_t']['alpha'], self.coef['delta_t']['beta'], 
                     self.coef['delta_t']['eps'], self.coef['delta_t']['mu'])
        
        tau = env_param(self.coef['tau']['alpha'], self.coef['tau']['beta'], 
                     self.coef['tau']['eps'], self.coef['tau']['mu'])
        
        gamma = env_param(self.coef['gamma']['alpha'], self.coef['gamma']['beta'], 
                     self.coef['gamma']['eps'], self.coef['gamma']['mu'])
        
        self.env_param['tr'] = tr
        self.env_param['delta_t'] = delta_t
        self.env_param['tau'] = tau
        self.env_param['gamma'] = gamma
        
        return tr, delta_t, tau, gamma
        
        # do you want to calculate the uncertainties? Then use the sigma
        
    
    def __get_env_param_coef__(self, IM, ZH, PS, RS):
                
        if IM == 'PGA' and ZH =='H' and PS == 'P' and RS == 'R':
            alpha_tr = 0.06
            beta_tr = 5.5e-4
            eps_tr = 0.27
            mu_tr = -0.37
            sigma_tr = 0.22
            
            alpha_delta = 0.0
            beta_delta = 2.58e-3
            eps_delta = 0.21
            mu_delta = -0.22
            sigma_delta = 0.39
            
            alpha_tau = 0.047
            beta_tau = 0.0
            eps_tau = 0.48
            mu_tau = -0.75
            sigma_tau = 0.28
            
            alpha_gamma = -0.032
            beta_gamma = -1.81e-3
            eps_gamma = -0.1
            mu_gamma = 0.64
            sigma_gamma = 0.16
            
        if IM == 'PGA' and ZH =='H' and PS == 'P' and RS == 'S':
            alpha_tr = 0.07
            beta_tr = 1.2e-3
            eps_tr = 0.24
            mu_tr = -0.38
            sigma_tr = 0.26
            
            alpha_delta = 0.03
            beta_delta = 2.37e-3
            eps_delta = 0.39
            mu_delta = -0.59
            sigma_delta = 0.36
            
            alpha_tau = 0.087
            beta_tau = -1.89e-3
            eps_tau = 0.58
            mu_tau = -0.87
            sigma_tau = 0.31
            
            alpha_gamma = -0.048  # this might be another typo, shouldn't be 0.48
            beta_gamma = -1.42e-3
            eps_gamma = -0.13
            mu_gamma = 0.71
            sigma_gamma = 0.21
            
        if IM == 'PGV' and ZH =='H' and PS == 'P' and RS == 'R':
            alpha_tr = 0.06
            beta_tr = 1.33e-3
            eps_tr = 0.23
            mu_tr = -0.34
            sigma_tr = 0.25
            
            alpha_delta = 0.054
            beta_delta = 1.93e-3
            eps_delta = 0.16
            mu_delta = -0.36
            sigma_delta = 0.40
            
            alpha_tau = 1.86e-2
            beta_tau = 5.37e-5
            eps_tau = 0.41
            mu_tau = -0.51
            sigma_tau = 0.3
            
            alpha_gamma = -0.044
            beta_gamma = -1.65e-3
            eps_gamma = -0.16
            mu_gamma = 0.72
            sigma_gamma = 0.20
            
        if IM == 'PGV' and ZH =='H' and PS == 'P' and RS == 'S':
            alpha_tr = 0.07
            beta_tr = 4.35e-4
            eps_tr = 0.47
            mu_tr = -0.68
            sigma_tr = 0.26
            
            alpha_delta = 0.03
            beta_delta = 2.03e-3
            eps_delta = 0.289
            mu_delta = -0.45
            sigma_delta = 0.40
            
            alpha_tau = 0.0403
            beta_tau = -1.26e-3
            eps_tau = 0.387
            mu_tau = -0.372
            sigma_tau = 0.37
            
            alpha_gamma = -6.17e-2
            beta_gamma = -2.0e-3
            eps_gamma = 0.0
            mu_gamma = 0.578
            sigma_gamma = 0.25
            
        if IM == 'FD' and ZH =='H' and PS == 'P' and RS == 'R':
            alpha_tr = 0.05
            beta_tr = 1.29e-3
            eps_tr = 0.27
            mu_tr = -0.34
            sigma_tr = 0.28
            
            alpha_delta = 0.047
            beta_delta = 0.0
            eps_delta = 0.45
            mu_delta = -0.68
            sigma_delta = 0.43
            
            alpha_tau = 0.0
            beta_tau = 0.0
            eps_tau = 0.19
            mu_tau = -0.07
            sigma_tau = 0.39
            
            alpha_gamma = -0.062
            beta_gamma = -2.3e-3
            eps_gamma = 0.0
            mu_gamma = 0.61
            sigma_gamma = 0.26
            
        if IM == 'FD' and ZH =='H' and PS == 'P' and RS == 'S':
            alpha_tr = 0.05
            beta_tr = 1.19e-3
            eps_tr = 0.47
            mu_tr = -0.58
            sigma_tr = 0.26
            
            alpha_delta = 0.051
            beta_delta = 1.12e-3
            eps_delta = 0.33
            mu_delta = -0.59
            sigma_delta = 0.41
            
            alpha_tau = 0.035
            beta_tau = -1.27e-3
            eps_tau = 0.19
            mu_tau = 0.03
            sigma_tau = 0.43
            
            alpha_gamma = -0.061
            beta_gamma = -1.9e-3
            eps_gamma = 0.11
            mu_gamma = 0.39
            sigma_gamma = 0.31
            
        if IM == 'PGA' and ZH =='H' and PS == 'S' and RS == 'R':
            alpha_tr = 0.064  # This is maybe a typo, it should be 0.064 instead of 0.64
            beta_tr = 0.0
            eps_tr = 0.48
            mu_tr = -0.89
            sigma_tr = 0.23
            
            alpha_delta = 0.0
            beta_delta = -4.87e-4
            eps_delta = 0.13
            mu_delta = 0.0024
            sigma_delta = 0.2
            
            alpha_tau = 0.037
            beta_tau = 0.0
            eps_tau = 0.39
            mu_tau = -0.59
            sigma_tau = 0.18
            
            alpha_gamma = -0.014
            beta_gamma = -5.28e-4
            eps_gamma = -0.11
            mu_gamma = 0.26
            sigma_gamma = 0.09
            
        if IM == 'PGA' and ZH =='H' and PS == 'S' and RS == 'S':
            alpha_tr = 0.055
            beta_tr = 1.21e-3
            eps_tr = 0.34
            mu_tr = -0.66
            sigma_tr = 0.25
            
            alpha_delta = 0.028
            beta_delta = 0.0
            eps_delta = 0.07
            mu_delta = -0.102
            sigma_delta = 0.23
            
            alpha_tau = 0.0557
            beta_tau = -8.2e-4
            eps_tau = 0.51
            mu_tau = -0.68
            sigma_tau = 0.24
            
            alpha_gamma = -0.015
            beta_gamma = -5.89e-4
            eps_gamma = -0.163
            mu_gamma = 0.23
            sigma_gamma = 0.13
            
        if IM == 'PGV' and ZH =='H' and PS == 'S' and RS == 'R':
            alpha_tr = 0.093
            beta_tr = 0.0
            eps_tr = 0.48
            mu_tr = -0.96
            sigma_tr = 0.25
            
            alpha_delta = 0.02
            beta_delta = 0.0
            eps_delta = 0.0
            mu_delta = 0.046
            sigma_delta = 0.23
            
            alpha_tau = 0.029
            beta_tau = 8.0e-4
            eps_tau = 0.25
            mu_tau = -0.31
            sigma_tau = 0.23
            
            alpha_gamma = -0.024
            beta_gamma = -1.02e-3
            eps_gamma = -0.06
            mu_gamma = 0.21
            sigma_gamma = 0.11
            
        if IM == 'PGV' and ZH =='H' and PS == 'S' and RS == 'S':
            alpha_tr = 0.087
            beta_tr = 4.0e-4
            eps_tr = 0.49
            mu_tr = -0.98
            sigma_tr = 0.30
            
            alpha_delta = 0.028
            beta_delta = 0.0
            eps_delta = 0.05
            mu_delta = -0.08
            sigma_delta = 0.23
            
            alpha_tau = 0.045
            beta_tau = -5.46e-4
            eps_tau = 0.46
            mu_tau = -0.55
            sigma_tau = 0.25
            
            alpha_gamma = -0.031
            beta_gamma = -4.61e-4
            eps_gamma = -0.162
            mu_gamma = 0.30
            sigma_gamma = 0.13
            
        if IM == 'FD' and ZH =='H' and PS == 'S' and RS == 'R':
            alpha_tr = 0.109
            beta_tr = 7.68e-4
            eps_tr = 0.38
            mu_tr = -0.87
            sigma_tr = 0.29
            
            alpha_delta = 0.04
            beta_delta = 1.1e-3
            eps_delta = -0.15
            mu_delta = 0.11
            sigma_delta = 0.23
            
            alpha_tau = 0.029
            beta_tau = 0.0
            eps_tau = 0.36
            mu_tau = -0.38
            sigma_tau = 0.26
            
            alpha_gamma = -0.025
            beta_gamma = -4.22e-4
            eps_gamma = -0.145
            mu_gamma = 0.262
            sigma_gamma = 0.12
            
        if IM == 'FD' and ZH =='H' and PS == 'S' and RS == 'S':
            alpha_tr = 0.12
            beta_tr = 0.0
            eps_tr = 0.45
            mu_tr = -0.89
            sigma_tr = 0.34
            
            alpha_delta = 0.03
            beta_delta = 0.0
            eps_delta = 0.037
            mu_delta = -0.066
            sigma_delta = 0.28
            
            alpha_tau = 0.038
            beta_tau = -1.34e-3
            eps_tau = 0.48
            mu_tau = -0.39
            sigma_tau = 0.30
            
            alpha_gamma = -2.67e-2
            beta_gamma = 2.0e-4
            eps_gamma = -0.22
            mu_gamma = 0.27
            sigma_gamma = 0.14
            
        if IM == 'PGA' and ZH =='Z' and PS == 'P' and RS == 'R':
            alpha_tr = 0.06
            beta_tr = 7.45e-4
            eps_tr = 0.37
            mu_tr = -0.51
            sigma_tr = 0.22
            
            alpha_delta = 0.0
            beta_delta = 2.75e-3
            eps_delta = 0.17
            mu_delta = -0.24
            sigma_delta = 0.41
            
            alpha_tau = 0.03
            beta_tau = 0.0
            eps_tau = 0.58
            mu_tau = -0.97
            sigma_tau = 0.26
            
            alpha_gamma = -0.027
            beta_gamma = -1.75e-3
            eps_gamma = -0.18
            mu_gamma = 0.74
            sigma_gamma = 0.15
            
        if IM == 'PGA' and ZH =='Z' and PS == 'P' and RS == 'S':
            alpha_tr = 0.06
            beta_tr = 5.87e-4
            eps_tr = 0.23
            mu_tr = -0.37
            sigma_tr = 0.23
            
            alpha_delta = 0.0
            beta_delta = 1.76e-3
            eps_delta = 0.36
            mu_delta = -0.48
            sigma_delta = 0.41
            
            alpha_tau = 0.057
            beta_tau = -1.36e-3
            eps_tau = 0.63
            mu_tau = -0.96
            sigma_tau = 0.28
            
            alpha_gamma = -0.024
            beta_gamma = -1.6e-3
            eps_gamma = -0.24
            mu_gamma = 0.84
            sigma_gamma = 0.18
            
        if IM == 'PGV' and ZH =='Z' and PS == 'P' and RS == 'R':
            alpha_tr = 0.06
            beta_tr = 7.32e-4
            eps_tr = 0.25
            mu_tr = -0.37
            sigma_tr = 0.26
            
            alpha_delta = 0.046
            beta_delta = 2.61e-3
            eps_delta = 0.0
            mu_delta = -0.21
            sigma_delta = 0.41
            
            alpha_tau = 0.03
            beta_tau = 8.6e-4
            eps_tau = 0.35
            mu_tau = -0.62
            sigma_tau = 0.29
            
            alpha_gamma = -0.039
            beta_gamma = -1.9e-3
            eps_gamma = -0.18
            mu_gamma = 0.76
            sigma_gamma = 0.18
            
        if IM == 'PGV' and ZH =='Z' and PS == 'P' and RS == 'S':
            alpha_tr = 0.06
            beta_tr = 1.1e-3
            eps_tr = 0.22
            mu_tr = -0.36
            sigma_tr = 0.24
            
            alpha_delta = 0.031
            beta_delta = 1.7e-3
            eps_delta = 0.26
            mu_delta = -0.52
            sigma_delta = 0.42
            
            alpha_tau = 0.031       # another typo? 0.031 instead of 0.31
            beta_tau = -6.4e-4
            eps_tau = 0.44
            mu_tau = -0.55
            sigma_tau = 0.32
            
            alpha_gamma = -0.037
            beta_gamma = -2.23e-3
            eps_gamma = -0.14
            mu_gamma = 0.71
            sigma_gamma = 0.22
            
        if IM == 'FD' and ZH =='Z' and PS == 'P' and RS == 'R':
            alpha_tr = 0.08
            beta_tr = 1.63e-3
            eps_tr = 0.13
            mu_tr = -0.33
            sigma_tr = 0.27
            
            alpha_delta = 0.058
            beta_delta = 2.02e-3
            eps_delta = 0.0
            mu_delta = -0.25
            sigma_delta = 0.42
            
            alpha_tau = 0.05
            beta_tau = 8.9e-4
            eps_tau = 0.16
            mu_tau = -0.39
            sigma_tau = 0.36
            
            alpha_gamma = -0.052
            beta_gamma = 1.67e-3
            eps_gamma = -0.21
            mu_gamma = 0.85
            sigma_gamma = 0.22
            
        if IM == 'FD' and ZH =='Z' and PS == 'P' and RS == 'S':
            alpha_tr = 0.067
            beta_tr = 1.21e-3
            eps_tr = 0.28
            mu_tr = -0.46
            sigma_tr = 0.27
            
            alpha_delta = 0.043
            beta_delta = 9.94e-4
            eps_delta = 0.19
            mu_delta = -0.42
            sigma_delta = 0.41
            
            alpha_tau = 0.052
            beta_tau = 0.0
            eps_tau = 0.12
            mu_tau = -0.17
            sigma_tau = 0.39
            
            alpha_gamma = -0.7
            beta_gamma = -2.5e-3
            eps_gamma = 0.0
            mu_gamma = 0.63
            sigma_gamma = 0.27
            
        if IM == 'PGA' and ZH =='Z' and PS == 'S' and RS == 'R':
            alpha_tr = 0.069
            beta_tr = 0.0
            eps_tr = 0.49
            mu_tr = -0.97
            sigma_tr = 0.23
            
            alpha_delta = 0.03
            beta_delta = -1.4e-3
            eps_delta = 0.22
            mu_delta = -0.17
            sigma_delta = 0.20
            
            alpha_tau = 0.031
            beta_tau = 0.0
            eps_tau = 0.34
            mu_tau = -0.44
            sigma_tau = 0.19
            
            alpha_gamma = 0.015
            beta_gamma = -4.64e-4
            eps_gamma = -0.12
            mu_gamma = 0.26
            sigma_gamma = 0.095
            
        if IM == 'PGA' and ZH =='Z' and PS == 'S' and RS == 'S':
            alpha_tr = 0.059
            beta_tr = 2.18e-3
            eps_tr = 0.26
            mu_tr = -0.66
            sigma_tr = 0.25
            
            alpha_delta = 0.03
            beta_delta = -1.78e-3
            eps_delta = 0.31
            mu_delta = -0.31
            sigma_delta = 0.25
            
            alpha_tau = 0.06
            beta_tau = -1.45e-3
            eps_tau = 0.51
            mu_tau = -0.6
            sigma_tau = 0.22
            
            alpha_gamma = -0.02
            beta_gamma = 0.0
            eps_gamma = -0.24
            mu_gamma = 0.38
            sigma_gamma = 0.13
            
        if IM == 'PGV' and ZH =='Z' and PS == 'S' and RS == 'R':
            alpha_tr = 0.12
            beta_tr = 0.0
            eps_tr = 0.50
            mu_tr = -1.14
            sigma_tr = 0.27
            
            alpha_delta = 0.018
            beta_delta = 0.0
            eps_delta = 0.0
            mu_delta = -0.072
            sigma_delta = 0.23
            
            alpha_tau = 0.04
            beta_tau = 9.4e-4
            eps_tau = 0.25
            mu_tau = -0.34
            sigma_tau = 0.23
            
            alpha_gamma = -0.028
            beta_gamma = -8.32e-4
            eps_gamma = -0.12
            mu_gamma = 0.32
            sigma_gamma = 0.11
            
        if IM == 'PGV' and ZH =='Z' and PS == 'S' and RS == 'S':
            alpha_tr = 0.11
            beta_tr = 1.24e-3
            eps_tr = 0.38
            mu_tr = -0.91
            sigma_tr = 0.31
            
            alpha_delta = 0.017
            beta_delta = -6.93e-4
            eps_delta = 0.12
            mu_delta = -0.05
            sigma_delta = 0.27
            
            alpha_tau = 0.051
            beta_tau = -1.41e-3
            eps_tau = 0.44
            mu_tau = -0.37
            sigma_tau = 0.26
            
            alpha_gamma = -0.03
            beta_gamma = 0.0
            eps_gamma = -0.21
            mu_gamma = 0.33
            sigma_gamma = 0.15
            
        if IM == 'FD' and ZH =='Z' and PS == 'S' and RS == 'R':
            alpha_tr = 0.12
            beta_tr = 1.3e-3
            eps_tr = 0.26
            mu_tr = -0.75
            sigma_tr = 0.30
            
            alpha_delta = 0.03
            beta_delta = 2.6e-4
            eps_delta = 0.0
            mu_delta = -0.02
            sigma_delta = 0.25
            
            alpha_tau = 0.02
            beta_tau = 0.0
            eps_tau = 0.30
            mu_tau = -0.22
            sigma_tau = 0.26
            
            alpha_gamma = -0.02
            beta_gamma = 0.0
            eps_gamma = -0.23
            mu_gamma = 0.31
            sigma_gamma = 0.12
            
        if IM == 'FD' and ZH =='Z' and PS == 'S' and RS == 'S':
            alpha_tr = 0.12 
            beta_tr = 0.0
            eps_tr = 0.44
            mu_tr = -0.82
            sigma_tr = 0.40
            
            alpha_delta = 0.02
            beta_delta = -7.18e-4
            eps_delta = 0.07
            mu_delta = -0.005
            sigma_delta = 0.26
            
            alpha_tau = 0.022
            beta_tau = -1.65e-3
            eps_tau = 0.44
            mu_tau = -0.19
            sigma_tau = 0.28
            
            alpha_gamma = -0.018
            beta_gamma = 5.65e-4
            eps_gamma = -0.25
            mu_gamma = 0.24
            sigma_gamma = 0.14
        
        self.coef['tr'] = {}
        self.coef['tr']['alpha'] = alpha_tr
        self.coef['tr']['beta'] = beta_tr
        self.coef['tr']['eps'] = eps_tr
        self.coef['tr']['mu'] = mu_tr
        self.coef['tr']['sigma'] = sigma_tr
        
        self.coef['delta_t'] = {}
        self.coef['delta_t']['alpha'] = alpha_delta
        self.coef['delta_t']['beta'] = beta_delta
        self.coef['delta_t']['eps'] = eps_delta
        self.coef['delta_t']['mu'] = mu_delta
        self.coef['delta_t']['sigma'] = sigma_delta
        
        self.coef['tau'] = {}
        self.coef['tau']['alpha'] = alpha_tau
        self.coef['tau']['beta'] = beta_tau
        self.coef['tau']['eps'] = eps_tau
        self.coef['tau']['mu'] = mu_tau
        self.coef['tau']['sigma'] = sigma_tau
        
        self.coef['gamma'] = {}
        self.coef['gamma']['alpha'] = alpha_gamma
        self.coef['gamma']['beta'] = beta_gamma
        self.coef['gamma']['eps'] = eps_gamma
        self.coef['gamma']['mu'] = mu_gamma
        self.coef['gamma']['sigma'] = sigma_gamma
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-poster')
    
    # If you run directly, it gives you an example
    
    env_gen = envelope_generator()
    M = 7.5
    R = 30
    evdp = 9
    IM = 'PGA'
    ZH = 'H'
    PS = 'S'
    RS = 'R'
    Sigma = 0

    vp = 6.10
    vs = 3.55

    p = env_gen.calc_envelope(M, R, evdp, vp, vs, IM, ZH, 'P', RS, 10, 15)
    s = env_gen.calc_envelope(M, R, evdp, vp, vs, IM, ZH, 'S', RS, 10, 15)
    
    # print out the PGA value for the Horizontal component
    amp = envelope_amplitude()
    p_0 = amp.calc_envelope_amplitude(M,R,evdp,'PGA',ZH = 'H',PS = 'P',RS = 'R', Sigma = 0)
    s_0 = amp.calc_envelope_amplitude(M,R,evdp,'PGA',ZH = 'H',PS = 'S',RS = 'R', Sigma = 0)
    
    print('P and S PGA at this location is %.2f, %.2f'%(p_0, s_0))

    ## Combine the P and S together

    n_p = len(p)
    n_s = len(s)

    if n_p > n_s:
        N = n_p - n_s
        s = np.pad(s, (0, N), 'constant')
    elif n_p < n_s:
        N = n_s - n_p
        p = np.pad(p, (0, N), 'constant')

    c = p + s
    
    plt.figure(figsize = (12, 8))
    plt.subplot(211)
    plt.plot(p, label = 'P wave envelope')
    plt.plot(s, label = 'S wave envelope')
    plt.ylabel('Acceleration $(m/s^2)$')
    plt.legend()
    
    plt.subplot(212)
    plt.plot(c, label = 'combined envelope')

    plt.xlabel('Time (sec)')
    plt.ylabel('Acceleration $(m/s^2)$')
    plt.legend()
    
    plt.show()