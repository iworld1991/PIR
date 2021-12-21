# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import root 
import copy as cp

x,y = np.random.randn(6).reshape([2,3]).shape


# + {"code_folding": [0, 8]}
def toVec(ma_coeffs,
          sigmas,
          t,
          ma_q):
    assert ma_coeffs.shape == (ma_q,)
    assert sigmas.shape == (2, t)
    return np.hstack([ma_coeffs.flatten(), sigmas.flatten()])

def toPara(vec,
           t,
           ma_q):
    assert len(vec) == 2*t + ma_q
    return vec[:ma_q-1], vec[ma_q:].reshape(2,t)


# + {"code_folding": [16, 38, 42, 54, 67, 75, 95, 121, 146, 154, 163, 167, 181, 206, 215, 247, 255, 259, 275, 295, 309, 326, 346, 352, 374, 396, 421, 439, 467, 485, 495, 505, 510, 521]}
## class of integrated moving average process, trend/cycle process allowing for serial correlation transitory shocks
class IMAProcess:
    '''
    inputs
    ------
    t: int, number of periods of the series
    process_para, dict, includes 
       - ma_coeffs: size f q for MA(q),  moving average coeffcients of transitory shocks. q = 0 by default.
       - sigmas:  size of 2 x t, draws of permanent and transitory risks from time varying volatility 
       In the case of stochastical volatility with constant transitory volatility and time 
       past-dependent permanent volatility, following parameters are used as well
       
       - rho: how persistent the innovation to permanent risks is
       - gamma: size of the innovation 
       - sigma_eps: constant transitory volatility 
    '''
    def __init__(self,
                 t = 100,                   ## length of sample period  
                 n_periods = np.array([1]), ## # of periods for time aggregation 
                 ma_coeffs = np.ones(1),    
                 sigmas = np.ones([2,100]),
                ):
        #self.process_para = process_para
        self.ma_coeffs = ma_coeffs
        self.ma_q = self.ma_coeffs.shape[0]
        self.t = t
        self.sigmas = sigmas
        self.n_agg = 1
        self.init_sigmas = np.array([0.1,0.1])
        self.init_sv = np.array([0.1,0.1])
        
        ## stochastic vol paras
        
        self.rho = 0.5
        self.gamma = 0.1
        self.sigma_eps = 0.1
        
    ## auxiliary function for ma cum sum
    def cumshocks(self,
                  shocks,
                  ma_coeffs):
        cum = []
        for i in range(len(shocks)):
            #print(shocks[i])
            #print(sum([ma_coeffs[back]*shocks[i-back] for back in range(len(ma_coeffs))]))
            cum.append(sum([ma_coeffs[back]*shocks[i-back] for back in range(len(ma_coeffs))]))
        return np.array(cum)         


        
##########
## new ###
##########
        
    def SimulateSeries(self,
                       n_sim = 200):
        t = self.t 
        ma_coeffs = self.ma_coeffs
        sigmas = self.sigmas
        ma_q = self.ma_q 
        np.random.seed(12345)                 
        p_draws = np.multiply(np.random.randn(n_sim*t).reshape([n_sim,t]), 
                              np.tile(sigmas[0,:],[n_sim,1]))  # draw permanent shocks
        np.random.seed(12342)
        t_draws = np.multiply(np.random.randn(n_sim*t).reshape([n_sim,t]), 
                              np.tile(sigmas[1,:],[n_sim,1]))  ## draw one-period transitory shocks
    
        t_draws_cum = np.array( [self.cumshocks(shocks = t_draws[i,:],
                                                ma_coeffs = ma_coeffs) 
                                 for i in range(n_sim)]
                              )
        series = np.cumsum(p_draws,axis = 1) + t_draws_cum 
        self.simulated = series
        return self.simulated 
       
    def SimulatedMoments(self):
        series = self.simulated 
        
        ## the first difference 
        diff = np.diff(series,axis=1)
        
        ## moments of first diff
        mean_diff = np.mean(diff,axis = 0)
        varcov_diff = np.cov(diff.T)
        
        self.SimMoms = {'Mean':mean_diff,
                       'Var':varcov_diff}
        return self.SimMoms
    
#########################
## estimate volatility #
########################
    
    ## simulated cross sectional vols before time aggregatio, i.e. monthly 
    
    def SimulateSVols(self,
                            n_sim = 200):
        rho = self.rho
        gamma = self.gamma
        sigma_eps = self.sigma_eps
        t = self.t
        init_sigmas = self.init_sigmas        
        sigmas_eps = sigma_eps*np.ones([n_sim,t])
        sigmas_theta = np.empty([n_sim,t])
        sigmas_theta[:,0] = 0.1
        
        random.seed(1235)
        mu_draws = gamma*np.random.randn(n_sim*t).reshape([n_sim,t]) 
        
        for i in range(n_sim):
            for j in range(t-1):
                sigmas_theta[i,j+1] = np.sqrt(np.exp(rho*np.log(sigmas_theta[i,j]**2) + mu_draws[i,j+1]))
        
        self.sigmas_theta_sim = sigmas_theta
        self.sigmas_eps_sim = sigmas_eps
        self.sigmas_sim = sigmas_theta + sigmas_eps 
    
        return self.sigmas_sim 
    
    ## time aggregated cross sectional vols
    
    def SimulateSVolsAgg(self,
                               n_agg = 12):
        
        # get the simulate monthly volatility 
        sigmas_theta_sim = self.sigmas_theta
        sigmas_eps_sim =  self.sigmas_eps
        
        ## create locations for yearly volatility 
        nsim, t = sigmas_theta_sim.shape
        sigmas_sim_agg = np.empty_like(sigmas_theta_sim)
        
        
        ## fill the volatility  
        for i in range (nsim):
            for j in range(t):
                sigma_theta_this = sum([(n_agg-k-1)**2*self.hstep_sigma_theta(sigmas_theta[i,j],k) for h in range(n_agg)])
                sigma_eps_this = n_agg**2*sigma_eps[i,j]
                sigmas_sim_agg[i,j] = sigma_theta_this + sigma_eps_this
                
                
        self.sigmas_sim_agg = sigmas_sim_agg
        return self.sigmas_sim_agg
    
    ## moms of time aggregated vols
                        
    def SimulateSVolsAggMoms(self):
        sigmas_sim_agg = self.sigmas_sim_agg
        sigmas_agg_av = np.mean(sigmas_sim_agg,
                                axis = 1)
        sigmas_agg_cov = np.cov(sigmas_sim_agg.T)
        self.sigmas_agg_sim_moms = {'Mean':sigmas_agg_av,
                                   'Var':sigmas_agg_cov}
    
    def TimeAggregate(self,
                      n_periods = 1):
        simulated = self.simulated
        t = self.t
        
        simulated_agg = np.array([np.sum(simulated[:,i-n_periods:i],axis=1) for i in range(n_periods,t+1)]).T
        self.simulated_agg = simulated_agg
        return self.simulated_agg
    
    def SimulateMomentsAgg(self):
        series_agg = self.simulated_agg 
        
        ## the first difference 
        diff = np.diff(series_agg,
                       axis = 1)
        ## moments of first diff
        mean_diff = np.mean(diff,axis = 0)
        varcov_diff = np.cov(diff.T)
        
        self.SimAggMoms = {'Mean':mean_diff,
                           'Var':varcov_diff}
        return self.SimAggMoms
    
    
##########
## new ###
##########
    def ComputeMomentsAgg(self,
                          n_agg = 1):
        sigmas = self.sigmas
        sigmas_theta = sigmas[0,:]
        sigmas_eps = sigmas[1,:]
        
        n = n_agg 
        t = self.t 
        
        t_truc = t - 2*n 
        
        ## prepare the locations for var-cov matrix 
        var_cov = np.zeros([t,t])
        
        ## prepare a (2n-1) x 1  vector [1,2...n,n-1..1]
        M_vec0 = np.arange(n-1)+1
        M_vec1 = np.flip(np.arange(n)+1)  
        M_vec =  np.concatenate((M_vec0,M_vec1))
        
        ## prepare a 2n x 1 vector [-1,-1...,1,1]
        I_vec0 = - np.ones(n)
        I_vec1 = np.ones(n)
        I_vec =  np.concatenate((I_vec0,I_vec1))
        
        for i in np.arange(t_truc)+n:
            for k in np.arange(n)+1:   ## !!!need to check here. 
                var_cov[i,i+k] = ( sum(M_vec[k:]*M_vec[:-k]*sigmas_theta[i+1-n:i+n-k]**2)
                                  + sum(I_vec[k:]*I_vec[:-k]*sigmas_eps[i-n:i+n-k]**2) ) # need to check 
                var_cov[i+k,i] = var_cov[i,i+k]
            var_cov[i,i] = sum(M_vec**2*sigmas_theta[i+1-n:i+n]**2)
        
        self.Moments_Agg = var_cov
        return self.Moments_Agg
    
    def ComputeGenMoments(self):
        ## parameters 
        t = self.t 
        ma_coeffs = self.ma_coeffs
        sigmas = self.sigmas
        p_sigmas = sigmas[0,:]
        t_sigmas = sigmas[1,:]
        ma_q = self.ma_q 
        
        ## generalized moments 
        mean_diff = np.zeros(t)[1:] 
        ## varcov is basically the variance covariance of first difference of income of this IMA(q) process
        ## Cov(delta y_t - delta y_{t+k}) forall k for all t
        varcov_diff = np.asmatrix( np.zeros((t)**2).reshape([t,t]) )
        
        for i in range(t):
            autocovf_this = p_sigmas[i]**2 + t_sigmas[i]**2 + t_sigmas[i-1]**2
            varcov_diff[i,i] = autocovf_this
            try:
                varcov_diff[i,i+1] = - t_sigmas[i]**2
                varcov_diff[i+1,i] = - t_sigmas[i]**2            
            except:
                pass
        varcov_diff = varcov_diff[1:,1:]
        self.GenMoms = {'Mean':mean_diff,
                       'Var':varcov_diff}
        return self.GenMoms
    
    def GetDataMoments(self,
                      data_moms_dct):
        self.data_moms_dct = data_moms_dct
        
    def GetDataMomentsAgg(self,
                          data_moms_agg_dct):
        self.data_moms_agg_dct = data_moms_agg_dct    

###################
#### vols 
###################

    def GetDataMomentsVolsAgg(self,
                              data_vols_moms_agg_dct):
        self.data_vols_moms_agg_dct = data_vols_moms_agg_dct 
        
    def ObjFunc(self,
                para):
        data_moms_dct = self.data_moms_dct
        t = self.t
        ma_q = self.ma_q
        ma_coeffs,sigmas = toPara(para,
                                  t,
                                  ma_q)
        self.ma_coeffs = ma_coeffs
        self.sigmas = sigmas
        model_moms_dct = self.ComputeGenMoments() 
        model_moms = np.array([model_moms_dct[key] for key in ['Var']]).flatten()
        data_moms = np.array([data_moms_dct[key] for key in ['Var']]).flatten()
        diff = np.linalg.norm(model_moms - data_moms)
        return diff
    
    def EstimatePara(self,
                     method = 'CG',
                     bounds = None,
                     para_guess = None,
                     options = {'disp':True}):
        t = self.t
        ma_q = self.ma_q
        
        para_est = minimize(self.ObjFunc,
                            x0 = para_guess,
                            method = method,
                            bounds = bounds,
                            options = options)['x']
        
        self.para_est = toPara(para_est,
                               t,
                               ma_q)
        
        return self.para_est    
    
    def EstimateParaRoot(self,
                         para_guess = None):
        t = self.t
        ma_q = self.ma_q
        
        para_est = root(self.ObjFunc,
                       x0 = para_guess)['x']
        
        self.para_est = toPara(para_est,
                               t,
                               ma_q)
        
        return self.para_est 
    
    def ObjFuncSim(self,
                   para_sim):
        data_moms_dct = self.data_moms_dct
        t = self.t
        ma_q = self.ma_q
        ma_coeffs,sigmas = toPara(para_sim,
                                  t,
                                  ma_q)
        self.ma_coeffs = ma_coeffs
        self.sigmas = sigmas
        model_series_sim = self.SimulateSeries(n_sim = 2000) 
        model_moms_dct = self.SimulatedMoments()  
        model_moms = np.array([model_moms_dct[key] for key in ['Var']]).flatten()
        data_moms = np.array([data_moms_dct[key] for key in ['Var']]).flatten()
        diff = np.linalg.norm(model_moms - data_moms)
        return diff
        
    def EstimateParabySim(self,
                          method = 'CG',
                          bounds = None,
                          para_guess = None,
                          options = {'disp':True}):
        t = self.t
        ma_q = self.ma_q
        
        para_est_sim = minimize(self.ObjFuncSim,
                                x0 = para_guess,
                                method = method,
                                bounds = bounds,
                                options = options)['x']
        
        self.para_est_sim = toPara(para_est_sim,
                                   t,
                                   ma_q)
        
        return self.para_est_sim    
    
    def ObjFuncAgg(self,
                   para_agg):
        data_moms_agg_dct = self.data_moms_agg_dct
        t = self.t
        ma_q = self.ma_q
        n_periods = self.n_periods
        ma_coeffs,sigmas = toPara(para_agg,
                                  t,
                                  ma_q)
        new_instance = cp.deepcopy(self)
        new_instance.t = t   
        new_instance.ma_coeffs = ma_coeffs
        new_instance.sigmas = sigmas
        model_series_sim = new_instance.SimulateSeries() 
        model_series_agg = new_instance.TimeAggregate(n_periods = n_periods)
        model_moms_agg_dct = new_instance.SimulateMomentsAgg()
        
        model_moms = np.array([model_moms_agg_dct[key] for key in ['Var']]).flatten()
        data_moms = np.array([data_moms_agg_dct[key] for key in ['Var']]).flatten()
        if len(model_moms) > len(data_moms):
            n_burn = len(model_moms) - len(data_moms)
            model_moms = model_moms[n_burn:]
        if len(model_moms) < len(data_moms):
            n_burn = -(len(model_moms) - len(data_moms))
            data_moms = data_moms[n_burn:]
        diff = np.linalg.norm(model_moms - data_moms)
        return diff
    
    def EstimateParaAgg(self,
                        method = 'CG',
                        bounds = None,
                        para_guess = None,
                        options = {'disp':True}):
        t = self.t
        ma_q = self.ma_q
        para_est_agg = minimize(self.ObjFuncAgg,
                                x0 = para_guess,
                                method = method,
                                bounds = bounds,
                                options = options)['x']
        
        self.para_est_agg = toPara(para_est_agg,
                                   t,
                                   ma_q)
        return self.para_est_agg  
    
##########################
#### vols 
#########################

    def ObjFuncAggVols(self,
                       para_vols_agg):
        
        self.rho,self.gamma,self.sigma_eps = para_vols_agg
        
        ## data agg vols
        data_vols_moms_agg_dct = self.data_vols_moms_agg_dct
        
        new_instance = cp.deepcopy(self)
        new_instance.SimulateSVols()
        new_instance.SimulateSVolsAgg()
        model_moms_agg_dct = new_instance.SimulateSVolsAggMoms()
        
        ## criteria 
        model_moms = np.array([model_moms_agg_dct[key] for key in ['Var']]).flatten()
        data_moms = np.array([data_moms_agg_dct[key] for key in ['Var']]).flatten()
        if len(model_moms) > len(data_moms):
            n_burn = len(model_moms) - len(data_moms)
            model_moms = model_moms[n_burn:]
        if len(model_moms) < len(data_moms):
            n_burn = -(len(model_moms) - len(data_moms))
            data_moms = data_moms[n_burn:]
        diff = np.linalg.norm(model_moms - data_moms)
        return diff
    
    def EstimateSVolsParaAgg(self,
                             method = 'Nelder-Mede',
                             bounds = None,
                             para_guess = None,
                             options = {'disp':True}):
        self.para_svols_est_agg = minimize(self.ObjFuncAggVols,
                                x0 = para_guess,
                                method = method,
                                bounds = bounds,
                                options = options)['x']

        return self.para_svols_est_agg 

    
##########
## new ###
########## 

    def ObjFuncAggCompute(self,
                          para_agg):
        data_moms_agg_dct = self.data_moms_agg_dct
        t = self.t
        ma_q = self.ma_q
        n_agg = self.n_agg
        ma_coeffs,sigmas = toPara(para_agg,
                                  t,
                                  ma_q)
        new_instance = cp.deepcopy(self)
        new_instance.t = t   
        new_instance.ma_coeffs = ma_coeffs
        new_instance.sigmas = sigmas
        #model_series_sim = new_instance.SimulateSeries() 
        #model_series_agg = new_instance.TimeAggregate(n_periods = n_periods)
        model_moms_agg_dct = new_instance.ComputeMomentsAgg(n_agg = self.n_agg)
        
        model_moms = np.array([model_moms_agg_dct[key] for key in ['Var']]).flatten()
        data_moms = np.array([data_moms_agg_dct[key] for key in ['Var']]).flatten()
        if len(model_moms) > len(data_moms):
            n_burn = len(model_moms) - len(data_moms)
            model_moms = model_moms[n_burn:]
        if len(model_moms) < len(data_moms):
            n_burn = -(len(model_moms) - len(data_moms))
            data_moms = data_moms[n_burn:]
        diff = np.linalg.norm(model_moms - data_moms)
        return diff
    
    def EstimateParaAggCompute(self,
                               method = 'CG',
                               bounds = None,
                               para_guess = None,
                               options = {'disp':True}):
        t = self.t
        ma_q = self.ma_q
        para_est_agg = minimize(self.ObjFuncAggCompute,
                                x0 = para_guess,
                                method = method,
                                bounds = bounds,
                                options = options)['x']
        
        self.para_est_agg_compute = toPara(para_est_agg,
                                           t,
                                           ma_q)
        return self.para_est_agg_compute  
    
    def Autocovar(self,
                  step = 1):
        cov_var = self.SimMoms['Var']
        if step >= 0:
            autovar = np.array([cov_var[i,i+step] for i in range(len(cov_var)-1)])
        if step < 0:
            autovar = np.array([cov_var[i+step,i] for i in range(abs(step),len(cov_var)-1)]) 
        self.autovar = autovar
        return self.autovar
    
    def AutocovarComp(self,
                  step = 1):
        cov_var = self.GenMoms['Var']
        if step >= 0:
            autovar = np.array([cov_var[i,i+step] for i in range(len(cov_var)-1)])
        if step < 0:
            autovar = np.array([cov_var[i+step,i] for i in range(abs(step),len(cov_var)-1)]) 
        self.autovarGen = autovar
        return self.autovarGen
    
    def AutocovarAgg(self,
                     step = 0):
        cov_var = self.SimAggMoms['Var']
        if step >=0:
            autovar = np.array([cov_var[i,i+step] for i in range(len(cov_var)-1)]) 
        if step < 0:
            autovar = np.array([cov_var[i,i+step] for i in range(abs(step),len(cov_var)-1)]) 
        self.autovar = autovar
        self.autovaragg = autovar

        return self.autovaragg 
    
    
#################
## other funcs
##################
    def hstep_sigma_theta(self,
                          sigma_theta_now
                          k):
        k_step_sigma_theta = self.rho**k*np.exp(-0.5*self.gamma)*sigma_theta_now**2 
        return k_step_sigma_theta
# + {"code_folding": [0]}
## debugging test of the data 

t = 10
ma_nosa = np.array([1])
p_sigmas = np.arange(t)  # sizes of the time-varying permanent volatility 
p_sigmas_rw = np.ones(t) # a special case of time-invariant permanent volatility, random walk 
p_sigmas_draw = np.random.uniform(0,1,t) ## allowing for time-variant shocks 

pt_ratio = 0.33
t_sigmas = pt_ratio * p_sigmas_draw # sizes of the time-varyingpermanent volatility
sigmas = np.array([p_sigmas_draw,
                   t_sigmas])

#dt = IMAProcess(t = t,
#         ma_coeffs = ma_nosa,
#         sigmas = sigmas)
#sim_data = dt.SimulateSeries(n_sim = 8000)
#sim_moms = dt.SimulatedMoments()

# + {"code_folding": [0]}
## invoke an instance 

dt_fake = IMAProcess(t = t,
                     ma_coeffs = ma_nosa,
                     sigmas = sigmas)
data_fake= dt_fake.SimulateSeries(n_sim = 5000)
moms_fake = dt_fake.SimulatedMoments()


# + {"code_folding": [0]}
## time aggregation 
n_agg = 3
dt_fake.TimeAggregate(n_periods = n_agg)
moms_fake_agg = dt_fake.SimulateMomentsAgg()

## and prepare fake data 

sigmas2 = sigmas*2 
dt_fake2 = IMAProcess(t = t,
                      ma_coeffs = ma_nosa,
                      sigmas = sigmas2)
data_fake2= dt_fake2.SimulateSeries(n_sim = 5000)
moms_fake2 = dt_fake2.SimulatedMoments()
dt_fake2.TimeAggregate(n_periods = n_agg)
moms_fake_agg2 = dt_fake2.SimulateMomentsAgg()

# + {"code_folding": []}
# simulated time aggregated moments 
agg_moms_sim = moms_fake_agg['Var']

# computed time aggregated moments 
agg_moms_com = dt_fake.ComputeMomentsAgg(n_agg = n_agg)

distance = np.linalg.norm((agg_moms_com[n_agg:,n_agg:] - agg_moms_sim))

# + {"code_folding": []}
## estimation 
#dt_fake.n_agg = 3
#dt_fake.GetDataMomentsAgg(moms_fake_agg2)
#dt_fake.EstimateParaAggCompute()
