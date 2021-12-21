# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 14:23:36 2021

@author: wdu

python 3.8.8
"""
import time
import numpy as np
from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType,ConsMarkovSolver
from HARK.utilities import plot_funcs, make_grid_exp_mult
from HARK.distribution import DiscreteDistribution, MeanOneLogNormal,combine_indep_dstns
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy import sparse as sp




#parameters 
job_sep = .2
job_find = .8
unemp_insurance = 0.05
states = 2


markov_array = np.array( [ [1 - job_sep*(1-job_find) ,  job_find ] ,  #" The sum of entries in each column in t should equal one. "
                          
                               [ job_sep*(1-job_find) , 1- job_find] ]          )

print(markov_array)


eigen, ss_dstn = sp.linalg.eigs(markov_array , k=1, which='LM')
print(ss_dstn)
print(eigen.real)
print(eigen[0])
print(ss_dstn[:,0])
ss_dstn = ss_dstn[:,0] / np.sum(ss_dstn[:,0])

print(ss_dstn)

print(np.dot(markov_array,ss_dstn))



HANK_SAM_Dict={
    # Parameters shared with the perfect foresight model
    "CRRA":2,                           # Coefficient of relative risk aversion
    "Rfree": np.array([1.03**.25]* states),                 # Interest factor on assets
    "DiscFac": 0.9735, #.96,            # Intertemporal discount factor
    "LivPrb" : [np.array([.99375]*states)],                # Survival probability
    "PermGroFac" : [np.array([1.00]*states)],               # Permanent income growth factor

    # Parameters that specify the income distribution over the lifecycle
   
    "PermShkStd" : [.06],                  # Standard deviation of log permanent shocks to income
    "PermShkCount" : 7,                    # Number of points in discrete approximation to permanent income shocks
    "TranShkStd" : [.3],                   # Standard deviation of log transitory shocks to income
    "TranShkCount" : 7,                    # Number of points in discrete approximation to transitory income shocks
    "UnempPrb" : 0.05, #.08                # Probability of unemployment while working
    "IncUnemp" :  .3,                      # Unemployment benefits replacement rate
    "UnempPrbRet" : 0.0005,                # Probability of "unemployment" while retired
    "IncUnempRet" : 0.0,                   # "Unemployment" benefits when retired
    "T_retire" : 0,                        # Period of retirement (0 --> no retirement)
    "tax_rate" : .2,                       # Flat income tax rate (legacy parameter, will be removed in future)

    # Parameters for constructing the "assets above minimum" grid
    "aXtraMin" : 0.001,                    # Minimum end-of-period "assets above minimum" value
    "aXtraMax" : 35,                       # Maximum end-of-period "assets above minimum" value
    "aXtraCount" : 48,                     # Number of points in the base grid of "assets above minimum"
    "aXtraNestFac" : 3,                    # Exponential nesting factor when constructing "assets above minimum" grid
    "aXtraExtra" : [None],                 # Additional values to add to aXtraGrid

    # A few other parameters
    "BoroCnstArt" : 0.0,                   # Artificial borrowing constraint; imposed minimum level of end-of period assets
    "vFuncBool" : True,                    # Whether to calculate the value function during solution
    "CubicBool" : False,                   # Preference shocks currently only compatible with linear cFunc
    "T_cycle" : 1,                         # Number of periods in the cycle for this agent type

    # Parameters only used in simulation
    "AgentCount" : 100000,                 # Number of agents of this type
    "T_sim" : 200,                         # Number of periods to simulate
    "aNrmInitMean" : np.log(.8)-(.5**2)/2,# Mean of log initial assets
    "aNrmInitStd"  : .5,                   # Standard deviation of log initial assets
    "pLvlInitMean" : 0.0,                  # Mean of log initial permanent income
    "pLvlInitStd"  : 0.0,                  # Standard deviation of log initial permanent income
    "PermGroFacAgg" : 1.0,                 # Aggregate permanent income growth factor
    "T_age" : None,                        # Age after which simulated agents are automatically killed
    
    # new parameters
     "dx"         : 0,                     # Deviation from steady state
     "jac"        : False,
     "jacW"       : False, 
     "job_sep"    : job_sep, 
     "job_find"   : job_find, 
    
     
    # markov array
    
    "MrkvArray"   :  [np.array( [ [1 - job_sep*(1-job_find) , job_find ] ,  
                          
                               [ job_sep*(1-job_find) , 1- job_find] ]          ).T]
     }


class HANK_SAM_agent(MarkovConsumerType):
    
        
    time_inv_ = MarkovConsumerType.time_inv_  + [

                                                   "wage",                                                  
                                                   "dx",
                                                   "jac",
                                                   "jacW",
                                         
                                            
    
                                                    
                                                  ]

    #def __init__(self,cycles=0,time_flow=True,**kwds):
    def __init__(self,cycles=0,**kwds):

        '''
        Just calls on MarkovConsumerType
        
        Parameters
        ----------
        cycles : int
            Number of times the sequence of periods should be solved.
        time_flow : boolean
            Whether time is currently "flowing" forward for this instance.
        
        Returns
        -------
        None
        '''       
        
        MarkovConsumerType.__init__(self, cycles = 0, **kwds)

        


     
    def define_distribution_grid(self, dist_mGrid=None, dist_pGrid=None, m_density = 0, num_pointsM = 48,  num_pointsP = 50, max_p_fac = 20.0):
        
        '''
        Defines the grid on which the distribution is defined. Stores the grid of market resources and permanent income as attributes of self.
        Grid for normalized market resources and permanent income may be prespecified 
        as dist_mGrid and dist_pGrid, respectively. If not then default grid is computed based off given parameters.
        
        Parameters
        ----------
        dist_mGrid : np.array
                Prespecified grid for distribution over normalized market resources
            
        dist_pGrid : np.array
                Prespecified grid for distribution over permanent income. 
            
        m_density: float
                Density of normalized market resources grid. Default value is mdensity = 0.
                Only affects grid of market resources if dist_mGrid=None.
            
        num_pointsM: float
                Number of gridpoints for market resources grid.
        
        num_pointsP: float
                 Number of gridpoints for permanent income. 
                 This grid will be exponentiated by the function make_grid_exp_mult.
                
        max_p_fac : float
                Factor that scales the maximum value of permanent income grid. 
                Larger values increases the maximum value of permanent income grid.
        
        Returns
        -------
        None
        '''  
 
        if self.cycles == 0:
            if dist_mGrid == None:    
                aXtra_Grid = make_grid_exp_mult(
                        ming=self.aXtraMin, maxg=self.aXtraMax, ng = num_pointsM, timestonest = 3) #Generate Market resources grid given density and number of points
                
                for i in range(m_density):
                    axtra_shifted = np.delete(aXtra_Grid,-1) 
                    axtra_shifted = np.insert(axtra_shifted, 0,1.00000000e-04)
                    dist_betw_pts = aXtra_Grid - axtra_shifted
                    dist_betw_pts_half = dist_betw_pts/2
                    new_A_grid = axtra_shifted + dist_betw_pts_half
                    aXtra_Grid = np.concatenate((aXtra_Grid,new_A_grid))
                    aXtra_Grid = np.sort(aXtra_Grid)
                    
                self.dist_mGrid =  aXtra_Grid
            else:
                self.dist_mGrid = dist_mGrid #If grid of market resources prespecified then use as mgrid
                
            if dist_pGrid == None:
                num_points = 50 #Number of permanent income gridpoints
                #Dist_pGrid is taken to cover most of the ergodic distribution
                p_variance = self.PermShkStd[0]**2 #set variance of permanent income shocks
                max_p = max_p_fac*(p_variance/(1-self.LivPrb[0]))**0.5 #Maximum Permanent income value
                one_sided_grid = make_grid_exp_mult(1.0+1e-3, np.exp(max_p), num_points, 1)
                self.dist_pGrid = np.append(np.append(1.0/np.fliplr([one_sided_grid])[0],np.ones(1)),one_sided_grid) #Compute permanent income grid
            else:
                self.dist_pGrid = dist_pGrid #If grid of permanent income prespecified then use it as pgrid
                
        elif self.cycles > 1:
            print('define_distribution_grid requires cycles = 0 or cycles = 1')
        
        elif self.T_cycle != 0:
            
            if dist_mGrid == None:
                aXtra_Grid = make_grid_exp_mult(
                        ming=self.aXtraMin, maxg=self.aXtraMax, ng = num_pointsM, timestonest = 3) #Generate Market resources grid given density and number of points
                
                for i in range(m_density):
                    axtra_shifted = np.delete(aXtra_Grid,-1) 
                    axtra_shifted = np.insert(axtra_shifted, 0,1.00000000e-04)
                    dist_betw_pts = aXtra_Grid - axtra_shifted
                    dist_betw_pts_half = dist_betw_pts/2
                    new_A_grid = axtra_shifted + dist_betw_pts_half
                    aXtra_Grid = np.concatenate((aXtra_Grid,new_A_grid))
                    aXtra_Grid = np.sort(aXtra_Grid)
                    
                self.dist_mGrid =  aXtra_Grid
                
            else:
                self.dist_mGrid = dist_mGrid #If grid of market resources prespecified then use as mgrid
                    
            if dist_pGrid == None:
                
                self.dist_pGrid = [] #list of grids of permanent income    
                
                for i in range(self.T_cycle):
                    
                    num_points = 50
                    #Dist_pGrid is taken to cover most of the ergodic distribution
                    p_variance = self.PermShkStd[i]**2 # set variance of permanent income shocks this period
                    max_p = 20.0*(p_variance/(1-self.LivPrb[i]))**0.5 # Consider probability of staying alive this period
                    one_sided_grid = make_grid_exp_mult(1.0+1e-3, np.exp(max_p), num_points, 2) 
                    
                    dist_pGrid = np.append(np.append(1.0/np.fliplr([one_sided_grid])[0],np.ones(1)),one_sided_grid) # Compute permanent income grid this period. Grid of permanent income may differ dependent on PermShkStd
                    self.dist_pGrid.append(dist_pGrid)

            else:
                self.dist_pGrid = dist_pGrid #If grid of permanent income prespecified then use as pgrid
                
                
    def calc_transition_matrix(self, shk_dstn = None):
        '''
        Calculates how the distribution of agents across market resources 
        transitions from one period to the next. If finite horizon problem, then calculates
        a list of transition matrices, consumption and asset policy grids for each period of the problem. 
        The transition matrix/matrices and consumption and asset policy grid(s) are stored as attributes of self.
        
        
        Parameters
        ----------
            shk_dstn: list 
                list of income shock distributions

        Returns
        -------
        None
        
        ''' 
        
        
        self.state_num = len(self.MrkvArray[0])
        
        

        

        if self.cycles == 0: 
            
            
            #parameters 
            job_sep = self.job_sep
            job_find = self.job_find # this value needs to be analytically computed from steady state
            unemp_insurance = self.unemp_insurance
            
            
            markov_array = self.MrkvArray
            
            eigen, ss_dstn = sp.linalg.eigs(markov_array[0] , k=1, which='LM')
         
            for i in range(len(eigen)):
                if eigen[i] == 1:
                    
                    ss_dstn = ss_dstn[:,i] / np.sum(ss_dstn[:,i])
                
            self.Rfree = self.Rfree[0]
            
           
            if shk_dstn == None:
                shk_dstn = self.IncShkDstn[0]
            
            dist_mGrid = self.dist_mGrid #Grid of market resources
            dist_pGrid = self.dist_pGrid #Grid of permanent incomes
            
            aNext_e = dist_mGrid - self.solution[0].cFunc[0](dist_mGrid)  #assets next period
            aNext_u = dist_mGrid - self.solution[0].cFunc[1](dist_mGrid)
            
            self.aPol_Grid_e = aNext_e # Steady State Asset Policy Grid
            self.aPol_Grid_u = aNext_u 
            
            self.cPol_Grid_e = self.solution[0].cFunc[0](dist_mGrid) #Steady State Consumption Policy Grid
            self.cPol_Grid_u = self.solution[0].cFunc[1](dist_mGrid)
            
            # Obtain shock values and shock probabilities from income distribution
            bNext_e = self.Rfree*aNext_e # Bank Balances next period (Interest rate * assets)
            bNext_u = self.Rfree*aNext_u # Bank Balances next period (Interest rate * assets)

            shk_prbs = shk_dstn[0].pmf  # Probability of shocks 
            tran_shks = shk_dstn[0].X[1] # Transitory shocks
            perm_shks = shk_dstn[0].X[0] # Permanent shocks
            LivPrb = self.LivPrb[0][0] # Update probability of staying alive
            
            
            if len(dist_pGrid) == 1: 
        
                #New borns have this distribution (assumes start with no assets and permanent income=1)
                NewBornDist = self.jump_to_grid_fast(tran_shks,shk_prbs,dist_mGrid)
                
                
                # Generate Transition Matrix
                TranMatrix_ee = np.zeros((len(dist_mGrid),len(dist_mGrid))) 
                for i in range(len(dist_mGrid)):
                    mNext_ij = bNext_e[i]/perm_shks + tran_shks # Compute next period's market resources given todays bank balances bnext[i]
                    TranMatrix_ee[:,i] = LivPrb*self.jump_to_grid_fast(mNext_ij, shk_prbs,dist_mGrid) + (1.0-LivPrb)*NewBornDist # this is the transition matrix if given you are employed today and employed tomorrow so you assume the employed consumption policy
                self.tran_matrix_ee = TranMatrix_ee
                
                # Generate Transition Matrix
                TranMatrix_eu = np.zeros((len(dist_mGrid),len(dist_mGrid))) 
                for i in range(len(dist_mGrid)):
                    mNext_ij = bNext_e[i]/perm_shks + unemp_insurance # Compute next period's market resources given todays bank balances bnext[i]
                    TranMatrix_eu[:,i] = LivPrb*self.jump_to_grid_fast(mNext_ij, shk_prbs,dist_mGrid) + (1.0-LivPrb)*NewBornDist # this is the transition matrix if given you are employed today and employed tomorrow so you assume the employed consumption policy
                self.tran_matrix_eu = TranMatrix_eu
                
                # Generate Transition Matrix
                TranMatrix_uu = np.zeros((len(dist_mGrid),len(dist_mGrid))) 
                for i in range(len(dist_mGrid)):
                    mNext_ij = bNext_u[i]/perm_shks + unemp_insurance  
                    # Compute next period's market resources given todays bank balances bnext[i]
                    TranMatrix_uu[:,i] = LivPrb*self.jump_to_grid_fast(mNext_ij, shk_prbs,dist_mGrid) + (1.0-LivPrb)*NewBornDist # this is the transition matrix if given you are unemployed today and unemployed tomorrow so you assume the unemployed consumption policy
                self.tran_matrix_uu = TranMatrix_uu
                
                # Generate Transition Matrix
                TranMatrix_ue = np.zeros((len(dist_mGrid),len(dist_mGrid))) 
                for i in range(len(dist_mGrid)):
                    mNext_ij = bNext_u[i]/perm_shks + tran_shks  
                    # Compute next period's market resources given todays bank balances bnext[i]
                    TranMatrix_ue[:,i] = LivPrb*self.jump_to_grid_fast(mNext_ij, shk_prbs,dist_mGrid) + (1.0-LivPrb)*NewBornDist # this is the transition matrix if given you are unemployed today and unemployed tomorrow so you assume the unemployed consumption policy
                self.tran_matrix_ue = TranMatrix_ue
                
                self.tran_matrix_u = job_find * TranMatrix_ue  +  (1 - job_find ) * TranMatrix_uu #This is the transition for someone who's state today is unemployed
                self.tran_matrix_e = ( 1 - job_sep*(1-job_find))*TranMatrix_ee  +  (job_sep*(1-job_find)) * TranMatrix_eu # This is the transition for someone who's state is employed today
                
                self.prb_emp = ss_dstn[0] 
                self.prb_unemp = 1 -self.prb_emp 
                self.tran_matrix =  self.prb_unemp * self.tran_matrix_u + self.prb_emp * self.tran_matrix_e # This is the transition matrix for the whole economy
        
            else:
                #New borns have this distribution (assumes start with no assets and permanent income=1)
                NewBornDist = self.jump_to_grid(tran_shks,np.ones_like(tran_shks),shk_prbs,dist_mGrid,dist_pGrid)
                
                # Generate Transition Matrix
                TranMatrix_ee = np.zeros((len(dist_mGrid)*len(dist_pGrid),len(dist_mGrid)*len(dist_pGrid)))
                for i in range(len(dist_mGrid)):
                    for j in range(len(dist_pGrid)):
                        mNext_ij = bNext_e[i]/perm_shks + tran_shks # Compute next period's market resources given todays bank balances bnext[i]
                        pNext_ij = dist_pGrid[j]*perm_shks # Computes next period's permanent income level by applying permanent income shock
                        TranMatrix_ee[:,i*len(dist_pGrid)+j] = LivPrb*self.jump_to_grid(mNext_ij, pNext_ij, shk_prbs,dist_mGrid,dist_pGrid) + (1.0-LivPrb)*NewBornDist # this is the transition matrix if given you are employed today and employed tomorrow so you assume the employed consumption policy
                self.tran_matrix_ee = TranMatrix_ee
                
                # Generate Transition Matrix
                TranMatrix_eu = np.zeros((len(dist_mGrid)*len(dist_pGrid),len(dist_mGrid)*len(dist_pGrid)))
                for i in range(len(dist_mGrid)):
                    for j in range(len(dist_pGrid)):
                        mNext_ij = bNext_e[i]/perm_shks + unemp_insurance # Compute next period's market resources given todays bank balances bnext[i]
                        pNext_ij = dist_pGrid[j]*perm_shks # Computes next period's permanent income level by applying permanent income shock
                        TranMatrix_eu[:,i*len(dist_pGrid)+j] = LivPrb*self.jump_to_grid(mNext_ij, pNext_ij, shk_prbs,dist_mGrid,dist_pGrid) + (1.0-LivPrb)*NewBornDist # this is the transition matrix if given you are employed today and employed tomorrow so you assume the employed consumption policy
                self.tran_matrix_eu = TranMatrix_eu
                
                # Generate Transition Matrix
                TranMatrix_uu = np.zeros((len(dist_mGrid)*len(dist_pGrid),len(dist_mGrid)*len(dist_pGrid)))
                for i in range(len(dist_mGrid)):
                    for j in range(len(dist_pGrid)):
                        mNext_ij = bNext_u[i]/perm_shks + unemp_insurance  
                        # Compute next period's market resources given todays bank balances bnext[i]
                        pNext_ij = dist_pGrid[j]*perm_shks # Computes next period's permanent income level by applying permanent income shock
                        TranMatrix_uu[:,i*len(dist_pGrid)+j] = LivPrb*self.jump_to_grid(mNext_ij, pNext_ij, shk_prbs,dist_mGrid,dist_pGrid) + (1.0-LivPrb)*NewBornDist # this is the transition matrix if given you are unemployed today and unemployed tomorrow so you assume the unemployed consumption policy
                self.tran_matrix_uu = TranMatrix_uu
                
                # Generate Transition Matrix
                TranMatrix_ue = np.zeros((len(dist_mGrid)*len(dist_pGrid),len(dist_mGrid)*len(dist_pGrid)))
                for i in range(len(dist_mGrid)):
                    for j in range(len(dist_pGrid)):
                        mNext_ij = bNext_u[i]/perm_shks + tran_shks  
                        # Compute next period's market resources given todays bank balances bnext[i]
                        pNext_ij = dist_pGrid[j]*perm_shks # Computes next period's permanent income level by applying permanent income shock
                        TranMatrix_ue[:,i*len(dist_pGrid)+j] = LivPrb*self.jump_to_grid(mNext_ij, pNext_ij, shk_prbs,dist_mGrid,dist_pGrid) + (1.0-LivPrb)*NewBornDist # this is the transition matrix if given you are unemployed today and unemployed tomorrow so you assume the unemployed consumption policy
                self.tran_matrix_ue = TranMatrix_ue
                
                
                self.tran_matrix_u = job_find * TranMatrix_ue  +  (1 - job_find ) * TranMatrix_uu #This is the transition for someone who's state today is unemployed
                self.tran_matrix_e = ( 1 - job_sep*(1-job_find))*TranMatrix_ee  +  (job_sep*(1-job_find)) * TranMatrix_eu # This is the transition for someone who's state is employed today
                
                self.prb_emp = ss_dstn[0] 
                self.prb_unemp = 1- self.prb_emp 
                self.tran_matrix =  self.prb_unemp*self.tran_matrix_u  +  self.prb_emp*self.tran_matrix_e # This is the transition matrix for the whole economy





        elif self.cycles > 1:
            print('calc_transition_matrix requires cycles = 0 or cycles = 1')
            
        elif self.T_cycle!= 0:
            
        
            
        
            
            # for finite horizon, we can account for changing levels of prb_unemp because of endogenous job finding probability by imposing a list of these values, so for q'th period, the probability is slightly different
            
            if shk_dstn == None:
                shk_dstn = self.IncShkDstn
            
            self.cPol_Grid_e = [] # List of consumption policy grids for each period in T_cycle
            self.cPol_Grid_u = [] # List of consumption policy grids for each period in T_cycle

            self.aPol_Grid_e = [] # List of asset policy grids for each period in T_cycle
            self.aPol_Grid_u = [] # List of asset policy grids for each period in T_cycle

            self.tran_matrix_e = [] # List of transition matrices
            self.tran_matrix_u = [] # List of transition matrices
            self.tran_matrix = [] # List of transition matrices
            
            dist_mGrid =  self.dist_mGrid
            
            self.prb_emp =[]
            self.prb_unemp = []
            
            dstn_0 = self.dstn_0 
            
            for k in range(self.T_cycle):
                
                job_sep = self.job_sep[k]
                job_find = self.job_find[k]
                
                unemp_insurance = self.unemp_insurance[k]
                
                
            
                                
                dstn_0 = np.dot(self.MrkvArray[k], dstn_0)
                
                if type(self.dist_pGrid) == list:
                    dist_pGrid = self.dist_pGrid[k] #Permanent income grid this period
                else:
                    dist_pGrid = self.dist_pGrid #If here then use prespecified permanent income grid
                
                Cnow_e = self.solution[k].cFunc[0](dist_mGrid)  #Consumption policy grid in period k
                Cnow_u = self.solution[k].cFunc[1](dist_mGrid) 
         
                
                self.cPol_Grid_e.append(Cnow_e)  # List of consumption policy grids for each period in T_cycle
                self.cPol_Grid_u.append(Cnow_u)  # List of consumption policy grids for each period in T_cycle
                

                aNext_e = dist_mGrid - Cnow_e # Asset policy grid in period k
                aNext_u = dist_mGrid - Cnow_u # Asset policy grid in period k

                self.aPol_Grid_e.append(aNext_e) # Add to list
                self.aPol_Grid_u.append(aNext_u) # Add to list
                
                
                if type(self.Rfree)==list:
                    bNext_e = self.Rfree[k][0]*aNext_e # we chose the index zero because it both agents face the same interest rates
                else:
                    bNext_e = self.Rfree*aNext_e
                    
                if type(self.Rfree)==list:
                    bNext_u = self.Rfree[k][0]*aNext_u
                else:
                    bNext_u = self.Rfree*aNext_u
                    
                    
                    
                #Obtain shocks and shock probabilities from income distribution this period
                shk_prbs = shk_dstn[k][0].pmf  #Probability of shocks this period , I choose the index zero, because we really only use the employe dshock distribution, the unemployed is already implemented automatically
                tran_shks = shk_dstn[k][0].X[1] #Transitory shocks this period
                perm_shks = shk_dstn[k][0].X[0] #Permanent shocks this period
                LivPrb = self.LivPrb[k][0] # Update probability of staying alive this period
                
                
                if len(dist_pGrid) == 1: 
            
                    #New borns have this distribution (assumes start with no assets and permanent income=1)
                    NewBornDist = self.jump_to_grid_fast(tran_shks,shk_prbs,dist_mGrid)
                    
                    
                    
                    # Generate Transition Matrix
                    TranMatrix_ee = np.zeros((len(dist_mGrid),len(dist_mGrid))) 
                    for i in range(len(dist_mGrid)):
                        mNext_ij = bNext_e[i]/perm_shks + tran_shks # Compute next period's market resources given todays bank balances bnext[i]
                        TranMatrix_ee[:,i] = LivPrb*self.jump_to_grid_fast(mNext_ij, shk_prbs,dist_mGrid) + (1.0-LivPrb)*NewBornDist # this is the transition matrix if given you are employed today and employed tomorrow so you assume the employed consumption policy
                    self.tran_matrix_ee = TranMatrix_ee
                    
                    # Generate Transition Matrix
                    TranMatrix_eu = np.zeros((len(dist_mGrid),len(dist_mGrid))) 
                    for i in range(len(dist_mGrid)):
                        mNext_ij = bNext_e[i]/perm_shks + unemp_insurance # Compute next period's market resources given todays bank balances bnext[i]
                        TranMatrix_eu[:,i] = LivPrb*self.jump_to_grid_fast(mNext_ij, shk_prbs,dist_mGrid) + (1.0-LivPrb)*NewBornDist # this is the transition matrix if given you are employed today and employed tomorrow so you assume the employed consumption policy
                    self.tran_matrix_eu = TranMatrix_eu
                    
                    # Generate Transition Matrix
                    TranMatrix_uu = np.zeros((len(dist_mGrid),len(dist_mGrid))) 
                    for i in range(len(dist_mGrid)):
                        mNext_ij = bNext_u[i]/perm_shks + unemp_insurance  
                        # Compute next period's market resources given todays bank balances bnext[i]
                        TranMatrix_uu[:,i] = LivPrb*self.jump_to_grid_fast(mNext_ij, shk_prbs,dist_mGrid) + (1.0-LivPrb)*NewBornDist # this is the transition matrix if given you are unemployed today and unemployed tomorrow so you assume the unemployed consumption policy
                    self.tran_matrix_uu = TranMatrix_uu
                    
                    # Generate Transition Matrix
                    TranMatrix_ue = np.zeros((len(dist_mGrid),len(dist_mGrid))) 
                    for i in range(len(dist_mGrid)):
                        mNext_ij = bNext_u[i]/perm_shks + tran_shks  
                        # Compute next period's market resources given todays bank balances bnext[i]
                        TranMatrix_ue[:,i] = LivPrb*self.jump_to_grid_fast(mNext_ij, shk_prbs,dist_mGrid) + (1.0-LivPrb)*NewBornDist # this is the transition matrix if given you are unemployed today and unemployed tomorrow so you assume the unemployed consumption policy
                    self.tran_matrix_ue = TranMatrix_ue
                    
                    
                    tran_matrix_u = job_find * TranMatrix_ue  +  (1 - job_find ) * TranMatrix_uu #This is the transition for someone who's state today is unemployed
                    tran_matrix_e = ( 1 - job_sep*(1-job_find))*TranMatrix_ee  +  (job_sep*(1-job_find)) * TranMatrix_eu # This is the transition for someone who's state is employed today
                    
                    self.tran_matrix_u.append( tran_matrix_u ) #This is the transition for someone who's state today is unemployed
                    self.tran_matrix_e.append( tran_matrix_e )
                    
                    prb_emp = dstn_0[0] 
                    prb_unemp = 1 - prb_emp 
                    
                    self.prb_emp.append(prb_emp)
                    self.prb_unemp.append(prb_unemp)
                    
                    tran_matrix_combined =  prb_unemp * tran_matrix_u + prb_emp * tran_matrix_e # This is the transition matrix for the whole economy
                    self.tran_matrix.append(tran_matrix_combined)

                    
                
                    # idea: calculate the ergodic distribution of tran_matrix_u and tran_matrix_e, will need to track these dstns. then to compute aggregate consumption need to do linear combination of these distributions with cPolGrids
            
                    
                    
                else:
                    
                    NewBornDist = self.jump_to_grid(tran_shks,np.ones_like(tran_shks),shk_prbs,dist_mGrid,dist_pGrid)

                    # Generate Transition Matrix this period
                    TranMatrix = np.zeros((len(dist_mGrid)*len(dist_pGrid),len(dist_mGrid)*len(dist_pGrid))) 
                    for i in range(len(dist_mGrid)):
                        for j in range(len(dist_pGrid)):
                            mNext_ij = bNext[i]/perm_shks + tran_shks # Compute next period's market resources given todays bank balances bnext[i]
                            pNext_ij = dist_pGrid[j]*perm_shks # Computes next period's permanent income level by applying permanent income shock
                            TranMatrix[:,i*len(dist_pGrid)+j] = LivPrb*self.jump_to_grid(mNext_ij, pNext_ij, shk_prbs, dist_mGrid, dist_pGrid) + (1.0-LivPrb)*NewBornDist #generate transition probabilities
                    TranMatrix = TranMatrix #columns represent the current state while rows represent the next state
                    #the 4th row , 6th column entry represents the probability of transitioning from the 6th element of the combined perm and m grid (grid of market resources multiplied by grid of perm income) to the 4th element of the combined perm and m grid
                    self.tran_matrix.append(TranMatrix)         
                

                
    def jump_to_grid(self, m_vals, perm_vals, probs, dist_mGrid, dist_pGrid ):
        
        '''
        Distributes values onto a predefined grid, maintaining the means. m_vals and perm_vals are realizations of market resources and permanent income while 
        dist_mGrid and dist_pGrid are the predefined grids of market resources and permanent income, respectively. That is, m_vals and perm_vals do not necesarily lie on their 
        respective grids. Returns probabilities of each gridpoint on the combined grid of market resources and permanent income.
        
        
        Parameters
        ----------
        m_vals: np.array
                Market resource values 
        
        perm_vals: np.array
                Permanent income values 
        
        probs: np.array
                Shock probabilities associated with combinations of m_vals and perm_vals. 
                Can be thought of as the probability mass function  of (m_vals, perm_vals).
        
        dist_mGrid : np.array
                Grid over normalized market resources
            
        dist_pGrid : np.array
                Grid over permanent income 

        Returns
        -------
        probGrid.flatten(): np.array
                 Probabilities of each gridpoint on the combined grid of market resources and permanent income
        '''
    
        probGrid = np.zeros((len(dist_mGrid),len(dist_pGrid)))
        mIndex = np.digitize(m_vals,dist_mGrid) - 1 # Array indicating in which bin each values of m_vals lies in relative to dist_mGrid. Bins lie between between point of Dist_mGrid. 
        #For instance, if mval lies between dist_mGrid[4] and dist_mGrid[5] it is in bin 4 (would be 5 if 1 was not subtracted in the previous line). 
        mIndex[m_vals <= dist_mGrid[0]] = -1 # if the value is less than the smallest value on dist_mGrid assign it an index of -1
        mIndex[m_vals >= dist_mGrid[-1]] = len(dist_mGrid)-1 # if value if greater than largest value on dist_mGrid assign it an index of the length of the grid minus 1
        
        #the following three lines hold the same intuition as above
        pIndex = np.digitize(perm_vals,dist_pGrid) - 1
        pIndex[perm_vals <= dist_pGrid[0]] = -1
        pIndex[perm_vals >= dist_pGrid[-1]] = len(dist_pGrid)-1
        
        for i in range(len(m_vals)):
            if mIndex[i]==-1: # if mval is below smallest gridpoint, then assign it a weight of 1.0 for lower weight. 
                mlowerIndex = 0
                mupperIndex = 0
                mlowerWeight = 1.0
                mupperWeight = 0.0
            elif mIndex[i]==len(dist_mGrid)-1: # if mval is greater than maximum gridpoint, then assign the following weights
                mlowerIndex = -1
                mupperIndex = -1
                mlowerWeight = 1.0
                mupperWeight = 0.0
            else: # Standard case where mval does not lie past any extremes
            #identify which two points on the grid the mval is inbetween
                mlowerIndex = mIndex[i] 
                mupperIndex = mIndex[i]+1
            #Assign weight to the indices that bound the m_vals point. Intuitively, an mval perfectly between two points on the mgrid will assign a weight of .5 to the gridpoint above and below
                mlowerWeight = (dist_mGrid[mupperIndex]-m_vals[i])/(dist_mGrid[mupperIndex]-dist_mGrid[mlowerIndex]) #Metric to determine weight of gridpoint/index below. Intuitively, mvals that are close to gridpoint/index above are assigned a smaller mlowerweight.
                mupperWeight = 1.0 - mlowerWeight # weight of gridpoint/ index above
                
            #Same logic as above except the weights here concern the permanent income grid
            if pIndex[i]==-1: 
                plowerIndex = 0
                pupperIndex = 0
                plowerWeight = 1.0
                pupperWeight = 0.0
            elif pIndex[i]==len(dist_pGrid)-1:
                plowerIndex = -1
                pupperIndex = -1
                plowerWeight = 1.0
                pupperWeight = 0.0
            else:
                plowerIndex = pIndex[i]
                pupperIndex = pIndex[i]+1
                plowerWeight = (dist_pGrid[pupperIndex]-perm_vals[i])/(dist_pGrid[pupperIndex]-dist_pGrid[plowerIndex])
                pupperWeight = 1.0 - plowerWeight
                
            # Compute probabilities of each gridpoint on the combined market resources and permanent income grid by looping through each point on the combined market resources and permanent income grid, 
            # assigning probabilities to each gridpoint based off the probabilities of the surrounding mvals and pvals and their respective weights placed on the gridpoint.
            # Note* probs[i] is the probability of mval AND pval occurring
            probGrid[mlowerIndex][plowerIndex] = probGrid[mlowerIndex][plowerIndex] + probs[i]*mlowerWeight*plowerWeight # probability of gridpoint below mval and pval 
            probGrid[mlowerIndex][pupperIndex] = probGrid[mlowerIndex][pupperIndex] + probs[i]*mlowerWeight*pupperWeight # probability of gridpoint below mval and above pval
            probGrid[mupperIndex][plowerIndex] = probGrid[mupperIndex][plowerIndex] + probs[i]*mupperWeight*plowerWeight # probability of gridpoint above mval and below pval
            probGrid[mupperIndex][pupperIndex] = probGrid[mupperIndex][pupperIndex] + probs[i]*mupperWeight*pupperWeight # probability of gridpoint above mval and above pval
            

        return probGrid.flatten()

    def jump_to_grid_fast(self, vals, probs ,Grid ):
        '''
        Distributes values onto a predefined grid, maintaining the means.
        ''' 
    
        probGrid = np.zeros(len(Grid))
        mIndex = np.digitize(vals,Grid) - 1
        mIndex[vals <= Grid[0]] = -1
        mIndex[vals >= Grid[-1]] = len(Grid)-1
        
 
        for i in range(len(vals)):
            if mIndex[i]==-1:
                mlowerIndex = 0
                mupperIndex = 0
                mlowerWeight = 1.0
                mupperWeight = 0.0
            elif mIndex[i]==len(Grid)-1:
                mlowerIndex = -1
                mupperIndex = -1
                mlowerWeight = 1.0
                mupperWeight = 0.0
            else:
                mlowerIndex = mIndex[i]
                mupperIndex = mIndex[i]+1
                mlowerWeight = (Grid[mupperIndex]-vals[i])/(Grid[mupperIndex] - Grid[mlowerIndex])
                mupperWeight = 1.0 - mlowerWeight
                
            probGrid[mlowerIndex] = probGrid[mlowerIndex] + probs[i]*mlowerWeight
            probGrid[mupperIndex] = probGrid[mupperIndex] + probs[i]*mupperWeight
            
        return probGrid.flatten()
    
        
    def calc_ergodic_dist(self, transition_matrix = None):
        
        '''
        Calculates the ergodic distribution across normalized market resources and
        permanent income as the eigenvector associated with the eigenvalue 1.
        The distribution is stored as attributes of self both as a vector and as a reshaped array with the ij'th element representing
        the probability of being at the i'th point on the mGrid and the j'th
        point on the pGrid.
        
        Parameters
        ----------
        transition_matrix: List 
                    transition matrix whose ergordic distribution is to be solved

        Returns
        -------
        None
        '''
        
        if transition_matrix == None:
            transition_matrix = [self.tran_matrix]
        
        
        eigen, ergodic_distr = sp.linalg.eigs(transition_matrix[0] , k=1 , which='LM')  # Solve for ergodic distribution
        ergodic_distr = ergodic_distr.real/np.sum(ergodic_distr.real)
        
        self.vec_erg_dstn = ergodic_distr #distribution as a vector
        self.erg_dstn = ergodic_distr.reshape((len(self.dist_mGrid),len(self.dist_pGrid))) # distribution reshaped into len(mgrid) by len(pgrid) array
        
    
    
    
    
    
#----------------------------------------------------------------------------------------    
    

example = HANK_SAM_agent(**HANK_SAM_Dict)
example.cycles = 0
example.T_cycles = 1
example.wage = 1.2
example.unemp_insurance = unemp_insurance
rfree = example.Rfree
#Income shock distributions for each markov state

# unemployed income shock distribution
# unemployed_IncShkDstn = DiscreteDistribution( np.ones(1), [np.ones(1), np.ones(1)*unemp_insurance] )




#employed Income shock distribution
TranShkDstn_e = MeanOneLogNormal(example.TranShkStd[0],123).approx(example.TranShkCount)

TranShkDstn_u = DiscreteDistribution( np.ones(1),  np.ones(1)*unemp_insurance )
TranShkDstn_e.X  = TranShkDstn_e.X*example.wage
PermShkDstn = MeanOneLogNormal(example.PermShkStd[0],123).approx(example.PermShkCount)

employed_IncShkDstn = combine_indep_dstns(PermShkDstn,TranShkDstn_e)
unemployed_IncShkDstn = combine_indep_dstns(PermShkDstn,TranShkDstn_u)


# Specify list of IncShkDstns for each state
example.IncShkDstn = [ [employed_IncShkDstn, unemployed_IncShkDstn]]

#solve the consumers problem
example.solve()



x = np.linspace(0.0001,10,1000)
plt.plot(x,example.solution[0].cFunc[0](x), label = 'employed' )
plt.plot(x,example.solution[0].cFunc[1](x), label = 'unemployed' )
plt.title('consumption policies')
plt.legend()
plt.show()

x = np.linspace(0.01,1,1000)
plt.plot(x,example.solution[0].vFunc[0](x), label = 'employed' )
plt.plot(x,example.solution[0].vFunc[1](x), label = 'unemployed' )
plt.title('Value functions')
plt.legend()
plt.show()


#example.define_distribution_grid()
#example.calc_transition_matrix()
#example.calc_ergodic_dist()

PermShk_ntrl_msr = deepcopy(PermShkDstn)
PermShk_ntrl_msr.pmf = PermShk_ntrl_msr.X*PermShk_ntrl_msr.pmf
IncShkDstn_ntrl_msr_e = [combine_indep_dstns(PermShk_ntrl_msr,TranShkDstn_e)]
IncShkDstn_ntrl_msr_u = [combine_indep_dstns(PermShk_ntrl_msr,TranShkDstn_u)]

       
example.define_distribution_grid(dist_pGrid = np.array([1]))
example.calc_transition_matrix(IncShkDstn_ntrl_msr_e)
example.calc_ergodic_dist()

steady_Dstn = example.vec_erg_dstn



eigene, ergodic_distre = sp.linalg.eigs(example.tran_matrix_e , k=1 , which='LM')  # Solve for ergodic distribution
ergodic_distre = ergodic_distre.real/np.sum(ergodic_distre.real)
        
eigenu, ergodic_distru = sp.linalg.eigs(example.tran_matrix_u , k=1 , which='LM')  # Solve for ergodic distribution
ergodic_distru = ergodic_distru.real/np.sum(ergodic_distru.real)


C_e = np.dot(example.cPol_Grid_e,ergodic_distre)
C_u = np.dot(example.cPol_Grid_u,ergodic_distru)

C =example.prb_emp * C_e + example.prb_unemp*C_u

print(C)


# a test to see if its stable
dstne = ergodic_distre
dstnu = ergodic_distru




Ce =[]
Cu =[]
Cagg =[]
Aagg =[]
for i in range(20):
    
    Ae = np.dot(example.aPol_Grid_e,dstne)
    Au = np.dot(example.aPol_Grid_u,dstnu)
    
    Ce.append(np.dot(example.cPol_Grid_e,dstne))
    Cu.append(np.dot(example.cPol_Grid_u,dstnu))
    
    Cagg.append(np.dot(example.cPol_Grid_e,dstne)* example.prb_emp + example.prb_unemp*np.dot(example.cPol_Grid_u,dstnu))
    Aagg.append(Ae* example.prb_emp +  Au*example.prb_unemp)
    
    dstne = np.dot(example.tran_matrix_e,dstne)
    dstnu = np.dot(example.tran_matrix_u,dstnu)


#plt.plot(Cagg)
#plt.show()
#print('Aggregate Consumption = ' +str(Cagg[10]))

#plt.plot(Aagg)
#plt.show()
#print('Aggregate Asset = ' +str(Aagg[10]))


'''

#Average value of being employed/unemployed
V_e = np.dot(example.solution[0].vFunc[0](example.dist_mGrid),ergodic_distre)
V_u = np.dot(example.solution[0].vFunc[1](example.dist_mGrid),ergodic_distru)


#Average Value to laborers
V_s =  V_e* example.prb_emp + V_u*example.prb_unemp

print('Value to workers ' + str(V_s))
'''


plt.plot(example.dist_mGrid,steady_Dstn, label = 'all')
plt.plot(example.dist_mGrid,dstne, label = 'employed')
plt.plot(example.dist_mGrid,dstnu, label = 'unemployed')
plt.title('permanent income weighted distribution')
plt.legend()
plt.show()

V_e = example.solution[0].vFunc[0](example.dist_mGrid)
V_u =example.solution[0].vFunc[1](example.dist_mGrid)

V_s = ( np.dot(V_e, steady_Dstn) - np.dot(V_u, steady_Dstn) )

print('Value to workers ' + str(V_s))

print('array of values ' + str(V_e - V_u))



class JAC_agent(HANK_SAM_agent):

    
    def update_solution_terminal(self):
        """
        Update the terminal period solution.  This method should be run when a
        new AgentType is created or when CRRA changes.
        Parameters
        ----------
        none
        Returns
        -------
        none
        """
        #IndShockConsumerType.update_solution_terminal(self)

        # Make replicated terminal period solution: consume all resources, no human wealth, minimum m is 0
        StateCount = self.MrkvArray[0].shape[0]
        self.solution_terminal.cFunc = example.solution[0].cFunc
        self.solution_terminal.vFunc = example.solution[0].vFunc
        self.solution_terminal.vPfunc = example.solution[0].vPfunc
        self.solution_terminal.vPPfunc = example.solution[0].vPPfunc
        self.solution_terminal.mNrmMin = np.zeros(StateCount)
        self.solution_terminal.hRto = np.zeros(StateCount)
        self.solution_terminal.MPCmax = np.ones(StateCount)
        self.solution_terminal.MPCmin = np.ones(StateCount)
        
    def check_markov_inputs(self):
        """
        Many parameters used by MarkovConsumerType are arrays.  Make sure those arrays are the
        right shape.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        StateCount = self.MrkvArray[0].shape[0]



# Jacobian
params = deepcopy(HANK_SAM_Dict)
params['T_cycle'] = 100
params['LivPrb']= params['T_cycle']*example.LivPrb
params['PermGroFac']=params['T_cycle']*example.PermGroFac
params['PermShkStd'] = params['T_cycle']*example.PermShkStd
params['TranShkStd']= params['T_cycle']*example.TranShkStd
params['Rfree'] = params['T_cycle']*[rfree]
params['MrkvArray'] = params['T_cycle']*example.MrkvArray

# ghost

ghost = JAC_agent(**params)
ghost.IncShkDstn = params['T_cycle']*example.IncShkDstn
ghost.del_from_time_inv('Rfree')
ghost.add_to_time_vary('Rfree')
ghost.dstn_0 = ss_dstn
ghost.cycles = 1

ghost.wage = 1.2
ghost.unemp_insurance = params['T_cycle']*[example.unemp_insurance]
ghost.job_find = params['T_cycle']*[example.job_find]
ghost.job_sep = params['T_cycle']* [example.job_sep]

ghost.solve()




ghost.define_distribution_grid(dist_pGrid = params['T_cycle']*[np.array([1])])
ghost.calc_transition_matrix(params['T_cycle']*[IncShkDstn_ntrl_msr_e])


Agg_Cg =[]
Agg_Ag =[]
Agg_ceg =[]
Agg_cug =[]

Agg_aeg =[]
Agg_aug =[]

dstne = ergodic_distre
dstnu = ergodic_distru
for i in range(ghost.T_cycle):
    
    Ae = np.dot(ghost.aPol_Grid_e[i],dstne)
    Au = np.dot(ghost.aPol_Grid_u[i],dstnu)
    Agg_aeg.append(Ae)
    Agg_aug.append(Au)
    
    Ce = np.dot(ghost.cPol_Grid_e[i],dstne)
    Cu = np.dot(ghost.cPol_Grid_u[i],dstnu)
    Agg_ceg.append(Ce)
    Agg_cug.append(Cu)
    
    Agg_Ag.append(Ae* ghost.prb_emp[i] + Au*ghost.prb_unemp[i])
    Agg_Cg.append(Ce* ghost.prb_emp[i] + Cu*ghost.prb_unemp[i])
    
    dstne = np.dot(ghost.tran_matrix_e[i],dstne)
    dstnu = np.dot(ghost.tran_matrix_u[i],dstnu)



#--------------------------------------------------------------------
#jacobian executed here

example2 = JAC_agent(**params)
dx = -.0001




#example2.Rfree = q*[rfree] + [rfree + dx] + (params['T_cycle'] - q )*[rfree]

example2.cycles = 1

example2.wage = example.wage

example2.unemp_insurance = params['T_cycle']*[example.unemp_insurance]

#example2.job_find = params['T_cycle']*[example.job_find]
example2.job_sep = params['T_cycle']* [example.job_sep]
example2.IncShkDstn = params['T_cycle']*[ [employed_IncShkDstn, unemployed_IncShkDstn] ]
example2.del_from_time_inv('Rfree')
example2.add_to_time_vary('Rfree')
example2.dstn_0 = ss_dstn

MrkvArray_dx = np.array( [ [1 - job_sep*(1-(job_find + dx)) , job_find + dx] ,  #" The sum of entries in each column in t should equal one. "
                          
                               [job_sep*(1- (job_find +dx) ), 1 -( job_find+dx) ] ]  ).T


CHist=[]
AHist=[]

test_set =[30]
#for q in range(params['T_cycle']):
    
for q in test_set:
    
    example2.MrkvArray = q*example.MrkvArray + [MrkvArray_dx] + (params['T_cycle'] - q )*example.MrkvArray
    
    example2.job_find =    q*[example.job_find] + [example.job_find + dx]+ (params['T_cycle'] - q )*[example.job_find] 
    
    example2.solve()


    start = time.time()

    example2.define_distribution_grid(dist_pGrid = params['T_cycle']*[np.array([1])])
    example2.calc_transition_matrix(params['T_cycle']*[IncShkDstn_ntrl_msr_e])
    
    print('seconds past : ' + str(time.time()-start))
    
    Agg_C =[]
    Agg_A =[]
    dstne = ergodic_distre
    dstnu = ergodic_distru
    
    Agg_ce =[]
    Agg_cu =[]
    
    Agg_ae =[]
    Agg_au =[]
    
    for i in range(example2.T_cycle):
        
        Ae = np.dot(example2.aPol_Grid_e[i],dstne)
        Au = np.dot(example2.aPol_Grid_u[i],dstnu)
        
        Agg_ae.append(Ae)
        Agg_au.append(Au)
        
        Ce = np.dot(example2.cPol_Grid_e[i],dstne)
        Cu = np.dot(example2.cPol_Grid_u[i],dstnu)
        Agg_ce.append(Ce)
        Agg_cu.append(Cu)
        
        Agg_A.append(Ae* example2.prb_emp[i] + Au*example2.prb_unemp[i])
        Agg_C.append(Ce* example2.prb_emp[i] + Cu*example2.prb_unemp[i])
        
        dstne = np.dot(example2.tran_matrix_e[i],dstne)
        dstnu = np.dot(example2.tran_matrix_u[i],dstnu)



    CHist.append((np.array(Agg_C)-np.array(Agg_Cg))/abs(dx))
    AHist.append((np.array(Agg_A) - np.array(Agg_Ag))/abs(dx))
        
    
    

    

    
plt.plot((np.array(Agg_C)-np.array(Agg_Cg))/abs(dx))
plt.plot(np.zeros(len(Agg_C)))
plt.title('IPR of Aggregate Consumption ')
plt.show()

plt.plot((np.array(Agg_A) - np.array(Agg_Ag))/abs(dx))
plt.plot(np.zeros(len(Agg_A)))
plt.title(' IPR of Aggregate Assets')
plt.show()


plt.plot((np.array(Agg_ce)-np.array(Agg_ceg))/abs(dx))
plt.title('IPR of Employed consumption')
plt.show()
plt.plot((np.array(Agg_cu)-np.array(Agg_cug))/abs(dx))
plt.title('IPR of Unemployed Consumption')
plt.show()

plt.plot((np.array(Agg_ae)-np.array(Agg_aeg))/abs(dx))
plt.title('IPR of Employed Savings')
plt.show()
plt.plot((np.array(Agg_au)-np.array(Agg_aug))/abs(dx))
plt.title('IPR of Unemployed Savings')
plt.show()



fig, axs = plt.subplots(2, 2)
axs[0, 0].plot((np.array(Agg_C)-np.array(Agg_Cg))/abs(dx),'darkgreen' )
axs[0, 0].set_title("IPR of Aggregate Consumption")
axs[1, 0].plot((np.array(Agg_ce)-np.array(Agg_ceg))/abs(dx),'forestgreen' )
axs[1, 0].set_title("IPR of Employed consumption")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot((np.array(Agg_cu)-np.array(Agg_cug))/abs(dx), 'forestgreen')
axs[0, 1].set_title("IPR of Unemployed Consumption")
fig.tight_layout()



fig, axs = plt.subplots(2, 2)
axs[0, 0].plot((np.array(Agg_A)-np.array(Agg_Ag))/abs(dx),'darkgreen' )
axs[0, 0].set_title("IPR of Aggregate Savings")
axs[1, 0].plot((np.array(Agg_ae)-np.array(Agg_aeg))/abs(dx),'forestgreen' )
axs[1, 0].set_title("IPR of Employed Savings")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot((np.array(Agg_au)-np.array(Agg_aug))/abs(dx), 'forestgreen')
axs[0, 1].set_title("IPR of Unemployed Savings")
fig.tight_layout()








