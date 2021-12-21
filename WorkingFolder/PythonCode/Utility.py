# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numba as nb
import numpy as np
from interpolation import interp, mlinterp
from numba import jit,njit, float64, int64, boolean



# + code_folding=[0]
def policyfunc(lc,
               a_star,
               σ_star,
               discrete = True):
    """
     * ifp is an instance of IFP
        * a_star is the endogenous grid solution
        * σ_star is optimal consumption on the grid    
    """
    if discrete==True:
        # Create consumption function by linear interpolation
        σ =  lambda a, z_idx: interp(a_star[:, z_idx], σ_star[:, z_idx], a) 
    else:
        # get z_grid 
        z_val = lc.z_val 

        # Create consumption function by linear interpolation
        a = a_star[:,0]                                ## aseet grid 
        σ =  interpolate.interp2d(a, z_val, σ_star.T) 
    
    return σ


# + code_folding=[0]
def policyfuncMA(lc,
                 aϵ_star,
                 σ_star):
    """
     * ifp is an instance of IFP
        * aϵ_star is the endogenous grid solution
        * σ_star is optimal consumption on the grid    
    """
   
   # get s_grid and eps grid 
    s_grid = lc.s_grid
    eps_grid = lc.eps_grid 

    # Create consumption function by linear interpolation 
    σ =  interpolate.interp2d(eps_grid, s_grid, σ_star.T) 
    
    return σ


# + code_folding=[]
@njit
def policyfuncMAjit(lc,
                 aϵ_star,
                 σ_star):
    """
     * ifp is an instance of IFP
        * aϵ_star is the endogenous grid solution
        * σ_star is optimal consumption on the grid    
    """
   
   # get z_grid 
    s_grid = lc.s_grid
    eps_grid = lc.eps_grid 
    
    # Create consumption function by linear interpolation
    σ =  lambda eps,a: mlinterp((eps_grid,s_grid),σ_star.T,(eps,a))
    
    return σ


# -

## a simple function that computes steady state of 2-state markov
@njit
def cal_ss_2markov(P):
    ## an analytical solution for 2-state markov for double checking 
    ## when P's row sums up to 1
    #P.T = [[q,1-p],[1-q,p]]
    q = P.T[0,0]
    p = P.T[1,1]
    h = (1-p)/(2-p-q)
    return np.array([h,1-h])


# ==============================================================================
# ============== Functions for generating state space grids  ===================
# ============= Copied from HARK  ==============================================
# ==============================================================================
def make_grid_exp_mult(ming, maxg, ng, timestonest=20):
    """
    Make a multi-exponentially spaced grid.

    Parameters
    ----------
    ming : float
        Minimum value of the grid
    maxg : float
        Maximum value of the grid
    ng : int
        The number of grid points
    timestonest : int
        the number of times to nest the exponentiation

    Returns
    -------
    points : np.array
        A multi-exponentially spaced grid

    Original Matab code can be found in Chris Carroll's
    [Solution Methods for Microeconomic Dynamic Optimization Problems]
    (http://www.econ2.jhu.edu/people/ccarroll/solvingmicrodsops/) toolkit.
    Latest update: 01 May 2015
    """
    if timestonest > 0:
        Lming = ming
        Lmaxg = maxg
        for j in range(timestonest):
            Lming = np.log(Lming + 1)
            Lmaxg = np.log(Lmaxg + 1)
        Lgrid = np.linspace(Lming, Lmaxg, ng)
        grid = Lgrid
        for j in range(timestonest):
            grid = np.exp(grid) - 1
    else:
        Lming = np.log(ming)
        Lmaxg = np.log(maxg)
        Lstep = (Lmaxg - Lming) / (ng - 1)
        Lgrid = np.arange(Lming, Lmaxg + 0.000001, Lstep)
        grid = np.exp(Lgrid)
    return grid


