#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 11:46:04 2021

@author: vvasiliau

"""

import numpy as np
import matplotlib.pyplot as plt
import graph_tool.all as gt
from spKMC import *
from spKMC_unitTests import *
from spKMC_sampling import *
import os
# import time
import json
plt.rcParams.update({'font.size': 25})



f=lambda a: int((abs(a)+a)/2)   

def calc_correction(Q_i__, Q_h__, N__, n_tests__, k__, FPR__, FNR__):
    """
    Conversion to the expected number of infectious given an observed nuber I_hat

    Parameters
    ----------
    Q_i__ : fraction of infectious tested
    Q_h__ : fraction of healthy tested
    N__ : total_size of the population
    n_tests__ : total number of tests administered
    k__ : Number of observed positives
    FPR__ : false positive rate
    FNR__ : false negative rate

    Returns
    -------
    I_hat : expected number of infectious

    """
    p = (k__ / n_tests__ - FPR__) / (1 - FNR__ - FPR__)
    I_hat = Q_h__ * N__ / (Q_i__ / p + Q_h__ - Q_i__)
    return I_hat

def s_star(FPR__, FNR__, Q_i__, Q_h__, N__, m__):
    """
    likelihood helper function
    """
    p_star_est = p_star(Q_i__, Q_h__, m__, N__)
    out_prob = p_star_est * (1 - FNR__) + (1 - p_star_est) * FPR__
    return out_prob


def p_star(Q_i__, Q_h__, m__, N__):
    """
    likelihood helper function
    """
    out_prob = Q_i__ * m__ / (Q_i__ * m__ + Q_h__ * (N__ - m__))
    return out_prob

def likelihood(m__, k__, n_tests__, N__, FPR__, FNR__, Q_i__, Q_h__):
    """
    returns likelihood i.e. prob of observing x=k confirmed cases if true parameter I =m

    Parameters
    ----------
    m__ : hypothesis number of infectious
    k__ : number of observed positives
    n_tests__ : number of tests administered
    N__ : total size of the population
    FPR__ : false positive rate
    FNR__ : false negative rate
    Q_i__ : fraction of infectious population tested
    Q_h__ : fraction of healthy population tested

    Returns
    -------
    likelihood_out : posterior likelihood

    """
    s_star_est = s_star(FPR__, FNR__, Q_i__, Q_h__, N__, m__)
    mu = n_tests__ * s_star_est
    var = n_tests__ * s_star_est * (1 - s_star_est)
    likelihood_out = 1 / np.sqrt(2 * np.pi * var) * np.exp( -0.5 * (k__ - mu)**2 / (var) )
    return likelihood_out



def posterior_estimate(I_list__, k__, n_tests__, N__, A_FPR__, B_FPR__, FPR_sample__, A_FNR__, B_FNR__, FNR_sample__,\
                             A_i__, B_i__, Q_i_sample__, A_h__, B_h__, Q_h_sample__, network_prior__, M__ ):  
    """
    Bayesian estimate o fthe number of infectious accounting for uncertainties in measurement procedure
    
    Parameters
    ----------
    I_list__ : Range of potential number of infectious to test
    k__ : Number of observed positives
    n_tests__ : Total number of tests administered
    N__ : Total size of the population
    A_FPR__ : alpha parameter for the beta distribution of FPR
    B_FPR__ : beta parameter for the beta distribution of FPR
    FPR_sample__ : bool if sample from the beta distribution for FPR
    A_FNR__ : alpha parameter for the beta distribution of FNR
    B_FNR__ : beta parameter for the beta distribution of FNR
    FNR_sample__ : bool if sample from the beta distribution for FNR
    A_i__ : alpha parameter for the beta distribution of Q_i which is the fraction of infectious tested
    B_i__ : beta parameter for the beta distribution of Q_i which is the fraction of infectious tested
    Q_i_sample__ : bool if sample from the beta distribution for Q_i
    A_h__ : alpha parameter for the beta distribution of Q_h which is the fraction of the healthy tested
    B_h__ : beta parameter for the beta distribution of Q_h which is the fraction of the healthy tested
    Q_h_sample__ : bool if sample from the beta distribution for Q_h
    network_prior__ : prior based on simulations on an ensemble of networks from which this one is believed to be
    M__ : number of samples to draw for the posterior estimation

    Returns
    -------
    I_est : posterior estimate of the number of infectious
    """
    
    likelihood_array = np.zeros(len(I_list__))
    norm = 0.0;

    if Q_i_sample__ == True:
        Q_i_set = np.random.beta(A_i__, B_i__, M__)
    else:
        Q_i_mean = A_i__ / (A_i__ + B_i__)
        Q_i_set = Q_i_mean * np.ones(M)
    
    if Q_h_sample__ == True:
        Q_h_set = np.random.beta(A_h__, B_h__, M__)
    else:
        Q_h_mean = A_h__ / (A_h__ + B_h__)
        Q_h_set = Q_h_mean * np.ones(M)
    
    if FPR_sample__ == True:
        FPR_set = np.random.beta(A_FPR__, B_FPR__, M__)
    else:
        FPR_mean = A_FPR/(A_FPR+B_FPR)
        FPR_set = FPR_mean*np.ones(M)
        
    if FNR_sample__ == True:
        FNR_set = np.random.beta(A_FNR__, B_FNR__, M__)
    else:
        FNR_mean = A_FNR__ / (A_FNR__ + B_FNR__)
        FNR_set = FNR_mean * np.ones(M__)
            
    for l in I_list__:
        likelihood_tmp = 0.0;
        for i in range(M__):
            Q_i = Q_i_set[i]
            Q_h = Q_h_set[i]
            FPR = FPR_set[i]
            FNR = FNR_set[i]
            likelihood_tmp = likelihood_tmp+likelihood(m__ = l, k__ = k__, n_tests__ = n_tests__,
                                                       N__ = N__, FPR__ = FPR, FNR__ = FNR, Q_i__ = Q_i, Q_h__ = Q_h)
        likelihood_array[l]= 1 / M__ * likelihood_tmp;
        norm = norm + 1 / M__ * likelihood_tmp;
    likelihood_array = likelihood_array / norm
    I_est = sum(likelihood_array * np.array(I_list__))
    
    return I_est


def calc_infected(n__,T__, R__):
    """
    Perform biased testing of infected population
    n__ - total length of sampling array. E.g. number of infected/healthy
    T__ - number of tests to perform
    R__ - in case of infected a TPR , in case of healthy FPR
    
    Returns
    -------
    count - number of individuals, identified as ill.
    n_tests - number of tests performed
    """
        
    if T__ <= n__:
        count =  np.sum(np.random.rand(T__) < R__)
        n_tests = T__
    else:
        count =  np.sum(np.random.rand(n__) < R__)
        n_tests = n__
        
    return count,  n_tests





if __name__ =="__main__":
    
    
    """
    f.write("epidemic_params = \{" "" \}")
    print(f"alpha"{epi_params["alpha"]})
    """
    
    # ========================================
    # epidemic parameters
    # ========================================
    epi_params = {"alpha":1/5, "beta": 3/14,"gamma":1/14}
    
    # ========================================
    #total time of observation
    # ========================================
    T = 100
    t_array = list(range(T))

    # ========================================
    # network parameters
    # ========================================
    N = 0.5*10**3 #size of the network
    num_sim = 1 # number of epidemic simulations to do
    num_sources = 1 #number of source nodes
    m,gamma = 3,1 # network model parameters
    network_param_label = f"price_network_m_{m}_gamma_{gamma}"
    g_lambda = lambda N: gt.price_network(N, m, gamma, directed=False) 
    g = g_lambda(N) #generate a network
    
    # ========================================
    # Sample epidemic trajectories and extract the infected which is what we are interesed in
    # ========================================
    g_spkmc = generate_directed_graph(g) #turn an undirected network to a directed network
    ensemble_epi_trajectories = sample_epi_trajectories(g_spkmc, N, num_sim, [], epi_params, T)
    X = ensemble_epi_trajectories["I"]
    
    # ========================================
    # ensemble of networks, network prior, bayes vars
    # ========================================
    num_sim_network = 10
    sample_epi_ensemble = sample_epi_trajectories_multi_rnd_sources_ensemble(g_lambda, N, \
                                                            num_sim_network, num_sources, epi_params, T)
    network_sample = False # if perofrm network brior
    if network_sample == True:
        delta= 5 #half the time window to consider for network prior
    else:
        delta = np.nan
    FPR_sample = True # if perform sampling for FPR
    FNR_sample = True # if perform sampling for FNR
    Q_i_sample = True # if perform sampling for Q_i
    Q_h_sample = True # if perform sampling for Q_h
    I_hat= np.zeros_like(X) #for storage of the Bayesian estimate
    M = 10 # number of samples to estimate posterior
    
    # =======================================
    # Testing parameters
    # =======================================
    X_hat = np.zeros(X.shape)
    N_tests = np.zeros(X.shape)
    
    (A_i,B_i) = (4,16)
    Q_i = A_i/(A_i+B_i)
    
    A_h = 5;
    B_h = 95;
    Q_h = A_h/(A_h+B_h)
    
    
    A_FPR,B_FPR = 10,90
    FPR = A_FPR/(A_FPR+B_FPR)
    
    A_FNR,B_FNR = 5,95
    FNR = A_FNR/(A_FNR+B_FNR)
    
    
    # =======================================
    # Testing procedure
    # =======================================
    for t in range(0, T):
        for i in range(num_sim):
            X_true= X[i,t]
            Tests_I = int(Q_i * X_true)
            Tests_H = int(Q_h * (N - X_true))
            N_tests[i,t]=Tests_I + Tests_H
            N_I_calc, ntests_I_calc = calc_infected(n__ = int(X_true), T__ = Tests_I, R__ = 1 - FNR)
            N_H_calc, ntests_H_calc =calc_infected(n__ = int(N - X_true), T__ = Tests_H, R__ = FPR)
            X_hat[i,t] = N_I_calc + N_H_calc
    
    
    # =======================================
    # Mean field correction 
    # =======================================
    mean_field = calc_correction(Q_i__ = Q_i, Q_h__ = Q_h, N__ = N, n_tests__ = N_tests, k__ = X_hat,\
                                         FPR__ = FPR, FNR__ = FNR)[0]
    
    
    
    # =======================================
    # Bayesian correction 
    # =======================================
    
    for t in range(X_hat.shape[1]):
        for sim in range(X_hat.shape[0]):
            k = X_hat[sim,t]
            nt = N_tests[sim,t]
            if network_sample == True:
                prior_distr = get_prior_distr( f(t-delta), t+delta, sample_epi_ensemble["I"], int(N))
            else:
                prior_distr = np.ones(int(N))
            I_est = posterior_estimate(I_list__ = np.array(range(0,int(N))), 
                                       k__ = k, 
                                       n_tests__ = nt, 
                                       N__ = N, 
                                       A_FPR__ = A_FPR, B_FPR__ = B_FPR, FPR_sample__ = FPR_sample, 
                                       A_FNR__ = A_FNR, B_FNR__ = B_FNR, FNR_sample__ = FNR_sample,
                                       A_i__ = A_i, B_i__ = B_i, Q_i_sample__ = Q_i_sample, 
                                       A_h__ = A_h, B_h__ = B_h, Q_h_sample__ = Q_h_sample,
                                       network_prior__ = prior_distr, M__ = M) 
            I_hat[sim,t] = I_est
            
    
    
    # =======================================
    # Plotting, save data
    # =======================================
    fig,ax =plt.subplots( figsize =(10,8))
    ax.plot(t_array,(X.mean(0)),alpha = 0.7,linewidth=4,label = "True $I$",color="lightseagreen")
    ax.plot(t_array,mean_field,color="darkviolet", label = "Mean field $\hat{I}$")
    ax.plot(t_array,I_hat.mean(0), label ="Bayesian $\hat{I}$",c="darkred",linewidth=4,linestyle = "-.")
    ax.plot(t_array,X_hat[0], label ="Observed $N_p$",c="goldenrod",linewidth=4,linestyle ="--")
    ax.set_xlabel("$t$")    
    # ax.set_ylim(0,N)
    ax.legend(loc=1,ncol=1,shadow=True)
    plt.tight_layout()
    # seed = int(time.time())
    name = "name_of_directory"
    os.mkdir(f"./results/simu_{name}")
    plt.savefig(f"./results/simu_{seed}/SEIR_measurement_reconstruction.pdf")
    
    param_dict = {"epi_params": epi_params, "T": T, "N": N, "num_sim": num_sim,
                  "num_sources": num_sources, 
                  "network_param_label": network_param_label, 
                  "num_sim_network": num_sim_network,#, 
                  "delta": delta, "FPR_sample": FPR_sample, 
                  "FNR_sample": FNR_sample, "Q_i_sample": Q_i_sample,
                  "Q_h_sample": Q_h_sample,"M": M,"A_i": A_i,"A_h": A_h, 
                  "A_FPR": A_FPR,"B_FPR": B_FPR, "A_FNR": A_FNR , "B_FNR": B_FNR}
    
        
    with open(f'./results/simu_{name}/parameters.json', 'w') as fp:
        json.dump(param_dict, fp)
        
    np.save(f"./results/simu_{name}/mean_field.npy",mean_field)
    np.save(f"./results/simu_{name}/I_hat.npy",I_hat)
    np.save(f"./results/simu_{name}/X_hat.npy",X_hat)
    np.save(f"./results/simu_{name}/X.npy",X)
        
        
        