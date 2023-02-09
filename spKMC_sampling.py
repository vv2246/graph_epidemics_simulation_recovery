# =======================sp-KMC-SEIR sampling utility algorithms=================================================
import graph_tool.all as gt
import numpy as np
from spKMC import *
from spKMC_unitTests import *
       
def sample_epi_trajectories(g__, N__, num_sim__, source__, epi_params__, T__):
    # samples epidemic trajectories from num_sim__ with fixed source, network
    S_dynamics = np.zeros((num_sim__,T__))
    E_dynamics = np.zeros((num_sim__,T__))
    I_dynamics = np.zeros((num_sim__,T__))
    R_dynamics = np.zeros((num_sim__,T__))
    ensemble_epi_trajectories = {"S":  S_dynamics, "E": E_dynamics, "I": I_dynamics, "R" : R_dynamics}
    for idx_count in range(0,num_sim__):
        if (source__ == []):
            source = list(np.random.randint(0,N__, size=1))
            dynamics_hidden_state = spKMC_SEIR_full_state(g__, epi_params__, source)
        else:
            dynamics_hidden_state = spKMC_SEIR_full_state(g__, epi_params__, source__)
            
        SEIR_dict_elem = event_driven_state_extraction(dynamics_hidden_state, T__)
        
        ensemble_epi_trajectories["S"][idx_count,:] = SEIR_dict_elem["S"]
        ensemble_epi_trajectories["E"][idx_count,:] = SEIR_dict_elem["E"]
        ensemble_epi_trajectories["I"][idx_count,:] = SEIR_dict_elem["I"]
        ensemble_epi_trajectories["R"][idx_count,:] = SEIR_dict_elem["R"]
        print(f"{100*idx_count/num_sim__}%", end =" ")
        
    return ensemble_epi_trajectories 
    
def sample_epi_trajectories_multi_rnd_sources_ensemble(g_lambda__, N__, num_sim__, num_sources__, epi_params__, T__):
    # samples epidemic trajectories from num_sim__ networks that are rewired versions of infput graph and 
    # and for each network from multiple random sources and returns epidemic trajectories 
    S_dynamics = np.zeros((num_sim__,T__))
    E_dynamics = np.zeros((num_sim__,T__))
    I_dynamics = np.zeros((num_sim__,T__))
    R_dynamics = np.zeros((num_sim__,T__))
    ensemble_epi_trajectories = {"S":  S_dynamics, "E": E_dynamics, "I": I_dynamics, "R" : R_dynamics}
    for idx_count in range(0,num_sim__):
        g = g_lambda__(N__)
        g_semidir = generate_directed_graph(g)
        
        source__array = list(np.random.randint(0,N__, size=num_sources__))
        dynamics_hidden_state = spKMC_SEIR_full_state(g_semidir, epi_params__, source__array)
        SEIR_dict_elem = event_driven_state_extraction(dynamics_hidden_state, T__)
        
        ensemble_epi_trajectories["S"][idx_count,:] = SEIR_dict_elem["S"]
        ensemble_epi_trajectories["E"][idx_count,:] = SEIR_dict_elem["E"]
        ensemble_epi_trajectories["I"][idx_count,:] = SEIR_dict_elem["I"]
        ensemble_epi_trajectories["R"][idx_count,:] = SEIR_dict_elem["R"]
        print(f"{100*idx_count/num_sim__}%", end =" ")
        
    return ensemble_epi_trajectories 
    
def sample_epi_trajectories_multi_rnd_sources_rewired_ensemble(g__, num_sim__, num_sources__, epi_params__, T__, \
                                     network_model_string__ = "configuration", n_iter_rewire__ = 1, edge_sweep_rewire__ = False):
    # samples epidemic trajectories from num_sim__ networks that are rewired versions of infput graph and 
    # and for each network from multiple random sources and returns epidemic trajectories 
    S_dynamics = np.zeros((num_sim__,T__))
    E_dynamics = np.zeros((num_sim__,T__))
    I_dynamics = np.zeros((num_sim__,T__))
    R_dynamics = np.zeros((num_sim__,T__))
    ensemble_epi_trajectories = {"S":  S_dynamics, "E": E_dynamics, "I": I_dynamics, "R" : R_dynamics}
    for idx_count in range(0,num_sim__):
        gt.random_rewire(g__, model = network_model_string__, edge_sweep = edge_sweep_rewire__, n_iter = n_iter_rewire__)
        source__array = list(np.random.randint(0,g__.get_vertices().shape[0], size=num_sources__))
        dynamics_hidden_state = spKMC_SEIR_full_state(g__, epi_params__, source__array)
        SEIR_dict_elem = event_driven_state_extraction(dynamics_hidden_state, T__)
        
        ensemble_epi_trajectories["S"][idx_count,:] = SEIR_dict_elem["S"]
        ensemble_epi_trajectories["E"][idx_count,:] = SEIR_dict_elem["E"]
        ensemble_epi_trajectories["I"][idx_count,:] = SEIR_dict_elem["I"]
        ensemble_epi_trajectories["R"][idx_count,:] = SEIR_dict_elem["R"]
        print(f"{100*idx_count/num_sim__}%", end =" ")
        
    return ensemble_epi_trajectories 
    
def spKMC_SEIR_full_state_rnd_sources(g_base__, epi_params__, num_sources__, debug__=0): 
    #return hidden states of epidemic dynamic from random num_sources__ sources
    assert g_base__.is_directed() == 1
    
    g__ = gt.Graph(directed=True)
    edge_weight_prop = g__.new_edge_property("double")
    g__.add_edge_list(g_base__.get_edges().copy())
    
    node_incubation_weights = np.random.exponential(1/epi_params__["alpha"], [g__.get_vertices().shape[0],1])
    edge_transmission_weights = np.random.exponential(1/epi_params__["beta"], [g__.get_edges().shape[0],])
    node_recovery_weights = np.random.exponential(1/epi_params__["gamma"], [g__.get_vertices().shape[0],1])

    edge_weights_filter = extract_edge_SEIR_weights(g__,node_recovery_weights,node_incubation_weights,edge_transmission_weights)
    edge_weight_prop.get_array()[:] = edge_weights_filter
    isdirected = g__.is_directed()
       
    dynamics_hidden_state_list = []
    source__array = np.random.randint(0,g__.get_vertices().shape[0], size=num_sources__)
    
    dist_list = [gt.shortest_distance(g__, source=g__.vertex(x), weights = edge_weight_prop, directed = isdirected)\
                         for x in source__array]
    
    dist_np_list = [np.asarray(x.get_array()[:]).reshape(g__.get_vertices().shape[0],1) for x in dist_list]
    
    
    dynamics_hidden_state_list = [ {"epidemic_geodesics": dist_x, "node_recovery": node_recovery_weights,\
                                 "node_incubation": node_incubation_weights} for dist_x in  dist_np_list]
    
    return dynamics_hidden_state_list

def sample_epi_trajectories_rewired_ensemble(g__, num_sim__, num_networks__, epi_params__, T__, \
                                     network_model_string__ = "configuration", n_iter_rewire__ = 1, edge_sweep_rewire__ = False):
    # samples epidemic trajectories from num_networks__ that are rewired versions of input graph and 
    # num_sim__ for each network from random sources and returns epidemic trajectories
    M = num_sim__*num_networks__
    S_dynamics = np.zeros((M,T__))
    E_dynamics = np.zeros((M,T__))
    I_dynamics = np.zeros((M,T__))
    R_dynamics = np.zeros((M,T__))
    ensemble_epi_trajectories = {"S":  S_dynamics, "E": E_dynamics, "I": I_dynamics, "R" : R_dynamics}
    idx_count = 0
    for net_id in range(0,num_networks__):
        gt.random_rewire(g__, model = network_model_string__, edge_sweep = edge_sweep_rewire__, n_iter = n_iter_rewire__)
        dynamics_hidden_state_list = spKMC_SEIR_full_state_rnd_sources(g__, epi_params__, num_sim__)
        SEIR_dict_list = [event_driven_state_extraction(dynamics_hidden_tmp, T__) for dynamics_hidden_tmp in \
                  dynamics_hidden_state_list] 
        for SEIR_dict_elem in SEIR_dict_list:
            ensemble_epi_trajectories["S"][idx_count,:] = SEIR_dict_elem["S"]
            ensemble_epi_trajectories["E"][idx_count,:] = SEIR_dict_elem["E"]
            ensemble_epi_trajectories["I"][idx_count,:] = SEIR_dict_elem["I"]
            ensemble_epi_trajectories["R"][idx_count,:] = SEIR_dict_elem["R"]
            idx_count += 1
            
        print(f"{100*net_id/num_networks__}%", end =" ")

    return ensemble_epi_trajectories

def get_prior_distr(t_start__, t_end__, X_dynamics__, N__):
    # Returns estimated prior distribution from epidemic trajectories by using kernel density 
    M = X_dynamics__.shape[0]
    T = X_dynamics__.shape[1]
    
    print(f"Estimating kde priors from time {t_start__} until {t_end__}")
    x_d = np.linspace(0, N__, N__)
    
    data = X_dynamics__[:,t_start__:t_end__+1].ravel()
    num_eff_samples =  np.sum(data>0)
    print("number of samples >0",num_eff_samples)
    sigma = max(1,(4/3)*np.std(data)*(num_eff_samples**(-1/5)))
    print(f"Using sigma {sigma}....")
        
    prior_distr = sum(np.exp(-(xi-x_d)**2/(2*sigma**2) ) for xi in data if xi > 0)
    if (np.sum(prior_distr)>0):
        prior_distr = prior_distr / sum(prior_distr)

    return prior_distr