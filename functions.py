import numpy as np

from parameters import EPSILON


def get_equilibrium(a, b, z, W, n, wealth, shot_firm, competitive_eq):
    if competitive_eq == "markup":
        return compute_equilibrium_markup(a, b, z, W, n, shot_firm)
    else:
        return compute_equilibrium2(a, b, z, W, n, wealth, shot_firm)


def compute_equilibrium(a, b, z, W, n, wealth, shot_firm=None):
    #start = time.clock()
    # first equation: good market balance
    Wtilde = (1-a) * b * W
    invMat = np.linalg.inv(np.eye(n) - Wtilde)
    tV = np.dot(invMat, (wealth/n)*np.ones([n,1]))
    V = np.transpose(tV)
    
    #eq1 = time.clock()
    #print("eq1", eq1 - start)
    # second equation: labor market balance
    h = np.sum(a*b*V)
    #eq2 = time.clock()
    
    # third equation: optimum production
    logP = np.dot(a*b*np.log(h) - np.log(z) - b*np.log(b) - (b-1)*np.log(V), invMat)
    P = np.exp(logP)
    X = V / P
    
    #eq3 = time.clock()
    #print("eq3", eq3 - eq2)
    #return
    return {"X":X.flatten(), "P":P.flatten(), "h": h}


def compute_equilibrium2(a, b, z, W, n, wealth, shot_firm=None): # with solve instead of inv
    #start = time.clock()
    # first equation: good market balance
    if shot_firm==None:
        hh_demand = (wealth/n)*np.ones([n,1])
    else:
        hh_demand = (wealth/(n-1))*np.ones([n,1])
        hh_demand[shot_firm] = 0
    Wtilde = (1-a) * b * W
    tV = np.linalg.solve(np.eye(n) - Wtilde, (wealth/n)*np.ones([n,1]))
    V = np.transpose(tV)
        
    #eq1 = time.clock()
    #print("eq1", eq1 - start)
    # second equation: labor market balance
    h = np.sum(a*b*V)
    #eq2 = time.clock()
    
    # third equation: optimum production
    correct = True
    if correct:
        deltaW = W.sum(axis=0)-1
        bModif = b*(1+deltaW*(1-a))
        #print('bModif', np.min(bModif), np.max(bModif))
        tlogP = np.linalg.solve(
            np.transpose(np.eye(n) - Wtilde), 
            np.transpose(a*b*np.log(h) - np.log(z) - bModif*np.log(b) - (bModif-1)*np.log(V))
        )
    else:
        tlogP = np.linalg.solve(
            np.transpose(np.eye(n) - Wtilde), 
            np.transpose(a*b*np.log(h) - np.log(z) - b*np.log(b) - (b-1)*np.log(V))
        )
    logP = np.transpose(tlogP)
    P = np.exp(logP)
    X = V / P

    check=False
    if check:
        epsilon = 1e-10
        #print('\nlabor supply: ', 1)
        #print("labor demand: ", np.sum(a*b*P*X)/h)
        dif1 = abs(1-np.sum(a*b*P*X)/h)>epsilon
        #print("Labor market check: ", dif1)
        
        #print('\ngood supply: ', X)
        #print("good demand: ", 1/P+np.transpose(np.dot(Wtilde*P*X/np.transpose(P), np.ones([n,1]))))
        dif2 = abs(X-1/P-np.transpose(np.dot(Wtilde*P*X/np.transpose(P), np.ones([n,1]))))>epsilon
        #print("Good markets check: ", dif2)
        
        labor_input = a*b*P*X/h
        good_flow = Wtilde*P*X/np.transpose(P)
        
        #print(W)
        GG = (good_flow/((1-a)*W))**((1-a)*b*W)
        GG[np.isnan(GG)]=1
        
        #print('\nCalculated prod:', z * (labor_input/a)**(a*b) * GG.prod(axis=0))
        #print("Prod:", X)
        dif3 = abs(X-z * (labor_input/a)**(a*b) * GG.prod(axis=0))>epsilon
        #print("Prod check: ", dif3)
        
        dif4 = abs(P*z*a*b*a**(-a*b) * labor_input**(a*b-1) * GG.prod(axis=0) - h)>epsilon
        #print('Optim labor condition:', dif4)
        
        P1 = P * z * (labor_input/a)**(a*b) * GG.prod(axis=0) * (1-a)*W*b / good_flow
        dif5 = abs(P - np.nanmean(P1, axis=1))>epsilon
        #print('Optim good conditions:', dif5)
        
        # Bilan
        print((dif1.any(), dif2.any(), dif3.any(), dif4.any(), dif5.any()))
    #eq3 = time.clock()
    #print("eq3", eq3 - eq2)
    #return
    return {"X":X.flatten(), "P":P.flatten(), "h": h}


def get_alpha(a, W):
    return a + (1 - a) * np.sum(W, axis=0)


def compute_cost_gap(a, b, z, W, n, wealth, Wbar, supplier_id_list, alternate_supplier_id_list, shot_firm, competite_eq):
    '''Compute a network-level metric. It is the mean of all so-called firm-level penalty.
    A firm's penalty is the difference between the maximum profit reachable with perfect anticipation and the current profit
    '''
    all_indiv_score = [
        evalute_cost_penalty(firm_id, a, b, z, W, n, wealth, Wbar, supplier_id_list, alternate_supplier_id_list,
                             shot_firm, competite_eq) for firm_id in range(n)]
    return np.mean(all_indiv_score)



def compute_equilibrium_markup(a, b, z, W, n, shot_firm=None):  # with solve instead of inv
    alpha = get_alpha(a, W)
    # Get V by solving the eigenvector pb
    M = (1 / alpha) * ((a / n) + (1 - a)[np.newaxis, :] * W)
    eigenvalues, eigenvectors = np.linalg.eig(M)
    # Since eigenvalues can have some floating-point errors, we check if the eigenvalue is close to 1
    index = np.isclose(eigenvalues, 1)
    # Extract the corresponding eigenvectors
    eigenvectors_1 = eigenvectors[:, index]
    ## Normalize the eigenvector (optional, depending on your application)
    # eigenvector_1 = eigenvectors_1 / np.linalg.norm(eigenvectors_1)
    v_unnormalized = np.transpose(eigenvectors_1)
    # check no imaginary part
    if np.sum(np.imag(v_unnormalized)) > EPSILON:
        print("IMAGINARY PART")
    v_unnormalized = np.real(v_unnormalized)
    # check all same sign
    if (v_unnormalized > 0).any() and (v_unnormalized < 0).any():
        raise ValueError("AT LEAST ONE NEGATIVE")

    v_unnormalized = abs(v_unnormalized).flatten()

    kappa = n / np.sum(v_unnormalized * a / alpha)
    v = kappa * v_unnormalized

    # Get p, by passing prod function to the log and solve Ap = b
    b_vector = (-np.log(z) + b * alpha * np.log(alpha) + (1 - b * alpha) * np.log(v))[:, np.newaxis]
    A_matrix = np.eye(n) - (b * (1 - a))[:, np.newaxis] * W

    # n = len(alpha)
    # A_matrix = np.zeros((n, n))
    # b_vector = np.zeros(n)
    # for i in range(n):
    #    b_vector[i] = -np.log(z[i]) + b[i] * alpha[i] * np.log(alpha[i]) + (1 - b[i] * alpha[i]) * np.log(v[i])
    #    for j in range(n):
    #        A_matrix[i, j] = -b[i] * (1 - a[i]) * W[j, i]
    # print(b_vector - b_vector_1)
    # print(A_matrix - A_matrix_1)
    # Solve the linear system A * log_p = b
    # pd.DataFrame(A_matrix).to_csv('A_matrix.csv', index=False)
    # pd.DataFrame(W).to_csv('W.csv', index=False)
    log_p = np.linalg.solve(A_matrix, b_vector)
    p = np.exp(log_p)

    # Get x
    v = v.flatten()
    p = p.flatten()
    x = v / p

    check = False
    if check:
        l = v * a / alpha
        # G = np.transpose(v * (1 - a) / alpha * W.T / np.transpose(p))
        G = v * (1 - a) * W / alpha / p[:, np.newaxis]
        L = np.sum(l)
        B = L
        print("Quantity of labor", L, "Budget", B)
        final_demand = B / (n * p)
        intermediary_demand = np.sum(G, axis=1)
        supply = x
        print("supply", supply)
        print("total demand", intermediary_demand + final_demand)
        print("intermediary_demand", intermediary_demand)
        print("final_demand", final_demand)
        print("dif", supply - (intermediary_demand + final_demand) > 1e-10)

    return {"X": x.flatten(), "P": p.flatten(), "h": 1}


def compute_partial_equilibrium_markup_and_profit(a, b, z, W, n, eq, firms_within_tiers, id_rewiring_firm, shot_firm):
    '''Compute equilibrium for selected firms only'''
    # Reduce the system
    W_reduced = W[np.ix_(firms_within_tiers, firms_within_tiers)]
    a_reduced = a[firms_within_tiers]
    b_reduced = b[firms_within_tiers]
    z_reduced = z[firms_within_tiers]
    n_reduced = len(firms_within_tiers)
    alpha = get_alpha(a, W)
    alpha_reduced = alpha[firms_within_tiers]
    firms_notin_tiers = list(set(range(n)) - set(firms_within_tiers))

    if len(firms_within_tiers) == n:
        partial_eq = compute_equilibrium_markup(a_reduced, b_reduced, z_reduced, W_reduced, n_reduced)

    else:
        # Solve market clearing
        # M_reduced = (1 / alpha_reduced) * ((a_reduced / n) + (1 - a_reduced)[np.newaxis, :] * W_reduced)
        # Omega1 = (1 / alpha[firms_within_tiers]) * np.sum(((a[firms_notin_tiers] / n) + (1 - a[firms_notin_tiers])[np.newaxis, :] * W[np.ix_(firms_within_tiers, firms_notin_tiers)]), axis=1)
        M_reduced = ((1 - a_reduced) / alpha_reduced)[np.newaxis, :] * W_reduced
        V_eq = eq['X'] * eq['P']
        Omega1 = 1 + np.sum(
            (V_eq[firms_notin_tiers] * (1 - a[firms_notin_tiers]) / alpha[firms_notin_tiers])[np.newaxis, :] * W[
                np.ix_(firms_within_tiers, firms_notin_tiers)], axis=1)
        V_reduced = np.linalg.solve(np.eye(len(firms_within_tiers)) - M_reduced, Omega1)

        # print('estimated V: '+str(V_reduced[firms_within_tiers.index(id_rewiring_firm)]))
        # print('current V: '+str(V_eq[id_rewiring_firm]))
        # print('final demand: '+str(np.sum(V_eq * a / alpha) / n))
        # print('other demand: ', np.sum( ((1 - a) / alpha)[np.newaxis, :] * W, axis=1)[id_rewiring_firm])
        # print(np.sum(V_eq * (1 - a) * W / alpha / eq['P'][:, np.newaxis], axis=1)[id_rewiring_firm])
        id_rewiring_firm_reduced = firms_within_tiers.index(id_rewiring_firm)
        # print(V_reduced[id_rewiring_firm_reduced])
        # print(Omega1)
        # exit()
        # Solve optimal production
        b_vector_reduced = (-np.log(z_reduced) + b_reduced * alpha_reduced * np.log(alpha_reduced) + (
                    1 - b_reduced * alpha_reduced) * np.log(V_reduced))[:, np.newaxis]
        Omega2 = np.sum(b_reduced * (1 - a_reduced) * W[np.ix_(firms_notin_tiers, firms_within_tiers)] * np.log(
            eq['P'][firms_notin_tiers][:, np.newaxis]), axis=0)[:, np.newaxis]
        A_matrix_reduced = np.eye(n_reduced) - (b_reduced * (1 - a_reduced))[:, np.newaxis] * W_reduced
        log_P_reduced = np.linalg.solve(A_matrix_reduced, b_vector_reduced + Omega2)
        P_reduced = np.exp(log_P_reduced)

        X_reduced = V_reduced / P_reduced

        # Build Partial Eq
        P_reduced = P_reduced.flatten()
        X_reduced = X_reduced.flatten()
        partial_eq = {
            "X": X_reduced,
            "P": P_reduced,
            'firms_within_tiers': firms_within_tiers
        }

    # alpha_rewiring_firm = a[id_rewiring_firm] + (1 - a[id_rewiring_firm]) * np.sum(W[:, id_rewiring_firm])
    # computeCostMarkup(id_rewiring_firm, partial_eq, firms_within_tiers, eq, W, a, alpha_rewiring_firm)
    estimated_new_cost = partial_eq['P'][firms_within_tiers.index(id_rewiring_firm)]

    return partial_eq, estimated_new_cost



def compute_cost(firm_id, a, b, W, X, P, h):
    labor_need = a[firm_id] * b[firm_id] * P[firm_id] * X[firm_id] / h
    good_needs = (1 - a[firm_id]) * b[firm_id] * P[firm_id] * X[firm_id] * np.transpose(W[:,firm_id]) / P
    cost = h * labor_need + np.sum(P * good_needs)
    return cost



def evaluate_best_alternative_cost(firm_id, a, b, z, W, n, wealth, Wbar, supplier_id_list, alternate_supplier_id_list, shot_firm, competite_eq):
    """Compute the best profit reachable by a switch
    Nonmyopic version: the firm update the W based on the switch and compute the new network-wide equilibrium
    """
    min_cost = 9999
    for id_visited_supplier in alternate_supplier_id_list[firm_id]:
        W[id_visited_supplier, firm_id] = Wbar[id_visited_supplier, firm_id]
        for id_replaced_supplier in supplier_id_list[firm_id]:
            #print('test', firm_id, id_replaced_supplier, id_visited_supplier)
            W[id_replaced_supplier, firm_id] = 0 # on enleve ce lien dans le W
            new_eq = get_equilibrium(a, b, z, W, n, wealth, shot_firm, competite_eq)
            new_cost = compute_cost(firm_id, a, b, W, new_eq['X'], new_eq['P'], new_eq['h'])
            if new_cost < min_cost:
                min_cost = new_cost
            W[id_replaced_supplier, firm_id] = Wbar[id_replaced_supplier, firm_id] #apres le test d'un supplier, on remet le lien dans W
        W[id_visited_supplier, firm_id] = 0 # a la fin du test, on remet W comme avant
    return min_cost


def evalute_cost_penalty(firm_id, a, b, z, W, n, wealth, Wbar, supplier_id_list, alternate_supplier_id_list, shot_firm,
                         competite_eq):
    """Compute the difference between the current profit and the max profit reachable by a switch
    """
    eq = get_equilibrium(a, b, z, W, n, wealth, shot_firm, competite_eq)
    current_cost = compute_cost(firm_id, a, b, W, eq['X'], eq['P'], eq['h'])
    min_alternative_cost = evaluate_best_alternative_cost(firm_id, a, b, z, W, n, wealth, Wbar, supplier_id_list,
                                                          alternate_supplier_id_list, shot_firm, competite_eq)
    dif = current_cost - min_alternative_cost
    if dif <= 0:
        # print('Firm '+str(firm_id)+': at the best profit')
        return 0
    else:
        # print('Firm '+str(firm_id)+': not at the best profit')
        return abs(dif)


def get_partial_eq_and_profit(a, b, z, W, n, wealth, eq, W_tm1, firms_within_tiers, id_rewiring_firm, shot_firm,
                              competitive_eq):
    if competitive_eq == "markup":
        return compute_partial_equilibrium_markup_and_profit(a, b, z, W, n, eq, firms_within_tiers, id_rewiring_firm,
                                                        shot_firm)
    else:
        return compute_partial_equilibrium_and_profit(a, b, z, W, n, wealth, eq, W_tm1, firms_within_tiers,
                                                  id_rewiring_firm, shot_firm)


def compute_partial_equilibrium_and_profit(a, b, z, W, n, wealth, eq, firms_within_tiers, id_rewiring_firm,
                                            shot_firm):
    '''Compute equilibrium for selected firms only
    Suppose that the firm doing this calculation knows:
    - prices, production, wage from last time step
    - all parameters of firms_within_tiers
    - the amount of work hired and intermediary demand from last time step of firms_within_tiers
    - the matrix W within firms_within_tiers
    '''
    # reduce the system
    # print(firms_within_tiers)
    W_reduced = W[np.ix_(firms_within_tiers, firms_within_tiers)]
    a_reduced = a[firms_within_tiers]
    b_reduced = b[firms_within_tiers]
    z_reduced = z[firms_within_tiers]

    # First equation: good market balance
    if shot_firm == None:  # we do not use n_reduced here because:
        # 1. the id of the shot_firm correspoonds to the non-reduced system
        # 2. the wealth corresponds to the non-reduced system
        hh_demand = (wealth / n) * np.ones([n, 1])
    else:
        hh_demand = (wealth / (n - 1)) * np.ones([n, 1])
        hh_demand[shot_firm] = 0

    partial_eq = compute_equilibrium_markup(a_reduced, b_reduced, z_reduced, W_reduced, n, shot_firm=None)
    partial_eq["firms_within_tiers"] = firms_within_tiers

    firm_id_reduced = firms_within_tiers.index(id_rewiring_firm)
    mean_cost = eq['P'][firm_id_reduced]

    return partial_eq, mean_cost


def compute_partial_equilibrium_and_profit2(a, b, z, W, n, wealth, eq, W_tm1, firms_within_tiers, id_rewiring_firm, shot_firm):
    '''Compute equilibrium for selected firms only
    Suppose that the firm doing this calculation knows:
    - prices, production, wage from last time step
    - all parameters of firms_within_tiers
    - the amount of work hired and intermediary demand from last time step of firms_within_tiers
    - the matrix W within firms_within_tiers
    '''
    # reduce the system
    #print(firms_within_tiers)
    W_reduced = W[np.ix_(firms_within_tiers, firms_within_tiers)]
    W_tm1_reduced = W_tm1[np.ix_(firms_within_tiers, firms_within_tiers)]
    a_reduced = a[firms_within_tiers]
    b_reduced = b[firms_within_tiers]
    z_reduced = z[firms_within_tiers]
    n_reduced = len(firms_within_tiers)

    # First equation: good market balance
    if shot_firm==None: # we do not use n_reduced here because:
    # 1. the id of the shot_firm correspoonds to the non-reduced system
    # 2. the wealth corresponds to the non-reduced system
        hh_demand = (wealth/n)*np.ones([n,1])
    else:
        hh_demand = (wealth/(n-1))*np.ones([n,1])
        hh_demand[shot_firm] = 0
    hh_demand_reduced = hh_demand[firms_within_tiers]
    ##calculate the intermediary demand per firm from last time step
    inter_demand_last_timestep = [eq['X'][i] * eq['P'][i] - hh_demand[(i, 0)] for i in firms_within_tiers]
    inter_demand_last_timestep_from_reduced = np.matmul(
        (1 - a_reduced) * b_reduced * W_tm1_reduced,
        np.transpose(np.array([eq['X'][firms_within_tiers] * eq['P'][firms_within_tiers],]))
    ).flatten()
    inter_demand_from_outside = inter_demand_last_timestep - inter_demand_last_timestep_from_reduced
    Wtilde_reduced = (1 - a_reduced) * b_reduced * W_reduced
    tV_reduced = np.linalg.solve(
        np.eye(n_reduced) - Wtilde_reduced, 
        hh_demand_reduced + np.transpose(np.array([inter_demand_from_outside,]))
    )
    V_reduced = np.transpose(tV_reduced)
    print((
        "prod:", eq['X'][firms_within_tiers], 
        ", price:", eq['P'][firms_within_tiers], 
        ", old V:", eq['P'][firms_within_tiers] * eq['X'][firms_within_tiers], 
        ", new V:", V_reduced
    ))



    # Second equation: labor market balance
    #firms_not_within_tiers = [id for id in list(range(n)) if id not in firms_within_tiers]
    work_supply_last_timestep = sum([eq['X'][i] * eq['P'][i] / eq['h'] * a[i] * b[i] for i in firms_within_tiers])
    h_reduced = np.sum(a_reduced * b_reduced * V_reduced) / work_supply_last_timestep
    print(("old wage:", eq['h'], "new wage:", h_reduced))
    
    
    # Third equation: optimum production
    deltaW_reduced = W_reduced.sum(axis=0) - 1
    bModif_reduced = b_reduced * (1 + deltaW_reduced * (1 - a_reduced))
    #input_factor_last_timestep = (1 - a) * b * W * eq['P'] * eq['X'] / np.transpose(eq['P'])
    input_factor_last_timestep = np.sum((1 - a) * b * W_tm1 * np.log(np.transpose(np.array([eq['P'],]))), axis=1)[firms_within_tiers]
    input_factor_last_timestep_from_reduced = np.sum((1 - a_reduced) * b_reduced * W_tm1_reduced * np.log(np.transpose(np.array([eq['P'][firms_within_tiers],]))), axis=1)
    input_factor_last_timestep_from_outside = input_factor_last_timestep - input_factor_last_timestep_from_reduced
    # replace 0 by 1 so that it does not create an issue with log
    #input_factor_last_timestep_from_outside[input_factor_last_timestep_from_outside == 0] = 1
    print(("input_factor_last_timestep_from_reduced", input_factor_last_timestep_from_reduced, "input_factor_last_timestep_from_outside", input_factor_last_timestep_from_outside))
    #print('bModif', np.min(bModif), np.max(bModif))
    print((a_reduced * b_reduced * np.log(h_reduced) - np.log(z_reduced) - bModif_reduced * np.log(b_reduced) - (bModif_reduced - 1) * np.log(V_reduced)))
    print((a_reduced * b_reduced * np.log(h_reduced) - np.log(z_reduced) - bModif_reduced * np.log(b_reduced) - (bModif_reduced - 1) * np.log(V_reduced) + np.array([input_factor_last_timestep_from_outside,])))
    '''print(np.transpose(a_reduced * b_reduced * np.log(h_reduced) \
            - np.log(z_reduced) \
            - bModif_reduced * np.log(b_reduced) \
            - (bModif_reduced - 1) * np.log(V_reduced) \
            + np.array([input_factor_last_timestep_from_outside,])
        ))'''
    tlogP_reduced = np.linalg.solve(
        np.transpose(np.eye(n_reduced) - Wtilde_reduced), 
        np.transpose(a_reduced * b_reduced * np.log(h_reduced) \
            - np.log(z_reduced) \
            - bModif_reduced * np.log(b_reduced) \
            - (bModif_reduced - 1) * np.log(V_reduced) \
            + np.array([input_factor_last_timestep_from_outside,])
        )
    )
    logP_reduced = np.transpose(tlogP_reduced)
    print(logP_reduced)
    P_reduced = np.exp(logP_reduced)
    #print(P_reduced)
    X_reduced = V_reduced / P_reduced
    print(("old P:", eq['P'][firms_within_tiers], "new P:", P_reduced))
    print(("old X:", eq['X'][firms_within_tiers], "new X:", X_reduced))

    # try to implement as in text. We don't do "input_factor_last_timestep_from_reduced". We know the last prices of the suppliers...

    # Build Partial Eq
    P_reduced = P_reduced.flatten()
    X_reduced = X_reduced.flatten()
    partial_eq = {
        "X": X_reduced, 
        "P": P_reduced, 
        "h": h_reduced,
        'firms_within_tiers': firms_within_tiers
    }
    # print(h_reduced, eq['h'])
    # print(eq['P'][firms_within_tiers])
    # print(P_reduced)
    #exit()
    # Compute profit
    firm_id_reduced = firms_within_tiers.index(id_rewiring_firm)
    profit = compute_profit(firm_id_reduced, a_reduced, b_reduced, W_reduced, X_reduced, P_reduced, h_reduced)

    return partial_eq, profit




def compute_profit(firm_id, a, b, W, X, P, h):
    '''Compute profit of a firm
    
    Parameters
    ----------
    firm_id: int
        index of the firm
    a: list/array
        share of labor of the firms
    b: list/array
        overall return to scale of the firms
    W: matrix
        input-output matrix
    X: list/array
        production vector
    P: list/array
        price vector
    h: float
        wage
        
    Return
    ------
    float
        profit of the firm
    '''
    labor_need = a[firm_id] * b[firm_id] * P[firm_id] * X[firm_id] / h
    good_needs = (1 - a[firm_id]) * b[firm_id] * P[firm_id] * X[firm_id] * np.transpose(W[:,firm_id]) / P
    profit = P[firm_id] * X[firm_id] - h * labor_need - np.sum(P * good_needs)
    return profit;


def compute_distance_btw_ntw(g1, g2):
    # Compute distance btw ntw. First version, using adjacency matrix
    M1 = np.array(g1.get_adjacency().data)
    M2 = np.array(g2.get_adjacency().data)
    return np.sum(np.sum(M1 * M2)) / np.sqrt(g1.ecount() * g2.ecount())


def compute_distance_btw_ntw2(g1, g2):
    # Compute distance btw ntw. Second version, using edgelist
    EL1 = np.array(g1.get_edgelist())
    EL2 = np.array(g2.get_edgelist())
    nbCommonRowsOf2dArrays(EL1, EL2)
    return nbCommonRowsOf2dArrays(EL1, EL2) / np.sqrt(g1.ecount() * g2.ecount())
    
    
# Compute the number of common rows in 2d arrays
def nbCommonRowsOf2dArrays(A, B):
    nrows, ncols = A.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
           'formats': ncols * [A.dtype]}
    C = np.intersect1d(A.view(dtype), B.view(dtype))
    C = C.view(A.dtype).reshape(-1, ncols)
    nrows, ncols = C.shape
    return nrows



### Hamiltonian

def evaluate_best_alternative_profit(firm_id, a, b, z, W, n, wealth, Wbar, supplier_id_list, alternate_supplier_id_list, shot_firm):
    """Compute the best profit reachable by a switch
    Non myopic version: the firm update the W based on the switch and compute the new network-wide equilibrium
    """
    max_profit = -9999
    for id_visited_supplier in alternate_supplier_id_list[firm_id]:
        W[id_visited_supplier, firm_id] = Wbar[id_visited_supplier, firm_id]
        for id_replaced_supplier in supplier_id_list[firm_id]:
            #print('test', firm_id, id_replaced_supplier, id_visited_supplier)
            W[id_replaced_supplier, firm_id] = 0 # on enleve ce lien dans le W
            new_eq = compute_equilibrium2(a, b, z, W, n, wealth, shot_firm)
            new_profit = compute_profit(firm_id, a, b, W, new_eq['X'], new_eq['P'], new_eq['h'])
            if new_profit>max_profit:
                max_profit = new_profit
            W[id_replaced_supplier, firm_id] = Wbar[id_replaced_supplier, firm_id] #apres le test d'un supplier, on remet le lien dans W
        W[id_visited_supplier, firm_id] = 0 # a la fin du test, on remet W comme avant
    return max_profit


def evalute_penalty(firm_id, a, b, z, W, n, wealth, Wbar, supplier_id_list, alternate_supplier_id_list, shot_firm):
    """Compute the difference between the current profit and the max profit reachable by a switch
    """
    eq = compute_equilibrium2(a, b, z, W, n, wealth, shot_firm)
    current_profit = compute_profit(firm_id, a, b, W, eq['X'], eq['P'], eq['h'])
    max_alternative_profit = evaluate_best_alternative_profit(firm_id, a, b, z, W, n, wealth, Wbar, supplier_id_list, alternate_supplier_id_list, shot_firm)
    dif = max_alternative_profit - current_profit
    if dif<=0:
        #print('Firm '+str(firm_id)+': at the best profit')
        return 0
    else:
        #print('Firm '+str(firm_id)+': not at the best profit')
        return abs(dif)


def compute_score(a, b, z, W, n, wealth, Wbar, supplier_id_list, alternate_supplier_id_list, shot_firm):
    '''Compute a network-level metric. It is the sum of all so-called firm-level penalty.
    A firm's penalty is the difference between the maximum profit reachable with perfect anticipation and the current profit 
    '''
    all_indiv_score = [evalute_penalty(firm_id, a, b, z, W, n, wealth, Wbar, supplier_id_list, alternate_supplier_id_list, shot_firm) for firm_id in range(n)]
    return sum(all_indiv_score)


def draw_random_vector_normal(mean, sd, n, min_val=None, max_val=None):
    vec = np.random.normal(mean, sd, n)
    if min_val or max_val:
        for k in range(len(vec)):
            if min_val:
                if vec[k] < min_val:
                    vec[k]=min_val
            if max_val:
                if vec[k] > max_val:
                    vec[k]=max_val
    return vec


def draw_random_vector_lognormal(mean, sd, n, min_val=None, max_val=None, integer=False):
    if mean == 0:
        return [0 for x in range(n)]

    else:
        mean_lognorm = np.log(mean ** 2 / (mean ** 2 + sd ** 2) ** (1 / 2))
        sd_lognorm = (np.log(1 + sd ** 2 / mean ** 2)) ** (1 / 2)
        vec = np.random.lognormal(mean_lognorm, sd_lognorm, n)
        if min_val or max_val:
            for k in range(len(vec)):
                if min_val:
                    if vec[k] < min_val:
                        vec[k] = min_val
                if max_val:
                    if vec[k] > max_val:
                        vec[k] = max_val
        if integer:
            vec = [int(round(x)) for x in list(vec)]
        return vec


def identify_firms_within_tier(id_firm, g, tier):
    neighboors = g.neighborhood(vertices=id_firm, order=tier, mode='all')
    neighboors.sort()
    return neighboors

