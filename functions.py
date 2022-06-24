def computeEquilibrium(a, b, z, W, n, wealth, shot_firm=None):
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


def computeEquilibrium2(a, b, z, W, n, wealth, shot_firm=None): # with solve instead of inv
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
        print(dif1.any(), dif2.any(), dif3.any(), dif4.any(), dif5.any())
    #eq3 = time.clock()
    #print("eq3", eq3 - eq2)
    #return
    return {"X":X.flatten(), "P":P.flatten(), "h": h}

# test:
#np.sum(a*b*P*X)/h

#np.power(np.full(n,h), a*b) / (z * np.power(b,b) * np.power(V, b-1)) * np.prod(np.power(np.dot(np.transpose(P),np.ones([1,3])), (1-a)*b*W), axis=0)
#P
#z*np.power(b,b) * np.power(P,b) * np.power(X,b) * np.power(np.full(n,h), -a*b) *  np.prod(np.power(np.dot(np.transpose(P),np.ones([1,3])), -(1-a)*b*W), axis=0)
#X
#G = (1-a)*W*b*P/np.transpose(P)*X
#L = a*b*P/h*X
#Profit = P*X - h*L - np.dot(P, G)


def computePartialEquilibriumAndProfit(a, b, z, W, n, wealth, eq, W_tm1, firms_within_tiers, id_rewiring_firm, shot_firm):
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
    print(
        "prod:", eq['X'][firms_within_tiers], 
        ", price:", eq['P'][firms_within_tiers], 
        ", old V:", eq['P'][firms_within_tiers] * eq['X'][firms_within_tiers], 
        ", new V:", V_reduced
    )



    # Second equation: labor market balance
    #firms_not_within_tiers = [id for id in list(range(n)) if id not in firms_within_tiers]
    work_supply_last_timestep = sum([eq['X'][i] * eq['P'][i] / eq['h'] * a[i] * b[i] for i in firms_within_tiers])
    h_reduced = np.sum(a_reduced * b_reduced * V_reduced) / work_supply_last_timestep
    print("old wage:", eq['h'], "new wage:", h_reduced)
    
    
    # Third equation: optimum production
    deltaW_reduced = W_reduced.sum(axis=0) - 1
    bModif_reduced = b_reduced * (1 + deltaW_reduced * (1 - a_reduced))
    #input_factor_last_timestep = (1 - a) * b * W * eq['P'] * eq['X'] / np.transpose(eq['P'])
    input_factor_last_timestep = np.sum((1 - a) * b * W_tm1 * np.log(np.transpose(np.array([eq['P'],]))), axis=1)[firms_within_tiers]
    input_factor_last_timestep_from_reduced = np.sum((1 - a_reduced) * b_reduced * W_tm1_reduced * np.log(np.transpose(np.array([eq['P'][firms_within_tiers],]))), axis=1)
    input_factor_last_timestep_from_outside = input_factor_last_timestep - input_factor_last_timestep_from_reduced
    # replace 0 by 1 so that it does not create an issue with log
    #input_factor_last_timestep_from_outside[input_factor_last_timestep_from_outside == 0] = 1
    print("input_factor_last_timestep_from_reduced", input_factor_last_timestep_from_reduced, "input_factor_last_timestep_from_outside", input_factor_last_timestep_from_outside)
    #print('bModif', np.min(bModif), np.max(bModif))
    print(a_reduced * b_reduced * np.log(h_reduced) - np.log(z_reduced) - bModif_reduced * np.log(b_reduced) - (bModif_reduced - 1) * np.log(V_reduced))
    print(a_reduced * b_reduced * np.log(h_reduced) - np.log(z_reduced) - bModif_reduced * np.log(b_reduced) - (bModif_reduced - 1) * np.log(V_reduced) + np.array([input_factor_last_timestep_from_outside,]))
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
    print("old P:", eq['P'][firms_within_tiers], "new P:", P_reduced)
    print("old X:", eq['X'][firms_within_tiers], "new X:", X_reduced)

    # try to implement as in text. We don't do "input_factor_last_timestep_from_reduced". We know the last prices of the suppliers...
    exit()
    
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
    profit = computeProfit(firm_id_reduced, a_reduced, b_reduced, W_reduced, X_reduced, P_reduced, h_reduced)

    return partial_eq, profit




def computeProfit(firm_id, a, b, W, X, P, h):
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


# Compute distance btw ntw. First version, using adjacency matrix
def computeDistanceBtwNtw(g1, g2):
    M1 = np.array(g1.get_adjacency().data)
    M2 = np.array(g2.get_adjacency().data)
    return np.sum(np.sum(M1 * M2)) / np.sqrt(g1.ecount() * g2.ecount())

#t = time.process_time()
#computeDistanceBtwNtw(g1, g2)
#print(time.process_time() - t)
#t = time.process_time()
#computeDistanceBtwNtw2(g1, g2)
#print(time.process_time() - t)

# Compute distance btw ntw. Second version, using edgelist
def computeDistanceBtwNtw2(g1, g2):
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

def evaluateBestAlternativeProfit(firm_id, a, b, z, W, n, wealth, Wbar, supplier_id_list, alternate_supplier_id_list, shot_firm):
    """Compute the best profit reachable by a switch
    Non myopic version: the firm update the W based on the switch and compute the new network-wide equilibrium
    """
    max_profit = -9999
    for id_visited_supplier in alternate_supplier_id_list[firm_id]:
        W[id_visited_supplier, firm_id] = Wbar[id_visited_supplier, firm_id]
        for id_replaced_supplier in supplier_id_list[firm_id]:
            #print('test', firm_id, id_replaced_supplier, id_visited_supplier)
            W[id_replaced_supplier, firm_id] = 0 # on enleve ce lien dans le W
            new_eq = computeEquilibrium2(a, b, z, W, n, wealth, shot_firm)
            new_profit = computeProfit(firm_id, a, b, W, new_eq['X'], new_eq['P'], new_eq['h'])
            if new_profit>max_profit:
                max_profit = new_profit
            W[id_replaced_supplier, firm_id] = Wbar[id_replaced_supplier, firm_id] #apres le test d'un supplier, on remet le lien dans W
        W[id_visited_supplier, firm_id] = 0 # a la fin du test, on remet W comme avant
    return max_profit


def evalutePenalty(firm_id, a, b, z, W, n, wealth, Wbar, supplier_id_list, alternate_supplier_id_list, shot_firm):
    """Compute the difference between the current profit and the max profit reachable by a switch
    """
    eq = computeEquilibrium2(a, b, z, W, n, wealth, shot_firm)
    current_profit = computeProfit(firm_id, a, b, W, eq['X'], eq['P'], eq['h'])
    max_alternative_profit = evaluateBestAlternativeProfit(firm_id, a, b, z, W, n, wealth, Wbar, supplier_id_list, alternate_supplier_id_list, shot_firm)
    dif = max_alternative_profit - current_profit
    if dif<=0:
        #print('Firm '+str(firm_id)+': at the best profit')
        return 0
    else:
        #print('Firm '+str(firm_id)+': not at the best profit')
        return abs(dif)


def computeScore(a, b, z, W, n, wealth, Wbar, supplier_id_list, alternate_supplier_id_list, shot_firm):
    '''Compute a network-level metric. It is the sum of all so-called firm-level penalty.
    A firm's penalty is the difference between the maximum profit reachable with perfect anticipation and the current profit 
    '''
    all_indiv_score = [evalutePenalty(firm_id, a, b, z, W, n, wealth, Wbar, supplier_id_list, alternate_supplier_id_list, shot_firm) for firm_id in range(n)]
    return sum(all_indiv_score)


def drawRandomVectorNormal(mean, sd, n, min_val=None, max_val=None):
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



def identifyFirmsWithinTier(id_firm, g, tier):
    neighboors = g.neighborhood(vertices=id_firm, order=tier, mode='all')
    neighboors.sort()
    return neighboors
