#================================================================================
# Retrieve parameter name
exp_name = exp_type
# Start counting time
starting_time = datetime.datetime.now()


#================================================================================
# Parameters that need n
wealth = n
nb_extra_suppliers=np.full(n, cc)

# Economic parameters: global return to scale b
# b
#b = np.full(n, 0.9)
eps = 5e-2
min_b = eps
max_b = 1-eps
b = drawRandomVectorNormal(0.9, sigma_b, n, min_b, max_b)
#nb_different = 30
#b[0:nb_different] = b[0:nb_different]+1
print("b: min "+str(min(b))+' max '+str(max(b)))

# Economic parameters: labor share a
eps = 5e-2
min_a = eps
a = np.array([drawRandomVectorNormal(0.5, sigma_a, n, min_a, min((1-eps)/item, 1-eps))[0] for item in list(b)])
#a = drawRandomVectorNormal(0.5, sigma_a, n, min_a, max_a)
print("a: min "+str(min(a))+' max '+str(max(a)))
#a[2] = 0.97

# Economic parameters: Productivity z
#z = np.full(n, 1.0)
min_z = 1e-1
z = drawRandomVectorNormal(1, sigma_z, n, min_val=min_z)
print("z: min "+str(min(z))+' max '+str(max(z)))


#================================================================================
# Create tech network Wbar and initial input-output network W0
## Option 1: Load existing network. needs g0, techgraph, M0, W0, c
if inputed_network: 
    subfolder = 'initial_network'
    g0 = igraph.load(subfolder+'/'+'g0'+'.'+format_graph, format=format_graph)
    tech_graph = igraph.load(subfolder+'/'+'tech_graph'+'.'+format_graph, format=format_graph)
    nb_suppliers = np.array(g0.degree(list(range(n)), mode="in"))
    M0 = np.array(g0.get_adjacency(attribute=None).data)
    Mbar = np.array(tech_graph.get_adjacency(attribute=None).data)
    Wbar = np.array(tech_graph.get_adjacency(attribute="weight", default=0).data)
    W0 = M0 * Wbar
    c = g0.ecount() / g0.vcount()
    #supplier_id_list = np.fromfile(subfolder+'/'+'supplier_id_list', sep=',')
    #alternate_supplier_id_list = np.fromfile(subfolder+'/'+'alternate_supplier_id_list', sep=',')
    supplier_id_list = np.load(subfolder+'/'+'supplier_id_list'+'.npy')
    alternate_supplier_id_list = np.load(subfolder+'/'+'alternate_supplier_id_list'+'.npy')
    if g0.vcount() != n:
        print("Inadequate inputed network: n is", n, "while g0.vcount() is", g0.vcount())
    #log
    if show_time:
        ntw_creation = datetime.datetime.now() - starting_time
        ntw_creation_time = datetime.datetime.now()
        print("Network loaded", ntw_creation)
## Option 2: Generate new graphs
else:
    exec(open('generateNetwork.py').read())
    #log
    if show_time:
        ntw_creation = datetime.datetime.now() - starting_time
        ntw_creation_time = datetime.datetime.now()
        print("Network created", ntw_creation)
    
print("a*b: max "+str(max(a*b)))
print("(1-a)*b*wji: max "+str(((1-a) * b * Wbar).max()))
#print(M0[0:nb_different, 0:nb_different])
#print(sum(sum(M0[0:nb_different, 0:nb_different]*np.transpose(M0[0:nb_different, 0:nb_different]))/2))


# Option to shut down one firm
shot_firm = None
if randomly_shoot_onefirm:
    #alternate_supplier_id_list
    shot_firm = random.randint(0,n-1)
    print("The shot firm is", shot_firm)
    Wbar[:, shot_firm] = 0
    Wbar[shot_firm, :] = 0
    #W0[shot_firm, :] = 0
    #a[shot_firm] = 1
    #b[shot_firm] = 0.00001
    z[shot_firm] = 0.00001



#================================================================================
# Create specific folder to store outputs
if export:
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
    output_folder = current_time + '_' + exp_name + '/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # export file
    file = open(output_folder + "results.txt", "w")
    if export_firm_profits:
        line = "t rewiring_ts utility_ts "  + ' ' \
        + ' '.join('x'+str(i) for i in range(n))\
        + ' '.join('p'+str(i) for i in range(n))\
        + ' '.join('profit'+str(i) for i in range(n))+'\n'
    else:
        line = "t rewiring_ts utility_ts score\n"
    file.write(line)
    if export_who_rewires:
        file_who_rewire = open(output_folder + "who_rewire.txt", "w")
        file_who_rewire.write("round ts who_rewire\n")


# Output parameters
if export:
    global_param_list = pd.DataFrame(data = {"NbRound": [NbRound], "topology": [topology], "n": [n], "c": [c], "cc": [cc], "sigma_w": [sigma_w], "sigma_z": [sigma_z], "sigma_b": [sigma_b]})
    global_param_list.to_csv(output_folder + 'global_param_list.txt', sep = " ", index=False)
    firm_param_list = pd.DataFrame(data = {"a": a, "b": b, "z": z, "nb_suppliers": nb_suppliers, "nb_extra_suppliers": nb_extra_suppliers})
    firm_param_list.to_csv(output_folder + 'firm_param_list.txt', sep = " ", index=False)
    np.array(g0.get_edgelist()).tofile(output_folder + 'M_0.txt', sep = " ")
    
# Output tech network
if export & save_network_on_off:
    np.array(tech_graph.get_edgelist()).tofile(output_folder + 'Mbar_.txt', sep = " ")
    np.array(igraph.EdgeSeq(tech_graph)["weight"]).tofile(output_folder + 'Wbar_edgelist.txt', sep = " ")
    
    
# Option to count the number of unique network
if count_nb_unique_ntw:
    name_Mfiles = ['M_0.txt']
    nb_unique_ntw = 1
# Initialise observables
#rewiring_ts = np.empty(1 + Tfinal) # from 0 to Tfinal. The value at 0 does not count
#rewiring_ts[0] = 0
#utility_ts = np.empty(1 + Tfinal) # from 0 to Tfinal
#rewiring_time = []




#================================================================================
# Compute initial equilibrium
W = W0.copy()
g = g0.copy()
#np.savetxt(output_folder+"W0.txt", W0)
del(W0, M0)
eq = computeEquilibrium2(a, b, z, W, n, wealth, shot_firm)
#print(eq)
utility_ts = -np.sum(np.log(eq['P']))
rewiring_ts = 0
rewiring_ts_last_round = 0
#utility_ts[0] = -np.sum(np.log(eq['P']))
score = computeScore(a, b, z, W, n, wealth, Wbar, supplier_id_list, alternate_supplier_id_list, shot_firm)
min_score = score


# Option to export initial firm prfits
if export:
    if export_firm_profits:
        line = str(0) + ' ' + str(rewiring_ts) + ' ' + str(utility_ts) + ' ' \
        + ' '.join(str(eq['X'][i]) for i in range(n)) + ' '\
        + ' '.join(str(eq['P'][i]) for i in range(n)) + ' '\
        + ' '.join(str(computeProfit(i, a, b, W, eq['X'], eq['P'], eq['h'])) for i in range(n))+'\n'
    else:
        line = str(0) + ' ' + str(rewiring_ts) + ' ' + str(utility_ts) + ' ' + str(score) + "\n"
    file.write(line)


# Counting the running time
if show_time:
    initialization_time = datetime.datetime.now() - ntw_creation_time
    loop_phase1 = 0
    loop_phase2 = 0
    loop_phase3 = 0
    looptime = datetime.datetime.now()



#================================================================================
# Start time loop
t = 0
eq_reached = 0
W_last_round = W.copy()
W_last_2_round = W.copy()
W_last_3_round = W.copy()
W_last_4_round = W.copy()

for r in range(1, NbRound + 1):
    
    # Update the time-changing variable
    rewiring_ts_last_round = rewiring_ts
    W_last_5_round = W_last_4_round.copy()
    W_last_4_round = W_last_3_round.copy()
    W_last_3_round = W_last_2_round.copy()
    W_last_2_round = W_last_round.copy()
    W_last_round = W.copy()
    
    # Update the rewiring order
    rewiring_order = np.random.choice(range(0,n), replace=False, size=n)
    
    # Loop through each firm
    print("\nRound: "+str(r))
    for i in range(n):
        t += 1
        #if (t%1 == 0):
        #    print(t)
        
        # Select one rewiring firm
        id_rewiring_firm = rewiring_order[i]
            
        # Update the delta W
        deltaW = W.sum(axis=0)-1
        bModif = b*(1+deltaW*(1-a))
        #print(max(bModif))
        
        # Compute current profit
        current_profit = computeProfit(id_rewiring_firm, a, b, W, eq['X'], eq['P'], eq['h'])
        potential_profit = current_profit
        #print('t', t, 'candidate firm for rewire', id_rewiring_firm, "score", score, "profit rewiring firm", profit)
    
        # Loop over current and potential suppliers to evaluate the best switch
        rewiring = 0 # flag that is turned to 1 if the firm rewire
        id_supplier_toremove = None # list that store the current supplier to be replaced, if any
        id_supplier_toadd = None # list that store the current supplier to be added, if any
        #print("Rewiring firm: "+str(id_rewiring_firm)+". Current profit: "+str(profit)+". Current score: "+str(score))
        #profit_dic = {}
        #score_dic = {}
        #max_alter_profit = evaluateBestAlternativeProfit(id_rewiring_firm, a, b, z, W, n, wealth, Wbar, supplier_id_list, alternate_supplier_id_list, shot_firm)
        #penalty = evalutePenalty(id_rewiring_firm, a, b, z, W, n, wealth, Wbar, supplier_id_list, alternate_supplier_id_list, shot_firm)
        #print("Firm "+str(id_rewiring_firm)+': '+str(penalty))
        # Save W
        W_last_ts = W.copy()
        
        # Visit one supplier
        for id_visited_supplier in alternate_supplier_id_list[id_rewiring_firm]:
            #profit_dic[id_visited_supplier] = {}
            #score_dic[id_visited_supplier] = {}
            W[id_visited_supplier, id_rewiring_firm] = Wbar[id_visited_supplier, id_rewiring_firm] # put the i/o coef of the technological matrix
        
            if show_time:
                loop_phase1_time = datetime.datetime.now()
                loop_phase1 = loop_phase1 + datetime.datetime.now() - looptime
            
            # And try to remove one of its current supplier
            for id_replaced_supplier in supplier_id_list[id_rewiring_firm]:
                #print('test', id_rewiring_firm, id_replaced_supplier, id_visited_supplier)
                W[id_replaced_supplier, id_rewiring_firm] = 0 # on enleve ce lien dans le W
                # If firms are myopic, they anticipate their new profit based on the current equilibrium
                # they do not take into account the impact of their rewiring on the system
                if myopic and (tier < 0):
                    estimated_new_profit = computeProfit(id_rewiring_firm, a, b, W, eq['X'], eq['P'], eq['h'])
                    # new_eq = computeEquilibrium2(a, b, z, W, n, wealth, shot_firm)
                    # new_profit = computeProfit(id_rewiring_firm, a, b, W, new_eq['X'], new_eq['P'], new_eq['h'])
                    # print("estimated_new_profit:", estimated_new_profit)
                    # print("new_profit:", new_profit)
                    # exit()
                # If firms are myopic but takes into account some tier
                # then need to identify the firms within those tiers
                elif myopic and (tier >= 0):
                    # need to update g igraph object so that we can apply the neighboorhood function
                    g.delete_edges([(id_replaced_supplier, id_rewiring_firm)])
                    g.add_edge(id_visited_supplier, id_rewiring_firm)
                    firms_within_tiers = identifyFirmsWithinTier(id_rewiring_firm, g, tier)
                    # print("firms_within_tiers:", firms_within_tiers)
                    # print(
                    #     'id_rewiring_firm:', id_rewiring_firm,
                    #     'id_replaced_supplier:', id_replaced_supplier,
                    #     'id_visited_supplier:', id_visited_supplier
                    # )
                    g.delete_edges([(id_visited_supplier, id_rewiring_firm)])
                    g.add_edge(id_replaced_supplier, id_rewiring_firm)
                    partial_eq, estimated_new_profit = computePartialEquilibriumAndProfit(a, b, z, W, n, wealth, eq, W_last_ts, firms_within_tiers, id_rewiring_firm, shot_firm)
                    
                    #print(id_rewiring_firm, firms_within_tiers, partial_eq, estimated_new_profit,
                    #computeProfit(id_rewiring_firm, a, b, W, eq['X'], eq['P'], eq['h'])
                    #exit()
                    # new_eq = computeEquilibrium2(a, b, z, W, n, wealth, shot_firm)
                    # new_profit = computeProfit(id_rewiring_firm, a, b, W, new_eq['X'], new_eq['P'], new_eq['h'])
                    # print("estimated_new_profit:", estimated_new_profit)
                    # print("new_profit:", new_profit)
                    # exit()
                # otherwise firms have a perfect anticipation
                # they evaluate their new profit based on the new equilibrium induced by rewiring
                else:
                    new_eq = computeEquilibrium2(a, b, z, W, n, wealth, shot_firm)
                    estimated_new_profit = computeProfit(id_rewiring_firm, a, b, W, new_eq['X'], new_eq['P'], new_eq['h'])
                    #profit_dic[id_visited_supplier][id_replaced_supplier] = new_profit
                
                # if the new profit is larger, then we save this switch
                if estimated_new_profit > potential_profit + epsilon:
                    potential_profit = estimated_new_profit
                    if myopic: # if myopic, the realized full equilibrium is computed after rewiring is done
                        new_eq = computeEquilibrium2(a, b, z, W, n, wealth, shot_firm)
                        #if tier > 0:
                            #print('partial h:', partial_eq['h'], 'new h:', new_eq['h'])
                            #print('partial P:', partial_eq['P'], 'new P:', new_eq['P'][partial_eq['firms_within_tiers']])
                            #print('partial X:', partial_eq['X'], 'new X:', new_eq['X'][partial_eq['firms_within_tiers']])
                            #print('estimated_new_profit', estimated_new_profit, "new_profit", computeProfit(id_rewiring_firm, a, b, W, new_eq['X'], new_eq['P'], new_eq['h']))

                    eq = new_eq
                    rewiring = 1
                    id_supplier_toremove = id_replaced_supplier
                    id_supplier_toadd = id_visited_supplier
                
                # Apres le test d'un supplier à remplacer, on remet le lien dans W
                W[id_replaced_supplier, id_rewiring_firm] = Wbar[id_replaced_supplier, id_rewiring_firm] 
            
            # Apres le test du nouveau supplier, on remet le lien dans W
            W[id_visited_supplier, id_rewiring_firm] = 0 # a la fin du test, on remet W comme avant
        
        #print(sum([computeProfit(id_firm, a, b, W, new_eq['X'], new_eq['P'], new_eq['h']) for id_firm in list(range(n))]))
        #print("My penalty before was: "+str(penalty)+", it is now: "+str(evalutePenalty(id_rewiring_firm, a, b, z, W, eq, n, wealth, Wbar, alternate_supplier_id_list, shot_firm)))
        #print(pd.DataFrame(profit_dic))
            
        #if rewiring == 1:
        #    print("Changed supplier "+str(id_supplier_toremove)+" to supplier "+str(id_supplier_toadd))
        #else:
        #    print("Changed nothing")
        
        if show_time:
            loop_phase2_time = datetime.datetime.now()
            loop_phase2 = loop_phase2 + datetime.datetime.now() - loop_phase1_time
        # at the end of the loop, equilibrium is the selected equilibrium
        
        # After testing all combinations, if there was profit to gain (rewiring == 1
        # we implement the latest switch, we corresponds to the maximal increase in profit
        if rewiring == 1: #si jamais c'est bon, on remplace pour de bon
            print(
                "Firm "+str(id_rewiring_firm),
                "changed supplier "+str(id_supplier_toremove)+" to supplier "+str(id_supplier_toadd),
                "profit increase is "+str(potential_profit - current_profit)
            )
            g.delete_edges([(id_supplier_toremove, id_rewiring_firm)])
            g.add_edge(id_supplier_toadd, id_rewiring_firm)
            W[id_supplier_toadd, id_rewiring_firm] = Wbar[id_supplier_toadd, id_rewiring_firm]
            W[id_supplier_toremove, id_rewiring_firm] = 0
            supplier_id_list[id_rewiring_firm].remove(id_supplier_toremove)
            supplier_id_list[id_rewiring_firm].append(id_supplier_toadd)
            alternate_supplier_id_list[id_rewiring_firm].remove(id_supplier_toadd)
            alternate_supplier_id_list[id_rewiring_firm].append(id_supplier_toremove)
            #if print_score:
            score = computeScore(a, b, z, W, n, wealth, Wbar, supplier_id_list, alternate_supplier_id_list, shot_firm)
            if score < min_score:
                min_score = score
                    #print("min_score:", min_score)
            if export:
                if export_who_rewires:
                    file_who_rewire.write(str(r) + ' ' + str(t) + ' ' + str(id_rewiring_firm)+"\n")
        
        # If there was no profit to gain, but penalty if positive, it means that the firm did not manage to find the good switch
        # if the firm is myopic, it can be normal, otherwise it should not happen
        #else:
        #    if penalty > 0:
        #        profit = computeProfit(id_rewiring_firm, a, b, W, eq['X'], eq['P'], eq['h'])
        #        max_alter_profit = evaluateBestAlternativeProfit(id_rewiring_firm, a, b, z, W, n, wealth, Wbar, supplier_id_list, alternate_supplier_id_list, shot_firm)
        #        print("Firm "+str(id_rewiring_firm)+': current profit '+str(profit)+' best alternative profit '+str(max_alter_profit))
        # print(alternate_supplier_id_list)
        # score = computeScore(a, b, z, W, eq, n, wealth, Wbar, alternate_supplier_id_list, shot_firm)
        # if score < min_score:
        #     min_score = score
                #print("min_score:", min_score)
        #print("Score: "+str(score))
        
        # Update observables
        rewiring_ts = rewiring_ts + rewiring
        utility_ts = -np.sum(np.log(eq['P']))
        
        
        # Export networks
        if export & save_network_on_off & (rewiring == 1):
            #rewiring_time.append(t);
            np.array(g.get_edgelist()).tofile(output_folder + 'M_' + str(t) + '.txt', sep = " ")
            if count_nb_unique_ntw:
                new_ntw = True
                for Mfile in name_Mfiles[::-1]:
                    comp = subprocess.check_output('cmp --silent M_0.txt M_final.txt || echo 1', shell=True)
                    if len(comp) == 0:
                        new_ntw = False
                        break
                if new_ntw:
                    nb_unique_ntw += 1
                name_Mfiles += ['M_' + str(t)]
                        
                
        # Export economic variables
        if export:
            if export_firm_profits:
                line = str(t) + ' ' + str(rewiring_ts) + ' ' + str(utility_ts) + ' ' \
                    + ' '.join(str(eq['X'][i]) for i in range(n)) + ' '\
                    + ' '.join(str(eq['P'][i]) for i in range(n)) + ' '\
        + ' '.join(str(computeProfit(i, a, b, W, eq['X'], eq['P'], eq['h'])) for i in range(n))+'\n'
            else:
                line = str(t) + ' ' + str(rewiring_ts) + ' ' + str(utility_ts) + ' ' + str(score) + "\n"
            file.write(line)
            
            
        # log time
        if show_time:
            loop_phase3 = loop_phase3 + datetime.datetime.now() - loop_phase2_time
            looptime = datetime.datetime.now()
            
        
        
            
            
    # Stop condition
    if apply_stop_condition:
        if rewiring_ts == rewiring_ts_last_round:
            eq_reached = 1
            print("Network equilibrium reached after", r-1, "turns and", rewiring_ts, "rewirings.")
            break
        if np.all(W == W_last_2_round) & np.all(W_last_round == W_last_3_round) \
           & np.all(W_last_2_round == W_last_4_round) & np.all(W_last_3_round == W_last_5_round):
            eq_reached = 2
            print("Limit cycle of period 2 reached after", r-1, "turns and", rewiring_ts, "rewirings.")
            break
        #if np.all(W == W_last_3_round):
        #    eq_reached = 3
        #    print("Limit cycle of period 3 reached after", r-1, "turns.")
        #    break
    
if (t == NbRound*n):
    eq_reached = 0
    print("Network equilibrium not reached")


total_time = (datetime.datetime.now() - starting_time).total_seconds()

#print(W[shot_firm,:])
#print(W[:,shot_firm])

if show_time:
    print("ntw_creation", ntw_creation)
    print("initialization_time", initialization_time)
    print("loop_phase1", loop_phase1)
    print("loop_phase2", loop_phase2)
    print("loop_phase3", loop_phase3)
    print("total_time", total_time)

#if export:
    #utility_ts.tofile(output_folder + "utility_ts.txt", sep = " ")
    #rewiring_ts.tofile(output_folder + "rewiring_ts.txt", sep = " ")

if export_final_network:
    subfolder = 'initial_network'
    g.save(subfolder+'/'+'g0'+'.'+format_graph, format=format_graph)
    tech_graph.save(subfolder+'/'+'tech_graph'+'.'+format_graph, format=format_graph)
    np.save(subfolder+'/'+'supplier_id_list', supplier_id_list)
    np.save(subfolder+'/'+'alternate_supplier_id_list', alternate_supplier_id_list)
    print("Final network data exported in folder:", subfolder)
    
if export:
    file.close()
    file = open(output_folder+"eq_and_time.txt", "w")
    file.write("eq_reached tfinal nb_rounds total_time\n")
    file.write(str(eq_reached) + ' ' + str(t) + ' ' + str(r) + ' ' + str(total_time) + "\n")
    file.close
    np.array(g.get_edgelist()).tofile(output_folder + 'M_final.txt', sep = " ")
    if export_who_rewires:
        file_who_rewire.close()
#print(W, sum(sum(W>0)), sum(sum(Wbar>0)), initial_graph.ecount(), g0.ecount(), g.ecount())


if export_who_rewires:
    file_who_rewire = pd.read_csv(os.path.join(output_folder, "who_rewire.txt"), sep=" ")
    nb_rewiring_per_firm = file_who_rewire['who_rewire'].value_counts()
    nb_rewiring_per_firm_all_firm = pd.DataFrame({"firm":list(range(n)), "nb_rewirings":0})
    nb_rewiring_per_firm_all_firm['nb_rewirings'] = nb_rewiring_per_firm_all_firm['firm'].map(nb_rewiring_per_firm)
    nb_rewiring_per_firm_all_firm['nb_rewirings'] = nb_rewiring_per_firm_all_firm['nb_rewirings'].fillna(0)

    def computeGiniCoef(dist):
        if sum(dist) == 0:
            return 0
        else:
            n = len(dist)
            numerator = 0
            for i in range(n):
                for j in range(n):
                    numerator += abs(dist[i] - dist[j])
            denominator = 2 * n**2 * sum(dist)/n
            return numerator/denominator
        
    gini = computeGiniCoef(nb_rewiring_per_firm_all_firm['nb_rewirings'].tolist())
    
    
if simple_export:
    if simple_export_suffix is not None:
        simple_export_filename = "simple_dyn_results"+str(simple_export_suffix)+".txt"
    else:
        simple_export_filename = "simple_dyn_results.txt"
    with open(simple_export_filename, "a") as myfile:
        myfile.write(
            str(n) + ' ' + str(c) + ' ' + str(cc) + ' ' + str(sigma_w) + ' ' \
            + str(sigma_z) + ' ' + str(sigma_b) + ' ' + topology + ' ' \
            + str(g.diameter()) + ' ' + str(tier) + ' ' \
            + str(eq_reached) + ' ' + str(r) + ' ' + str(rewiring_ts) + ' ' \
            + str(total_time) + ' ' + str(min_score) + "\n"
        )

if export_initntw_experiment:
    simple_export_filename = "tmp/initntw_experiment.txt"
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
    with open(simple_export_filename, "a") as myfile:
        myfile.write(str(n)+' '+str(c)+' '+str(cc)+' '+str(sigma_w)+' '+topology+' '+str(eq_reached)+' '+str(r)+' '+str(total_time)+' '+str(current_time)+"\n")
    np.array(g.get_edgelist()).tofile('tmp/Mf_'+current_time+'.txt', sep = " ")
    if inputed_network == False: #if it is a new initial network save the initial network
        np.array(g0.get_edgelist()).tofile('tmp/M0_'+current_time+'.txt', sep = " ")


if simple_export_with_giniNbNet:
    if simple_export_suffix is not None:
        simple_export_filename = "simple_dyn_results"+str(simple_export_suffix)+".txt"
    else:
        simple_export_filename = "simple_dyn_results.txt"
    with open(simple_export_filename, "a") as myfile:
        myfile.write(
            str(n) + ' ' + str(c) + ' ' + str(cc) + ' ' + str(sigma_w) + ' ' \
            + str(sigma_z) + ' ' + str(sigma_b) + ' ' + topology + ' ' + str(tier) \
            + ' ' + str(eq_reached) + ' ' + str(r) + ' ' + str(rewiring_ts) + ' ' \
            + str(total_time) + ' ' + str(min_score) + ' ' \
            + str(gini) + ' ' + str(nb_unique_ntw) + "\n"
        )

if simple_export_with_gini:
    if simple_export_suffix is not None:
        simple_export_filename = "simple_dyn_results"+str(simple_export_suffix)+".txt"
    else:
        simple_export_filename = "simple_dyn_results.txt"
    with open(simple_export_filename, "a") as myfile:
        myfile.write(
            str(n) + ' ' + str(c) + ' ' + str(cc) + ' ' + str(sigma_w) + ' ' \
            + str(sigma_z) + ' ' + str(sigma_b) + ' ' + topology + ' ' + str(tier) \
            + ' ' + str(eq_reached) + ' ' + str(r) + ' ' + str(rewiring_ts) + ' ' \
            + str(total_time) + ' ' + str(min_score) + ' ' \
            + str(gini) + ' ' + "\n"
        )
        
del(g, tech_graph, W, Mbar, Wbar, nb_suppliers, supplier_id_list, alternate_supplier_id_list, nb_extra_suppliers)
del(total_time, rewiring_ts, utility_ts)
del(a,b,z,wealth)


if save_network_on_off and compute_distance_matrix:
    import re
    
    def edgevectorFromFile(filename):
        with open(filename, 'r') as file:
            edgevector =  file.read()
        edgevector = str(edgevector).split(' ')
        return [edgevector[i]+' '+edgevector[i+1] for i in range(len(edgevector)) if i % 2 == 0]
        
        
    def getAllEdgevectors(folder):
        all_M_filenames = os.listdir(folder)#, pattern = "M_[0-9*]", full.names = T)
        all_M_filenames = [item for item in all_M_filenames if re.match("M_(\d*).txt", str(item))]
        all_M_filenames = {int(re.match("M_(\d*).txt", str(item)).group(1)): item for item in all_M_filenames}
        edgevectors = {ts: edgevectorFromFile(os.path.join(folder, filename)) for ts, filename in all_M_filenames.items()}
        return edgevectors
    
    def computeUnscaledDistance(edgevector1, edgevector2):
        return len(set(edgevector1) - set(edgevector2)) + len(set(edgevector2) - set(edgevector1))
    
    def computeDistanceMatrix(folder):
        edgevectors = getAllEdgevectors(folder)
        nb_networks = len(edgevectors)
        print("Computing distance between", nb_networks, "networks.")
        res = np.zeros((nb_networks, nb_networks), dtype=int)
        for i in range(nb_networks):
            for j in range(i+1, nb_networks):
                res[i,j] = computeUnscaledDistance(edgevectors.values()[i], edgevectors.values()[j])
        res = pd.DataFrame(res, index=edgevectors.keys(), columns=edgevectors.keys())
        return res
        
    res = computeDistanceMatrix(output_folder)
    res.to_csv(os.path.join(output_folder, "distanceMatrix.csv"))
    
    os.system('cd '+output_folder+'; rm M*.txt;')
    
if simple_export_with_giniNbNet:
    os.system('cd '+output_folder+'; rm M*.txt;')


