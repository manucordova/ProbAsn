####################################################################################################
###                                                                                              ###
###                          Functions for probabilistic assignment                              ###
###                               Author: Manuel Cordova (EPFL)                                  ###
###                                Last modified: 20.10.2020                                     ###
###                                                                                              ###
####################################################################################################

# Import libraries
import numpy as np
import scipy.optimize as op
import random
import sys
import itertools as it
import tqdm.auto as tqdm
import tqdm.contrib.itertools as it2
import time
import re
import copy

# Import local libraries
import sim



def get_centers_1D(x, ys):
    """
    Obtain the maximum of each distribution
    
    Inputs: - x         Array of shifts
            - ys        Probability density functions
    
    Output: - centers   Maximum of the distributions
    """
    
    # Initialize array of centers
    centers = []
    
    # Get the center of each distribution (withing the set x values)
    for y in ys:
        centers.append(x[np.where(y == np.max(y))[0][0]])
    
    return np.array(centers)



def get_centers_2D(X, Y, Zs):
    """
    Obtain the maximum of each distribution
    
    Inputs: - x         Array of shifts
            - ys        Probability density functions
    
    Output: - centers   Maximum of the distributions
    """
    
    #Initialize array of centers
    centers = []
    
    # Get the center of each distribution (within the set X and Y values)
    for Z in Zs:
        centers.append([X[np.where(Z == np.max(Z))[0],np.where(Z == np.max(Z))[1]][0],
                        Y[np.where(Z == np.max(Z))[0], np.where(Z == np.max(Z))[1]][0]])
    
    return np.array(centers)



def compute_scores_1D(exp, stat, errs, scal, centers, norm=None, Nmax=None, seed=None, acc=None, N=101):
    """
    Compute the scores for every possible assignment. Centers should be the points where each distribution is maximum,
    so that the weighting is correct. If the variable "acc" is set to None, the probability density at the experimental
    shift yields the score. If "acc" is set to a value, the probability between the experimental shift e - acc and
    e + acc (computed as the numerical integral) is used as the score. All scores are normalized 
    
    Inputs: - exp       List of experimental shifts
            - stat      List of shifts in each distribution
            - errs      List of errors in each distribution
            - scal      Scaling parameters for the conversion from shielding to shift
            - centers   List of centers of the distributions
            - norm      Whether or not the probability density functions should be normalized
            - Nmax      Maximum number of shifts to select to construct the distribution
            - seed      Seed for random selection of shifts
            - acc       Accuracy of the shifts
            - N         Number of points in each shift if an accuracy is set
    
    Output: - scores    Matrix of scores for all possible assignments
    """
    
    # Initialize array of scores
    scores = []
    
    # If no accuracy is set, take the shifts as elements of the array x
    if acc is None:
        x = np.array(exp)
    
    # Otherwise, append the array of N element between e - acc and e + acc to x, for each shift e
    else:
        x = []
        for e in exp:
            x.extend(list(np.linspace(e-acc, e+acc, N)))
        x = np.array(x)
    
    # Loop over all distributions
    for s, er, c in zip(stat, errs, centers):
        
        # Append the center to the array x (for normalization)
        if acc is None:
            x2 = np.append(x, c*scal[0]+scal[1])
        else:
            x2 = np.append(x, np.linspace(c*scal[0]+scal[1]-acc, c*scal[0]+scal[1]+acc, N), axis=0)
        
        # Compute the values of the distribution on the array x
        if Nmax:
            y = sim.gausses(x2, s*scal[0]+scal[1], er, norm=norm, Nmax=Nmax, seed=seed)
        else:
            y = sim.gausses(x2, s*scal[0]+scal[1], er, norm=norm, Nmax=len(s))
        
        # If an accuracy is set, get the integral
        if acc is not None:
            y2 = []
            for i in range(len(exp)+1):
                y2.append(np.trapz(y[i*N:(i+1)*N], x=x2[i*N:(i+1)*N]))
            y = np.array(y2)
        
        # Normalize the scores of this distribution (such that at the maximum, the score is 1)
        y /= y[-1]
        
        # Append the scores of this distribution
        scores.append(y[:-1]/np.sum(y[:-1]))
    
    return np.array(scores)



def compute_scores_2D(exp, stat, errs, scal_x, scal_y, centers, norm=None, Nmax=None, seed=None, acc=None, N=101):
    """
    Compute the scores for every possible assignment. Centers should be the points where each 2D distribution is maximum,
    so that the weighting is correct.
    
    Inputs: - exp       List of experimental shifts
            - stat      List of shifts in each distribution
            - errs      List of errors in each distribution
            - scal      Scaling parameters for the conversion from shielding to shift
            - centers   List of centers of the distributions
            - norm      Whether or not the probability density functions should be normalized
            - Nmax      Maximum number of shifts to select to construct the distribution
            - seed      Seed for random selection of shifts
            - acc       Accuracy of the shifts (array of the accuracy along x, and along y)
            - N         Number of points in each shift (along each axis) if an accuracy is set
    
    Output: - scores    Matrix of scores for all possible assignments
    """
    
    # Initialize matrix of scores
    scores = []
    
    # If no accuracy is set, set x- and y-axes as the experimental shifts
    if acc is None:
        x = np.array(exp)[:,0]
        y = np.array(exp)[:,1]
    
    # Otherwise, append the array of N element between e - acc and e + acc to x, for each shift e. Same for y.
    else:
        x = []
        y = []
        for e in exp:
            x.extend(list(np.linspace(e[0]-acc[0], e[0]+acc[0], N)))
            y.extend(list(np.linspace(e[1]-acc[1], e[1]+acc[1], N)))
        x = np.array(x)
        y = np.array(y)
    
    # Loop over all distributions
    for j, (s, er, c) in enumerate(zip(stat, errs, centers)):
        
        # Append the top of the distribution to the array of shifts
        if acc is None:
            x2 = np.append(x, c[0]*scal_x[0]+scal_x[1])
            y2 = np.append(y, c[1]*scal_y[0]+scal_y[1])
        else:
            x2 = np.append(x, np.linspace(c[0]*scal_x[0]+scal_x[1]-acc[0],
                                          c[0]*scal_x[0]+scal_x[1]+acc[0], N), axis=0)
            y2 = np.append(y, np.linspace(c[1]*scal_y[0]+scal_y[1]-acc[1],
                                          c[1]*scal_y[0]+scal_y[1]+acc[1], N), axis=0)
        
        # If an accuracy is set, get the integral
        if acc is not None:
            Z2 = []
            for i in range(len(exp)):
                
                x3 = np.append(x2[i*N:(i+1)*N], x2[-N:])
                y3 = np.append(y2[i*N:(i+1)*N], y2[-N:])
                
                # Compute the values of the distribution on the grid [x, y]
                # If a maximum number of shifts to use is set, get the distribution with this number of shifts
                if Nmax:
                    X, Y, Z = sim.gausses_2D(x3, y3, s[:,0]*scal_x[0]+scal_x[1], s[:,1]*scal_y[0]+scal_y[1], er[:,0], er[:,1], Nmax=Nmax, seed=seed)
                # Otherwise, use all shifts in the distribution
                else:
                    X, Y, Z = sim.gausses_2D(x3, y3, s[:,0]*scal_x[0]+scal_x[1], s[:,1]*scal_y[0]+scal_y[1], er[:,0], er[:,1], Nmax=len(s[:,0]))
                
                Z2.append(np.trapz(np.trapz(Z[:N,:N], x=y3[:N]), x=x3[:N]))
            Z2.append(np.trapz(np.trapz(Z[N:,N:], x=y3[N:]), x=x3[N:]))
            Z = np.array(Z2)
        
        else:
        
            # Compute the values of the distribution on the grid [x, y]
            # If a maximum number of shifts to use is set, get the distribution with this number of shifts
            if Nmax:
                X, Y, Z = sim.gausses_2D(x2, y2, s[:,0]*scal_x[0]+scal_x[1], s[:,1]*scal_y[0]+scal_y[1], er[:,0], er[:,1], Nmax=Nmax, seed=seed)
            # Otherwise, use all shifts in the distribution
            else:
                X, Y, Z = sim.gausses_2D(x2, y2, s[:,0]*scal_x[0]+scal_x[1], s[:,1]*scal_y[0]+scal_y[1], er[:,0], er[:,1], Nmax=len(s[:,0]))
                
            Z = np.copy(np.diag(Z))
            
        # Normalize the scores of this distribution (such that at the maximum, the score is 1)
        Z /= Z[-1]
        
        print("Distribution {}/{} computed.".format(j+1, len(centers)))
        
        # Append the scores of this distribution
        scores.append(Z[:-1]/np.sum(Z[:-1]))
    
    return np.array(scores)



def increase_threshold(thresh, thresh_inc):
    """
    Increase the threshold based on a string
    
    Inputs: - thresh        Initial threshold
            - thresh_inc    How the threshold should be increased. "xN" to multiply it by N, "+N" to increase it by N
            
    Output: - thresh        Increased threshold
    """
    
    if thresh_inc.startswith("x"):
        return thresh * float(thresh_inc.split("x")[1])
    
    elif thresh_inc.startswith("+"):
        return thresh + float(thresh_inc.split("+")[1])

    raise ValueError("Unkown update of the threshold: {}".format(thresh_inc))



def check_valid_assignment(possible_assignments, labels, n_exp):
    """
    Check if a valid assignment can be generated, i.e. if all shifts are assigned to at least one distribution,
    and if all distributions are assigned to one (and only one) shift.
    
    Inputs: - possible_assignments      List of possible assignments for each distribution
            - labels                    Labels of the distributions
            - n_exp                     Number of experimental shifts

    Output: - True/False                Whether a valid assignment can be generated
    """
    
    tmp_asn = possible_assignments.copy()
    for i, l in enumerate(labels):
        if "/" in l:
            n = l.count("/")
            for _ in range(n):
                tmp_asn.append(tmp_asn[i])

    # Check if any of the possible assignments is valid (if every shift can be attributed to at least one distribution)
    for p in it.product(*tmp_asn):
        if len(np.unique(list(p))) == n_exp:
            return True
    
    return False



def generate_valid_asns(possible_asns, scores, n_dist, n_exp, ls, es, equiv, valid_asns=[], already_linked=[], rank=0, Nmax=None, Nmax_rank=0, max_excess=None, disp_rank=-1, t_start=None):
    """
    Recursively generate valid assignments given possible assignments for each nucleus.
    
    Inputs: - possible_asns     Possible assignments for each nucleus/distribution
            - scores            Matrix of scores for individual assignments
            - n_dist            Number of distributions
            - n_exp             Number of experiments
            - ls                Labels of the nuclei/distributions
            - es                Experimental shifts
            - equiv             Equivalent nuclei/distributions
            - valid_asns        Already found valid assignments
            - already_linked    Already assigned nuclei/distributions
            - rank              Assignment rank
            - Nmax              Maximum number of assignments to consider
            - Nmax_rank         Rank from which to start reducing the number of assignments
            - max_excess        Maximum excess for assignment (defined as the maximum number of nuclei/distributions assigned to a single experimental shift minus one)
            - disp_rank         Rank up to which progress should be displayed
            - t_start           Starting time
            
    Output: - valid_asns        List of valid assignments generated
    """
    
    # If all distributions are already assigned, append the assignment found
    if rank == n_dist:
        
        # Sort the equivalent parts
        sorted_asgn = already_linked.copy()
        already_eq = []
        for eq in equiv:
            if len(eq) > 1 and eq[0] not in already_eq:
                tmp = sorted([already_linked[i] for i in eq])
                for i, e in enumerate(eq):
                    sorted_asgn[e] = tmp[i]
                already_eq.extend(eq)
        
        # Return the list of already generated valid assignments
        valid_asns.append(sorted_asgn)
        return valid_asns
    
    # Get the number of assignments to consider for this nucleus/distribution
    rank_len = len(possible_asns[rank])
    if Nmax is not None and rank >= Nmax_rank:
        rank_len = min(Nmax, rank_len)
    
    # Get the intermediate scores
    these_scores = []
    for a in possible_asns[rank]:
        # Generate the assignment
        tmp_asn = already_linked.copy()
        tmp_asn.append(a)
        
        # Check if the assignment is valid so far (get the excess and maximum individual excess)
        excess = 0
        ind_excess = 0
        for j in np.unique(tmp_asn):
            excess += max(tmp_asn.count(j) - 1, 0)
            ind_excess = max(ind_excess, tmp_asn.count(j) - 1)
            
        # Check that same nuclei are assigned to the same shift (important for 2D simulated experiments)
        
        for i, a1 in enumerate(tmp_asn):
            for j, a2 in enumerate(tmp_asn):
                if i < j:
                    for k in range(len(ls[i])):
                        if ls[i,k] == ls[j,k] and es[a1,k] != es[a2,k]:
                            excess = n_dist - n_exp + 1
                            break
                if excess > n_dist - n_exp:
                    break
            if excess > n_dist - n_exp:
                break

        # If the assignment is valid so far, try to assign the next distribution
        if excess > n_dist - n_exp:
            score = 0.
        else:
            
            if max_excess is not None and ind_excess > max_excess:
                score = 0.
            
            else:
                # Get the corresponding score
                score = 1.
                for i, a in enumerate(tmp_asn):
                    score *= scores[i, a]
                
        these_scores.append(score)
    
    # Sort the intermediate scores by decreasing value
    score_inds = np.argsort(these_scores)[::-1]
    
    # Remove instances where the score is zero
    score_inds = score_inds[:np.count_nonzero(these_scores)]
    
    # If Nmax is set, get the Nmax highest scores
    if Nmax is not None and rank >= Nmax_rank:
        score_inds = score_inds[:Nmax]
    
    asn_num = 0
    for i, a in enumerate(possible_asns[rank]):
        if i in score_inds:
            asn_num += 1
            
            if rank <= disp_rank:
                space = ""
                for _ in range(rank):
                    space += " "
                t_stop = time.time()
                if t_start is not None:
                    print(space + "Rank {}, {}/{}, {} valid assignments so far. Time elapsed: {:.2f} s".format(rank, asn_num, len(score_inds), len(valid_asns), t_stop - t_start))
                else:
                    print(space + "Rank {}, {}/{}, {} valid assignments so far".format(rank, asn_num, len(score_inds), len(valid_asns)))

            # Try assigning distribution "rank" to experiment "a"
            tmp_asn = already_linked.copy()
            tmp_asn.append(a)

            # Check if the assignment is valid so far
            excess = 0
            ind_excess = 0
            for j in np.unique(tmp_asn):
                excess += max(tmp_asn.count(j) - 1, 0)
                ind_excess = max(ind_excess, tmp_asn.count(j) - 1)
            
            # Check that same nuclei are assigned to the same shift (important for 2D simulated experiments)
            for i, a1 in enumerate(tmp_asn):
                for j, a2 in enumerate(tmp_asn):
                    if i < j:
                        for k in range(len(ls[i])):
                            # If two instances of the same nucleus (same label) is not associated with the same experimental shift, the assignment is invalid
                            if ls[i,k] == ls[j,k] and es[a1,k] != es[a2,k]:
                                excess = n_dist - n_exp + 1
                                break
                    if excess > n_dist - n_exp:
                        break
                if excess > n_dist - n_exp:
                    break

            # If the assignment is valid so far, try to assign the next distribution
            if excess <= n_dist - n_exp and (max_excess is None or ind_excess <= max_excess):
                valid_asns = generate_valid_asns(possible_asns, scores, n_dist, n_exp, ls, es, equiv, valid_asns=valid_asns, already_linked=tmp_asn, rank=rank+1, Nmax=Nmax, Nmax_rank=Nmax_rank, max_excess=max_excess, disp_rank=disp_rank, t_start=t_start)
    
    return valid_asns



def get_possible_assignments(scores, labels, exp, thresh=10., thresh_increase="x2", check_valid=False, nMax=-1):
    """
    Get all possible assignments for each probability distribution, given that if an individual probability of
    assigning the distribution to a shift is more than "thresh" times higher than the probability of assigning it
    to another shift, the latter possibility is discarded.
    
    Inputs: - scores                Array of individual scores
            - labels                List of labels of the distributions
            - exp                   List of experimental shifts
            - thresh                Threshold to discard a score
            - thresh_increase       How the threshold should be increased. "xN" to multiply it by N, "+N" to increase it by N
            - check_valid           Whether or not we should check if a valid assignment can be generated
            - nMax                  Maximum number of assignments for a given distribution
    
    Output: - possible_assignments  Lists of possible assignments for each distribution
    """
    
    if type(thresh) == str and thresh == "inf":
        possible_assignments = []
        for i in range(len(scores)):
            possible_assignments.append(list(range(len(scores[i]))))
        
        return possible_assignments, thresh
    
    valid = False
    while not valid:
    
        consistent = False
        while not consistent:
            cleaned_scores = np.copy(scores)
            for i in range(len(cleaned_scores)):
                m = np.max(cleaned_scores[i])
                # Discard assignments that have an individual score lower than 1/thresh times the maximum score
                for j in range(len(cleaned_scores[0])):
                    if cleaned_scores[i, j] < m / thresh:
                        cleaned_scores[i, j] = 0

            consistent = True
            # Check that each distribution is attributed to at least one experiment
            for i in range(len(cleaned_scores)):
                if np.sum(cleaned_scores[i]) == 0:
                    consistent = False
                    thresh = increase_threshold(thresh, thresh_increase)
                    break

            if consistent:
                # Check that each experiment is attributed to at least one distribution
                for j in range(len(cleaned_scores[0])):
                    if np.sum(cleaned_scores[:, j]) == 0:
                        consistent = False
                        thresh = increase_threshold(thresh, thresh_increase)
                        break

        # Get the possible assignments
        possible_assignments = []
        for i in range(len(cleaned_scores)):
            possible_assignments.append(list(np.where(cleaned_scores[i] > 0)[0]))
    
        # Check that the possible individual assignments are able to generate a valid assignment (with a score > 0)
        if check_valid:
            valid = check_valid_assignment(possible_assignments, labels, cleaned_scores.shape[1])
        else:
            valid = True
        
        if not valid:
            thresh = increase_threshold(thresh, thresh_increase)

    print("Scores cleaned up, threshold set to {}".format(thresh))
    
    # Clean up assignments, i.e. if only one nucleus/distribution can be assigned to a given shift, then it must be assigned to that shift
    change = True
    while change:
        change = False
        # Loop over all distributions with only one equivalent
        for i, a in enumerate(possible_assignments):
            if len(a) > 1 and "/" not in labels[i]:
                # Loop over all possible assignments
                for ai in a:
                    found = False
                    for j, a2 in enumerate(possible_assignments):
                        if i != j and ai in a2:
                            found = True
                            break
                    # If the specific assignment is not found anywhere else, it can only be assigned to the corresponding distribution
                    if not found:
                        possible_assignments[i] = [ai]
                        change = True
                        break
            if change:
                break
    
    if nMax > 0:
        backup_assignments = possible_assignments.copy()
        n_valid = False
        
        while not n_valid:
            possible_assignments = backup_assignments.copy()
            # Only select the nMax most probable assignments for each distribution

            for i, a in enumerate(possible_assignments):
                if len(a) > nMax:
                    these_scores = [cleaned_scores[i, j] for j in a]
                    sorted_asn_inds = np.argsort(these_scores)[::-1]
                    possible_assignments[i] = [a[j] for j in sorted_asn_inds[:nMax]]

            unique_assignments = []
            for a in possible_assignments:
                for ia in a:
                    if ia not in unique_assignments:
                        unique_assignments.append(ia)
            
            if len(unique_assignments) < len(exp):
                n_valid = False
            else:
                n_valid = True
                    
            if check_valid and n_valid:
                n_valid = check_valid_assignment(possible_assignments, labels, cleaned_scores.shape[1])

            if not n_valid:
                nMax += 1
    
        print("Maximum number of assignments set to {}".format(nMax))
    
    # If 2D assignments
    if "-" in labels[0]:
        complete_asns = []
        for i, (l, a) in enumerate(zip(labels, possible_assignments)):
            complete_asns.append(a)
            if len(l.split("/")) > 1:
                ls = [tmp.split("-") for tmp in l.split("/")]
                
                for i1, l1 in enumerate(ls):
                    for i2, l2 in enumerate(ls):
                        if i2 > i1:
                            for k, (l1k, l2k) in enumerate(zip(l1, l2)):
                                if l1k == l2k:
                                    for ai in complete_asns[i]:
                                        match = exp[ai].split("\\")[k]
                                        for j, e in enumerate(exp):
                                            if match in e.split("\\") and e not in [exp[aj] for aj in complete_asns[i]]:
                                                complete_asns[i].append(j)
    
        possible_assignments = []
        for a in complete_asns:
            possible_assignments.append(list(np.sort(a)))
        
    return possible_assignments, thresh



def reduce_possible_assignments(possible_assignments, labels, n_exp):
    """
    Reduce the number of possible assignments
    
    Inputs: - possible_assignments      List of possible assignments for each distribution
            - labels                    Labels of the distributions
            - n_exp                     Number of experimental shifts
            
    Output: - reduced_assignments       Reduced list of possible assignments
    """
    
    tmp_assignments = []
    new_labels = []
    for asgn, l in zip(possible_assignments, labels):
        tmp_l = l.split("/")
        
        for l in tmp_l:
            tmp_assignments.append(asgn)
            new_labels.append(l)
    
    change = True
    while change:
        change = False
        
        # Loop over all possible assignments
        for i, a in enumerate(tmp_assignments):
            
            print(i, a)
            
            for ai in a:
                tmp_asn = tmp_assignments.copy()
                tmp_asn[i] = [ai]
                
                # Check if a valid assignment can be made
                
                valid = check_valid_assignment(tmp_asn, new_labels, n_exp)
                
                if not valid:
                    change = True
                    tmp_assignments[i].remove(ai)
                    break
    
    reduced_assignments = []
    for l in labels:
        if "/" in l:
            tmp_l = l.split("/")
            tmp_asn = []
            for l2 in tmp_l:
                i = new_labels.index(l2)
                tmp_asn.extend(tmp_assignments[i])
                
            reduced_assignments.append(list(np.unique(tmp_asn)))
        else:
            i = new_labels.index(l)
            reduced_assignments.append(tmp_assignments[i])
        
    
    return reduced_assignments



def get_assignment_pool(possible_asn, already_assigned):
    """
    Get an assignment pool
    
    Inputs:     - possible_asn          List of possible assignments for each distribution
                - already_assigned      List of already assigned distributions
    
    Outputs:    - dist_pool             Pool of distributions
                - shift_pool            Shifts in the pool of distributions
    """
    
    # Initialize arrays of possible shifts and distributions
    shift_pool = []
    dist_pool = []
    
    # Loop over all distribution
    for i, a in enumerate(possible_asn):
        if i not in already_assigned:
            # Get the possible shifts for the first not already assigned distribution
            shift_pool.extend(a)
            dist_pool.append(i)
            break
    
    change = True
    while change:
        change = False
        # Loop over all distributions
        for i, a in enumerate(possible_asn):
            if i not in dist_pool and i not in already_assigned:
                for ai in a:
                    # If any of the possible shifts for this distribution is within the pool, add this distribution
                    # to the possible assignment pool
                    if ai in shift_pool:
                        dist_pool.append(i)
                        shift_pool.extend([aj for aj in a if aj not in shift_pool])
                        change = True
                        break
    
    return dist_pool, shift_pool



def get_probabilistic_assignment(scores, possible_assignments, exp, labels, Nmax=None, Nmax_rank=0, disp_rank=-1, max_excess=None, order="default", pool_inds=None):
    """
    Get the probabilistic assignment given individual possible assignments and scores
    
    Inputs:     - scores                    Matrix of scores
                - possible_assignments      List of possible assignments for each nucleus/distribution
                - exp                       List of experimental shifts
                - labels                    List of labels of the nuclei/distributions
                - Nmax                      Maximum number of assignments to consider for the assignment generation
                - Nmax_rank                 Rank from which to select only the Nmax most probable assignments
                - disp_rank                 Rank up to which to display progress
                - max_excess                Maximum number of nuclei/distributions that can be assigned to a single shift
                - order                     Order in which to make the assignment ("default", "increase", "decrease", "span", "randN")
    
    Outputs:    - all_dist_pools            Nuclei/distributions in each pool
                - all_shift_pools           Shifts in each pool
                - all_cleaned_asns          Possible assignments for each pool
                - all_cleaned_scores        Scores of the possible assignments for each pool
                - new_labels                Individual labels for each nucleus/distribution
                - all_equivs                List of equivalent nuclei/distributions
    """
    
    # Separate topologically equivalent nuclei
    tmp_asns = []
    tmp_scores = []
    new_labels = []
    label_inds = []
    equiv_inds = []
    for asgn, l, s in zip(possible_assignments, labels, scores):
        tmp_l = l.split("/")
        tmp_eq = []
        for l in tmp_l:
            tmp_asns.append(asgn)
            new_labels.append(l)
            tmp_scores.append(s)
            tmp_eq.append(new_labels.index(l))
            tmp_i = []
            
            for l2 in l.split("-"):
                tmp_i.append(int(re.findall("\d+", l2)[0]))
        
            label_inds.append(tmp_i)
            
        for l in tmp_l:
            equiv_inds.append(tmp_eq)
    
    tmp_scores = np.array(tmp_scores)
    
    label_inds = np.array(label_inds)
    
    # Get the numerical experimental shifts
    exp_nums = []
    for e in exp:
        tmp_exp = []
        for e2 in e.split("\\"):
            tmp_exp.append(e2)
        exp_nums.append(tmp_exp)
    
    exp_nums = np.array(exp_nums)
            
    # Initialize the array of assigned nuclei and shifts
    already_assigned = []
        
    all_cleaned_asns = []
    all_cleaned_scores = []
    all_dist_pools = []
    all_shift_pools = []
    all_equivs = []
        
    pool_ind = 0
        
    # Loop until all distributions are assigned
    while len(already_assigned) < len(new_labels):
        # Get the assignment pool
        dist_pool, shift_pool = get_assignment_pool(tmp_asns, already_assigned)
        
        if pool_inds is None or pool_ind in pool_inds:
        
            all_dist_pools.append(dist_pool)
            all_shift_pools.append(shift_pool)
            
            tmp_pp = "Ambiguity found between "
                
            already_labs = []
            for l in [new_labels[i] for i in dist_pool]:
                tmp_pp += "{}, ".format(l)
            
            tmp_pp = tmp_pp[:-2]
            print(tmp_pp)
            
            tmp_asn = []
            for i in dist_pool:
                tmp_asn.append(tmp_asns[i])
                
            print("Corresponding shifts: {}".format(", ".join([exp[i] for i in shift_pool])))
            
            print("Mapping {} nuclei on {} shifts".format(len(dist_pool), len(shift_pool)))
                
            start = time.time()
            print("Generating all possible assignments...")
            
            # Generate the valid assignments
            if order == "default":
                pool_label_inds = label_inds[dist_pool]
                
                # Get the equivalent nuclei in the current distribution pool
                these_eqs = []
                for i in dist_pool:
                    tmp_eq = []
                    for eq in equiv_inds[i]:
                        tmp_eq.append(dist_pool.index(eq))
                    these_eqs.append(tmp_eq)
                    
                pool_asns = generate_valid_asns(tmp_asn, tmp_scores[dist_pool], len(dist_pool), len(shift_pool), pool_label_inds, exp_nums, these_eqs, valid_asns=[], already_linked=[], rank=0, Nmax=Nmax, Nmax_rank=Nmax_rank, max_excess=max_excess, disp_rank=disp_rank, t_start=start)
                
            elif order == "increase":
                
                # Reorder the distributions by increasing number of possible assignments
                asn_len = []
                for a in tmp_asn:
                    asn_len.append(len(a))
                sorted_asn_inds = list(np.argsort(asn_len))
                new_asn = []
                new_scores = []
                pool_label_inds = []
                
                # Get the equivalent nuclei in the current distribution pool
                these_eqs = []
                for i in sorted_asn_inds:
                    new_asn.append(tmp_asn[i])
                    new_scores.append(tmp_scores[dist_pool[i]])
                    pool_label_inds.append(label_inds[dist_pool[i]])
                    
                    tmp_eq = []
                    for eq in equiv_inds[i]:
                        tmp_eq.append(dist_pool.index(eq))
                    these_eqs.append(tmp_eq)
                    
                    
                new_scores = np.array(new_scores)
                pool_label_inds = np.array(pool_label_inds)
                
                tmp_pool_asns = generate_valid_asns(new_asn, new_scores, len(dist_pool), len(shift_pool), pool_label_inds, exp_nums, these_eqs, valid_asns=[], already_linked=[], rank=0, Nmax=Nmax, Nmax_rank=Nmax_rank, max_excess=max_excess, disp_rank=disp_rank, t_start=start)
                
                # Reorder the assignments
                pool_asns = []
                for a in tmp_pool_asns:
                    pool_asns.append([a[sorted_asn_inds.index(i)] for i in range(len(a))])
            
            elif order == "decrease":
                
                # Reorder the distributions by decreasing number of possible assignments
                asn_len = []
                for a in tmp_asn:
                    asn_len.append(len(a))
                sorted_asn_inds = list(np.argsort(asn_len)[::-1])
                new_asn = []
                new_scores = []
                pool_label_inds = []
                
                # Get the equivalent nuclei in the current distribution pool
                these_eqs = []
                for i in sorted_asn_inds:
                    new_asn.append(tmp_asn[i])
                    new_scores.append(tmp_scores[dist_pool[i]])
                    pool_label_inds.append(label_inds[dist_pool[i]])
                    
                    tmp_eq = []
                    for eq in equiv_inds[i]:
                        tmp_eq.append(dist_pool.index(eq))
                    these_eqs.append(tmp_eq)
                    
                new_scores = np.array(new_scores)
                pool_label_inds = np.array(pool_label_inds)
                
                tmp_pool_asns = generate_valid_asns(new_asn, new_scores, len(dist_pool), len(shift_pool), pool_label_inds, exp_nums, these_eqs, valid_asns=[], already_linked=[], rank=0, Nmax=Nmax, Nmax_rank=Nmax_rank, max_excess=max_excess, disp_rank=disp_rank, t_start=start)
                
                pool_asns = []
                for a in tmp_pool_asns:
                    pool_asns.append([a[sorted_asn_inds.index(i)] for i in range(len(a))])
                
            elif order == "span":
                
                pool_asns = []
                for asn_inds in tqdm.tqdm(list(it.permutations(range(len(tmp_asn))))):
                    new_asn = []
                    new_scores = []
                    pool_label_inds = []

                    # Get the equivalent nuclei in the current distribution pool
                    these_eqs = []
                    for i in sorted_asn_inds:
                        new_asn.append(tmp_asn[i])
                        new_scores.append(tmp_scores[dist_pool[i]])
                        pool_label_inds.append(label_inds[dist_pool[i]])

                        tmp_eq = []
                        for eq in equiv_inds[i]:
                            tmp_eq.append(dist_pool.index(eq))
                        these_eqs.append(tmp_eq)
                        
                    new_scores = np.array(new_scores)
                    pool_label_inds = np.array(pool_label_inds)

                    tmp_pool_asns = generate_valid_asns(new_asn, new_scores, len(dist_pool), len(shift_pool), pool_label_inds, exp_nums, these_eqs, valid_asns=[], already_linked=[], rank=0, Nmax=Nmax, Nmax_rank=Nmax_rank, max_excess=max_excess, disp_rank=disp_rank, t_start=start)

                    for a in tmp_pool_asns:
                        pool_asns.append([a[asn_inds.index(i)] for i in range(len(a))])
            
            elif "rand" in order:
                num_iters = int(order.split("rand")[1])
                pool_asns = []
                ps = list(it.permutations(range(len(tmp_asn))))
                
                if len(ps) > num_iters:
                    for pi in np.random.choice(len(ps), num_iters):
                        asn_inds = list(ps[pi])
                        print(asn_inds)
                        new_asn = []
                        new_scores = []
                        pool_label_inds = []

                        # Get the equivalent nuclei in the current distribution pool
                        these_eqs = []
                        for i in sorted_asn_inds:
                            new_asn.append(tmp_asn[i])
                            new_scores.append(tmp_scores[dist_pool[i]])
                            pool_label_inds.append(label_inds[dist_pool[i]])

                            tmp_eq = []
                            for eq in equiv_inds[i]:
                                tmp_eq.append(dist_pool.index(eq))
                            these_eqs.append(tmp_eq)
                            
                        new_scores = np.array(new_scores)
                        pool_label_inds = np.array(pool_label_inds)

                        tmp_pool_asns = generate_valid_asns(new_asn, new_scores, len(dist_pool), len(shift_pool), pool_label_inds, exp_nums, these_eqs, valid_asns=[], already_linked=[], rank=0, Nmax=Nmax, Nmax_rank=Nmax_rank, max_excess=max_excess, disp_rank=disp_rank, t_start=start)

                        for a in tmp_pool_asns:
                            pool_asns.append([a[asn_inds.index(i)] for i in range(len(a))])
                
                else:
                    for p in ps:
                        asn_inds = list(p)
                        new_asn = []
                        new_scores = []
                        pool_label_inds = []

                        # Get the equivalent nuclei in the current distribution pool
                        these_eqs = []
                        for i in sorted_asn_inds:
                            new_asn.append(tmp_asn[i])
                            new_scores.append(tmp_scores[dist_pool[i]])
                            pool_label_inds.append(label_inds[dist_pool[i]])

                            tmp_eq = []
                            for eq in equiv_inds[i]:
                                tmp_eq.append(dist_pool.index(eq))
                            these_eqs.append(tmp_eq)
                            
                        new_scores = np.array(new_scores)
                        pool_label_inds = np.array(pool_label_inds)

                        tmp_pool_asns = generate_valid_asns(new_asn, new_scores, len(dist_pool), len(shift_pool), pool_label_inds, exp_nums, these_eqs, valid_asns=[], already_linked=[], rank=0, Nmax=Nmax, Nmax_rank=Nmax_rank, max_excess=max_excess, disp_rank=disp_rank, t_start=start)

                        for a in tmp_pool_asns:
                            pool_asns.append([a[asn_inds.index(i)] for i in range(len(a))])
            
            else:
                raise ValueError("Unknown order: {}".format(order))
            
            print("  {} valid assignments found!".format(len(pool_asns)))
            
            # Get the scores of the assignments found
            print("  Computing the scores for the valid assignments...")
            pool_scores = []
            for a in pool_asns:
                s = 1.
                for i, j in zip(dist_pool, a):
                    s *= tmp_scores[i, j]
                pool_scores.append(s)
            
            pool_scores = np.array(pool_scores)
            
            stop = time.time()
            print("Done. Time elapsed: {:.2f} s".format(stop-start))

            # Gather equivalent nuclei
            equiv = []
            for i in dist_pool:
                equiv.append([dist_pool.index(j) for j in equiv_inds[i]])
            all_equivs.append(equiv)
                
            print("Cleanup equivalent assignments...")
            
            start = time.time()
            
            cleaned_asns, cleaned_inds = np.unique(pool_asns, return_index=True, axis=0)
            
            cleaned_scores = pool_scores[cleaned_inds]
            
            cleaned_scores /= np.sum(cleaned_scores)
            
            stop = time.time()
            
            print("{} unique valid assignment extracted from the"
                  " {} assignments generated ({:.2f}%)."
                  " Time elapsed: {:.4f}s".format(len(cleaned_scores), len(pool_scores),
                                                  len(cleaned_scores)/len(pool_scores)*100, stop-start))
            
            all_cleaned_asns.append(cleaned_asns)
            all_cleaned_scores.append(cleaned_scores)
        
        already_assigned.extend(dist_pool)
        pool_ind += 1

    return all_dist_pools, all_shift_pools, all_cleaned_asns, all_cleaned_scores, new_labels, all_equivs



def print_probabilistic_assignment(dist_pools, shift_pools, pool_asns, pool_scores, labels, exp, equivs, f, display=False):
    """
    Print all assignments generated along with their probability
    
    Inputs: - dist_pools    Pools of nuclei/distributions
            - shift_pools   Pools of experimental shifts
            - pool_asns     Pools of generated assignments
            - pool_scores   Pools of scores
            - labels        Labels of the nuclei/distributions
            - exp           List of experimental shifts
            - equivs        List of equivalent nuclei/distributions
            - f             File to save the output to
            - display       Whether to display the results or not (WARNING: the output may be very large if a large number of assignments is generated!)
    """
    
    # Get the maximum length for label and shift display
    N_l = 0
    N_e = 0
    
    fac_e = 1
    for ds, eqs in zip(dist_pools, equivs):
        for eq in eqs:
            l = "/".join([labels[ds[i]] for i in eq])
            N_l = max(N_l, len(l))
        
            if "/" in l:
                n = l.count("/")
                fac_e = max(fac_e, n+1)
    
    for e in exp:
        N_e = max(N_e, len(e))
    N_e = max(N_e, len("100.00000000%"))
    
    N_e *= fac_e
    N_e += fac_e - 1
    N_e += 4
    
    # Initialize arrays
    already_assigned = []
    
    # Find the unambiguous assignments
    pp = "Unambiguous assignments:\n"
    
    # Loop over all assignment pools
    for ds, sh, asns, scores, eqs in zip(dist_pools, shift_pools, pool_asns, pool_scores, equivs):
        asns = np.array(asns)
        
        already_eq = []
        
        # Loop over all distributions (or multiplet of equivalent distributions)
        for eq in eqs:
            if eq[0] not in already_eq:
                these_asns = asns[:,eq]
                
                # Check if there is only one unique assignment for this distribution (or multiplet of equivalent distributions)
                unique_asns = []
                for a in these_asns:
                    if sorted(a) not in unique_asns:
                        unique_asns.append(sorted(a))
                        
                if len(unique_asns) == 1:
                    
                    # Print the corresponding unambiguous assignment
                    a = np.unique(unique_asns[0])
                    
                    # Print the label
                    l = "/".join([labels[ds[i]] for i in eq])
                    for _ in range(N_l - len(l)):
                        pp += " "
                    pp += l
                    
                    # Print the shift
                    e = "/".join([exp[i] for i in a])
                    for _ in range(N_e - len(e)):
                        pp += " "
                    pp += e
                    pp += "\n"
                    
                    already_assigned.extend([ds[i] for i in eq])
                
                already_eq.extend(eq)
        
    pp += "\n"
    
    # Loop over all assignment pools
    for ds, sh, asns, scores, eqs in zip(dist_pools, shift_pools, pool_asns, pool_scores, equivs):
        asns = np.array(asns)
        
        todo = False
        # If any distribution in the pool is not already unambiguously assigned:
        for d in ds:
            if d not in already_assigned:
                todo = True
                break
                
        if todo:
            
            pp += "Ambiguity between "
            
            sorted_inds = np.argsort(scores)[::-1]
            
            sel_ds = [d for d in ds if d not in already_assigned]
            sel_ds_inds = [ds.index(d) for d in sel_ds]
            sel_eqs = [eq for i, eq in enumerate(eqs) if i in sel_ds_inds]
            
            for d in sel_ds:
                pp += "{}, ".format(labels[d])
            pp = pp[:-2] + "\n"
            
            pp += "Corresponding shifts: "
            for s in sh:
                pp += "{}, ".format(exp[s])
            pp = pp[:-2] + "\n"
            
            pp += "Mapping {} distributions on {} shifts ({} possible assignments)\n".format(len(sel_ds), len(sh), len(scores))
            
            already_eq = []
            for i, d, eq in zip(sel_ds_inds, sel_ds, sel_eqs):
                
                if eq[0] not in already_eq:
                    these_asns = asns[:,eq]

                    # Print the label
                    l = "/".join([labels[ds[j]] for j in eq])
                    for _ in range(N_l - len(l)):
                        pp += " "
                    pp += l

                    # Print the shifts
                    for j in sorted_inds:
                        e = "/".join([exp[k] for k in these_asns[j]])
                        for _ in range(N_e - len(e)):
                            pp += " "
                        pp += e

                    pp += "\n"

                    already_eq.extend(eq)
        
            # Print the probabilities
            for _ in range(N_l - 1):
                pp += " "
            pp += "P"
            
            for j in sorted_inds:
                p = "{:.8f}%".format(scores[j]*100)
                for _ in range(N_e - len(p)):
                    pp += " "
                pp += p
            
            pp += "\n\n"
            
    # Save the output
    with open(f, "w") as F:
        F.write(pp)
    
    # Display the output
    if display:
        print(pp)
            
    return




def print_individual_probs(labels, exp, scores, f, display=False):
    """
    Print individual probabilities of assignment
    
    Inputs: - labels    Labels of the nuclei/distributions
            - exp       List of experimental shifts
            - scores    Matrix of assignment scores
            - f         File to save the output to
            - display   Whether to display the result or not
    """
    
    # Get the maximum length for label and shift display
    N_l = 0
    for l in labels:
        N_l = max(N_l, len(l))
    N_l = max(N_l, len("Label"))
    N_l += 4

    N_e = 0
    for e in exp:
        N_e = max(N_e, len(e))
    N_e = max(N_e, len("99.99%"))
    N_e += 4

    # Print the experimental shifts
    pp = ""
    for _ in range(N_l - len("Label")):
        pp += " "

    pp += "Label"

    for e in exp:
        for _ in range(N_e - len(e)):
            pp += " "
        pp += e

    pp += "\n"

    for l, s in zip(labels, scores):

        # Print the label
        for _ in range(N_l - len(l)):
            pp += " "
        pp += l

        # Print the assignment scores
        for si in s:
            si_str = "{:.2f}%".format(si*100)
            for _ in range(N_e - len(si_str)):
                pp += " "
            pp += si_str

        pp += "\n"

    # Save the output
    with open(f, "w") as F:
        F.write(pp)
    
    # Display the output
    if display:
        print(pp)
    
    return
    
    
    
def get_posterior(prior, likelihood):
    
    post = np.multiply(prior, likelihood)
    
    for i in range(post.shape[0]):
        if np.sum(post[i] > 0):
            post[i] /= np.sum(post[i])
    
    return post
    
    

def get_split_posterior(prior, likelihood):
    
    post = {}
    
    for key in prior.keys():
        post[key] = np.multiply(prior[key], likelihood[key])
        for i in range(prior[key].shape[0]):
            if np.sum(post[key][i]) > 0:
                post[key][i] /= np.sum(post[key][i])
    return post



def update_scores(old_scores, dist_pools, shift_pools, pool_asns, pool_scores, equivs, labels, all_labels, N=-1, p=-1):
    """
    Update the individual assignment scores given the possible assignments and associated probabilities
    
    Inputs: - old_scores    Original individual scores
            - dist_pools    Nuclei/distributions in each pool
            - shift_pools   Experimental shifts in each pool
            - pool_asns     Possible assignments for each pool
            - pool_scores   Scores associated with the possible assignments for each pool
            - equivs        List of equivalent nuclei/distributions
            - labels        List of labels
            - all_labels    List of individual labels
            - N             Maximum number of assignments to take into account
            - p             Maximum cummulative probability to take into account
    
    Output: - scores        Updated individual scores
    """
    
    init_N = N
    
    scores = np.copy(old_scores)
    # Loop over all assignment pools
    for ds, sh, these_asns, these_scores, eqs in zip(dist_pools, shift_pools, pool_asns, pool_scores, equivs):
    
        N = init_N
        # If a maximum number is given, only take into account the N most probable assignments
        if N > 0 and p > 0:
            raise ValueError("Select either N and p, but NOT both at the same time.")
        if N > 0 or p > 0:
            tmp_asns = copy.deepcopy(these_asns)
            sorted_inds = np.argsort(these_scores)[::-1]
            
            if p > 0:
                p_sum = 0.
                N = 0
                for s in these_scores[sorted_inds]:
                    p_sum += s
                    N += 1  
                    if p_sum > p / 100:
                        print("{:.2f}%/{:.2f}%\t{}/{}".format(p_sum*100, p, N, len(sorted_inds)))
                        break
            
            sorted_inds = sorted_inds[:N]
            these_asns = [tmp_asns[i] for i in sorted_inds]
            these_scores = these_scores[sorted_inds]

        these_asns = np.array(these_asns)
        
        already_d = []
        for d, eq in zip(ds, eqs):
            if d not in already_d:
                already_d.extend([ds[i] for i in eq])

                # Get the label corresponding to distribution d
                l = all_labels[d]
                for d_ind, l2 in enumerate(labels):
                    if l == l2 or l + "/" in l2 or l2.endswith("/" + l):
                        break
                scores[d_ind, :] = 0.

                # Get the individual scores of distribution d among the possible assignment
                sh_scores = np.zeros_like(sh, dtype="float")
                for i in eq:
                    for j, s in enumerate(sh):
                        match_inds = np.where(these_asns[:,i] == s)[0]
                        sh_scores[j] += np.sum(these_scores[match_inds])
                
                # Normalize the obtained scores
                sh_scores /= np.sum(these_scores)

                # Update the score matrix
                for s, sc in zip(sh, sh_scores):
                    scores[d_ind, s] = sc
    return scores



def update_split_scores(old_scores, dist_pools, shift_pools, pool_asns, pool_scores, equivs, labels, all_labels, N=-1, p=-1):
    """
    Update the individual assignment scores given the possible assignments and associated probabilities.
        Split the individual scores for different tuples of equivalent nuclei/distributions
    
    Inputs: - old_scores    Original individual scores
            - dist_pools    Nuclei/distributions in each pool
            - shift_pools   Experimental shifts in each pool
            - pool_asns     Possible assignments for each pool
            - pool_scores   Scores associated with the possible assignments for each pool
            - equivs        List of equivalent nuclei/distributions
            - labels        List of labels
            - all_labels    List of individual labels
            - N             Maximum number of assignments to take into account
    
    Output: - scores        Updated individual scores
    """
    
    init_N = N
    
    # Initialize the dictionary of scores
    scores = {}
    
    # Get all possible numbers of equivalent nuclei
    lens = []
    for eqs in equivs:
        for eq in eqs:
            if len(eq) not in lens:
                lens.append(len(eq))
        
    
    for le in lens:

        # Initialize the score matrix
        arr_shape = ()
        arr_shape += (old_scores.shape[0], )
        for _ in range(le):
            arr_shape += (old_scores.shape[1], )

        scores[le] = np.zeros(arr_shape)
    
        # Loop over all assignment pools
        for ds, sh, these_asns, these_scores, eqs in zip(dist_pools, shift_pools, pool_asns, pool_scores, equivs):

            N = init_N
            
            # If a maximum number is given, only take into account the N most probable assignments
            if N > 0 and p > 0:
                raise ValueError("Select either N and p, but NOT both at the same time.")
            if N > 0 or p > 0:
                tmp_asns = copy.deepcopy(these_asns)
                sorted_inds = np.argsort(these_scores)[::-1]

                if p > 0:
                    p_sum = 0.
                    N = 0
                    for s in these_scores[sorted_inds]:
                        p_sum += s
                        N += 1  
                        if p_sum > p / 100:
                            print("{:.2f}%/{:.2f}%\t{}/{}".format(p_sum*100, p, N, len(sorted_inds)))
                            break

                sorted_inds = sorted_inds[:N]
                these_asns = [tmp_asns[i] for i in sorted_inds]
                these_scores = these_scores[sorted_inds]

            these_asns = np.array(these_asns)
            already_d = []

            for d, eq in zip(ds, eqs):
                if d not in already_d and len(eq) == le:
                    already_d.extend([ds[i] for i in eq])

                    # Get the label corresponding to distribution d
                    l = all_labels[d]
                    for d_ind, l2 in enumerate(labels):
                        if l == l2 or l + "/" in l2 or l2.endswith("/" + l):
                            break
                    
                    # Get the shape of the individual score array
                    sh_shape = ()
                    for _ in range(le):
                        sh_shape += (len(sh), )
                    
                    # Get the individual scores of distribution d among the possible assignment
                    sh_scores = np.zeros(sh_shape, dtype="float")
                    
                    # Update the score matrix
                    for a, s in zip(these_asns, these_scores):
                        inds = (d_ind, )
                        inds += tuple(a[eq])
                        
                        scores[le][inds] += s
                        
                    scores[le][d_ind] /= np.sum(scores[le][d_ind])
    return scores



def print_split_individual_probs(labels, exp, scores, f, thresh=0.,  display=False):
    """
    Print individual assignment probabilities, split by number of equivalent nuclei/distributions
    
    Inputs: - labels        List of labels of nuclei/distributions
            - exp           List of experimental shifts
            - scores        Score dictionary (split by length of equivalent nuclei/distributions)
            - f             File to save the output to
            - thresh        Threshold for displaying a possible assignment
            - display       Whether to display the output or not
    """
    
    # Get the number of individual score matrices
    lens = scores.keys()
    
    pp = ""
    
    for le in lens:
        
        these_lab_inds = []
        these_exp_inds = []
        these_exps = []
        
        for i, l in enumerate(labels):
            this_le = l.count("/")+1
                
            # Check if the current label matches the current number of equivalent nuclei/distributions
            if this_le == le:
                these_lab_inds.append(i)
                
                # Get the assignments to consider
                sc_inds = np.array(np.where(scores[le][i] > thresh))
                
                # Get the experiments to assign the current label to
                for j in range(sc_inds.shape[1]):
                    if "/".join([exp[k] for k in sorted(sc_inds[:, j])]) not in these_exps:
                        these_exp_inds.append(sc_inds[:,j])
                        these_exps.append("/".join([exp[k] for k in sorted(sc_inds[:, j])]))
        
        # Get the maximum length of label and shift for display
        N_l = 0
        for l in [labels[i] for i in these_lab_inds]:
            N_l = max(N_l, len(l))
        N_l = max(N_l, len("Label"))
        N_l += 4

        N_e = 0
        for e in these_exps:
            N_e = max(N_e, len(e))
        N_e = max(N_e, len("99.99%"))
        N_e += 4

        for _ in range(N_l - len("Label")):
            pp += " "

        pp += "Label"

        # Print the shifts
        for e in these_exps:
            for _ in range(N_e - len(e)):
                pp += " "
            pp += e
        
        pp += "\n"
        
        # Print the labels and scores
        for i in these_lab_inds:
            l = labels[i]

            # Print the label
            for _ in range(N_l - len(l)):
                pp += " "
            pp += l
            
            # Print the scores
            for j in these_exp_inds:
                
                si = scores[le][i][tuple(j)]
                
                si_str = "{:.2f}%".format(si*100)
                
                for _ in range(N_e - len(si_str)):
                    pp += " "
                pp += si_str
                
            
            pp += "\n"
    
        pp += "\n"

    # Save the output
    with open(f, "w") as F:
        F.write(pp)
    
    # Display the output
    if display:
        print(pp)
    return



def split_scores(scores, labels):
    """
    Split the score matrix by the number of equivalent nuclei/distributions.
    
    Inputs: - scores            Score matrix
            - labels            List of labels of nuclei/distributions
    
    Output: - split_scores      Dictionary of scores split by the number of equivalent nuclei/distributions
    """
    
    # Initialize score dictionary
    split_scores = {}
    
    # Get all numbers of equivalent nuclei/distributions
    lens = []
    for l in labels:
        if l.count("/") + 1 not in lens:
            lens.append(l.count("/") + 1)
            
    for le in lens:
        arr_shape = ()
        arr_shape += (scores.shape[0], )
        
        sc_shape = ()
        for _ in range(le):
            arr_shape += (scores.shape[1], )
            sc_shape += (scores.shape[1], )

        # Generate score matrix
        split_scores[le] = np.zeros(arr_shape)
    
        for i, l in enumerate(labels):
            if l.count("/") + 1 == le:
                
                sc = np.zeros(sc_shape)
                
                # Update the split score matrix
                if le == 1:
                    split_scores[le][i] = scores[i]
                
                else:
                    A = np.outer(scores[i], scores[i])
                    shape = (scores.shape[1], scores.shape[1])
                    A.reshape(shape)
                    
                    for _ in range(le-2):
                        A = np.tensordot(A, scores[i], axes=0)
                        shape += (scores.shape[1],)
                    
                    for inds in np.ndindex(A.shape):
                        for j, k in zip(inds[:-1], inds[1:]):
                            if k < j:
                                A[inds] = 0
                    
                    split_scores[le][i] = A
                
    return split_scores
