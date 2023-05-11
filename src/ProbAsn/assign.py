####################################################################################################
###                                                                                              ###
###                          Functions for probabilistic assignment                              ###
###                               Author: Manuel Cordova (EPFL)                                  ###
###                                Last modified: 03.09.2021                                     ###
###                                                                                              ###
####################################################################################################

# Import libraries
import numpy as np
import time
import re



def increase_threshold(thresh, thresh_inc):
    """
    Increase the threshold based on a string

    Inputs: - thresh        Initial threshold
            - thresh_inc    How the threshold should be increased.
                                "xN": multiply by N
                                "+N": increase by N

    Output: - thresh        Increased threshold
    """

    if thresh_inc.startswith("x"):
        return thresh * float(thresh_inc.split("x")[1])

    elif thresh_inc.startswith("+"):
        return thresh + float(thresh_inc.split("+")[1])

    raise ValueError("Unkown threshold update: {}".format(thresh_inc))



def get_possible_assignments(scores, labels, exp, thresh=100., thresh_increase="x2"):
    """
    Get all possible assignments for each probability distribution, given that if an individual probability of
    assigning the distribution to a shift is more than "thresh" times higher than the probability of assigning it
    to another shift, the latter possibility is discarded.

    Inputs: - scores                Array of individual scores
            - labels                List of labels of the distributions
            - exp                   List of experimental shifts
            - thresh                Relative probability threshold to discard a score
            - thresh_increase       How the threshold should be increased.
                                        "xN": multiply by N
                                        "+N": increase by N

    Output: - possible_assignments  Lists of possible assignments for each distribution
            - thresh                Updated relative probability threshold to discard a score
    """

    # If we do not set a threshold, consider all assignments as possible
    #   NOT RECOMMENDED, scaling is factorial!
    if thresh < 0.:
        print("WARNING: Not setting a threshold for considering plausible assignments is not recommended, scaling is factorial!")
        possible_assignments = [list(range(len(exp))) for _ in scores]
        return possible_assignments, thresh

    consistent = False
    while not consistent:
        cleaned_scores = np.copy(scores)
        for i in range(len(cleaned_scores)):
            m = np.max(cleaned_scores[i])
            # Discard assignments that have an individual score lower than 1/thresh times the maximum score
            cleaned_scores[i, cleaned_scores[i] < m / thresh] = 0.

        # Get the possible assignments
        possible_assignments = []
        for i, s in enumerate(cleaned_scores):
            possible_assignments.append(list(np.where(s > 0.)[0]))

        # Clean up assignments, i.e. if only one distribution can be assigned to a given shift, then it must be assigned to that shift
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
                        # If the specific assignment is not found anywhere else, the shift can only be assigned to distribution i
                        if not found:
                            cleaned_scores[i,[aj for aj in a if aj != ai]] = 0.
                            possible_assignments[i] = [ai]
                            change = True
                            break
                if change:
                    break

        consistent = True
        # Check that each distribution can be assigned to at least one experimental shift
        for s in cleaned_scores:
            if np.sum(s) == 0.:
                consistent = False
                thresh = increase_threshold(thresh, thresh_increase)
                break

        # Check that each experimental shift can be assigned to at least one distribution
        for s in cleaned_scores.T:
            if np.sum(s) == 0.:
                consistent = False
                thresh = increase_threshold(thresh, thresh_increase)
                break

    print("  Scores cleaned up, threshold set to {}\n".format(thresh))

    # When assigning 2D distributions, if two central or neighbouring atoms in different graphs are the same,
    #   then we merge all possible assignments
    if "-" in labels[0]:
        complete_asns = []
        for i, (l, a) in enumerate(zip(labels, possible_assignments)):
            complete_asns.append(a)
            if "/" in l:
                ls = [tmp.split("-") for tmp in l.split("/")]
                for i1, l1 in enumerate(ls):
                    for i2, l2 in enumerate(ls):
                        if i2 > i1:
                            for k, (l1k, l2k) in enumerate(zip(l1, l2)):
                                if l1k == l2k:
                                    for ai in complete_asns[i]:
                                        match = exp[ai].split("\\")[k]
                                        for j, e in enumerate(exp):
                                            if match == e.split("\\")[k] and e not in [exp[aj] for aj in complete_asns[i]]:
                                                complete_asns[i].append(j)

        possible_assignments = []
        for a in complete_asns:
            possible_assignments.append(list(np.sort(a)))

    return possible_assignments, thresh



def get_assignment_pool(possible_asn, assigned):
    """
    Get an assignment pool, i.e. a set of distributions and shifts to assign independently from the rest

    Inputs:     - possible_asn      List of possible assignments for each distribution
                - assigned          List of already assigned distributions

    Outputs:    - dist_pool         Pool of distributions
                - shift_pool        Shifts in the pool of distributions
    """

    # Initialize arrays of possible shifts and distributions
    dist_pool = []
    shift_pool = []

    # Loop over all distribution
    for i, a in enumerate(possible_asn):
        if i not in assigned:
            # Get the possible shifts for the first not already assigned distribution
            shift_pool.extend(a)
            dist_pool.append(i)
            break

    change = True
    while change:
        change = False
        # Loop over all distributions
        for i, a in enumerate(possible_asn):
            if i not in dist_pool and i not in assigned:
                for ai in a:
                    # If any of the possible shifts for this distribution is within the pool, add this distribution
                    # to the possible assignment pool
                    if ai in shift_pool:
                        dist_pool.append(i)
                        shift_pool.extend([aj for aj in a if aj not in shift_pool])
                        change = True
                        break

    return dist_pool, shift_pool



def generate_global_asns(possible_asns, scores, n_dist, n_exp, ls, es, equiv,
                         global_asns=[], already_linked=[], dqsq=False, rank=0, max_asn=None,
                         r_max_asn=0, max_excess=None, disp_rank=-1, t_start=None):
    """
    Recursively generate global assignments given possible individual assignments.

    Inputs: - possible_asns     Possible assignments for each nucleus/distribution
            - scores            Matrix of scores for individual assignments
            - n_dist            Number of distributions
            - n_exp             Number of experiments
            - ls                Labels of the nuclei/distributions
            - es                Experimental shifts
            - equiv             Equivalent nuclei/distributions
            - global_asns       Already found global assignments
            - already_linked    Already assigned nuclei/distributions
            - dqsq              Whether the neighbouring shift should be double quantum (only for 2D experiments)
            - rank              Assignment rank
            - max_asn           Maximum number of assignments to consider
            - r_max_asn         Rank from which to start reducing the number of assignments
            - max_excess        Maximum excess for assignment (defined as the maximum
                                    number of nuclei/distributions assigned
                                    to a single experimental shift minus one)
            - disp_rank         Rank up to which progress should be displayed
            - t_start           Starting time

    Output: - global_asns       List of global assignments generated
    """

    # If all distributions are already assigned, append the assignment found
    if rank == n_dist:

        # Sort the equivalent parts
        sorted_asn = already_linked.copy()
        already_eq = []
        for eq in equiv:
            if len(eq) > 1 and eq[0] not in already_eq:
                tmp = sorted([already_linked[i] for i in eq])
                for i, e in enumerate(eq):
                    sorted_asn[e] = tmp[i]
                already_eq.extend(eq)

        # Return the list of already generated global assignments
        global_asns.append(sorted_asn)
        return global_asns

    # Get the number of assignments to consider for this distribution
    rank_len = len(possible_asns[rank])
    if max_asn is not None and rank >= r_max_asn:
        rank_len = min(max_asn, rank_len)

    # Get the intermediate scores
    these_scores = []
    for a in possible_asns[rank]:
        # Generate the assignment
        this_asn = already_linked.copy()
        this_asn.append(a)

        # Check if the assignment is valid so far (get the excess and maximum individual excess)
        excess = 0
        ind_excess = 0
        for j in np.unique(this_asn):
            excess += max(this_asn.count(j) - 1, 0)
            ind_excess = max(ind_excess, this_asn.count(j) - 1)

        # Check that same nuclei are assigned to the same shift (important for 2D simulated experiments)
        for i, a1 in enumerate(this_asn):
            for j, a2 in enumerate(this_asn):
                if i < j:
                    for k in range(len(ls[i])):
                        # If the same nucleus is assigned to two different shifts, discard the assignment
                        if ls[i,k] == ls[j,k] and es[a1,k] != es[a2,k]:
                            excess = n_dist - n_exp + 1
                            break
                if excess > n_dist - n_exp:
                    break
            if excess > n_dist - n_exp:
                break

        # If the assignment is valid so far, try to assign the next distribution
        if excess > n_dist - n_exp or (max_excess is not None and ind_excess > max_excess):
            score = 0.
        else:
            # Get the corresponding score
            score = 1.
            for i, a in enumerate(this_asn):
                score *= scores[i, a]

        these_scores.append(score)

    # Sort the intermediate scores by decreasing value
    score_inds = np.argsort(these_scores)[::-1]

    # Remove instances where the score is zero
    score_inds = score_inds[:np.count_nonzero(these_scores)]

    # If max_asn is set, get the max_asn highest scores
    if max_asn is not None and rank >= r_max_asn:
        score_inds = score_inds[:max_asn]

    asn_num = 0
    for i in score_inds:
        asn_num += 1

        if rank <= disp_rank:
            space = ""
            for _ in range(rank):
                space += " "
            t_stop = time.time()
            if t_start is not None:
                print(space + "Assigning nucleus {}, {}/{}, {} valid assignments generated until now. Time elapsed: {:.2f} s".format(rank, asn_num, len(score_inds), len(global_asns), t_stop - t_start))
            else:
                print(space + "Assigning nucleus {}, {}/{}, {} valid assignments generated until now".format(rank, asn_num, len(score_inds), len(global_asns)))

        # Try assigning distribution "rank" to experiment at index i
        this_asn = already_linked.copy()
        this_asn.append(possible_asns[rank][i])

        # Check if the assignment is valid so far
        excess = 0
        ind_excess = 0
        for j in np.unique(this_asn):
            excess += max(this_asn.count(j) - 1, 0)
            ind_excess = max(ind_excess, this_asn.count(j) - 1)

        # Check that same nuclei are assigned to the same shift (important for 2D simulated experiments)
        for i, a1 in enumerate(this_asn):
            for j, a2 in enumerate(this_asn):
                if i < j:
                    for k in range(len(ls[i])):
                        # If two instances of the same nucleus (same label) are not associated with the same experimental shift, the assignment is invalid
                        if ls[i,k] == ls[j,k] and es[a1,k] != es[a2,k]:
                            excess = n_dist - n_exp + 1
                            break
                if excess > n_dist - n_exp:
                    break
            if excess > n_dist - n_exp:
                break

        # If the assignment is valid so far, try to assign the next distribution
        if excess <= n_dist - n_exp and (max_excess is None or ind_excess <= max_excess):
            global_asns = generate_global_asns(possible_asns, scores, n_dist, n_exp, ls, es, equiv,
                                               global_asns=global_asns, already_linked=this_asn,
                                               rank=rank+1, max_asn=max_asn, r_max_asn=r_max_asn,
                                               max_excess=max_excess, disp_rank=disp_rank, t_start=t_start)

    return global_asns



def get_probabilistic_assignment(scores, possible_assignments, exp, labels,
                                 max_asn=None, r_max_asn=0, disp_rank=-1,
                                 max_excess=None, order="default", pool_inds=None, verbose=False):
    """
    Get the probabilistic assignment given individual possible assignments and scores

    Inputs:     - scores                    Matrix of scores
                - possible_assignments      List of possible assignments for each nucleus/distribution
                - exp                       List of experimental shifts
                - labels                    List of labels of the nuclei/distributions
                - max_asn                   Maximum number of assignments to consider for the assignment generation
                - r_max_asn                 Rank from which to select only the max_asn most probable assignments
                - disp_rank                 Rank up to which to display progress
                - max_excess                Maximum number of distributions that can be assigned to a single shift
                - order                     Order in which to make the assignment
                                                "default": Perform the assignment in the default order
                                                "increase": Perform the assignment starting with
                                                    the distributions with the least possible individual assignments
                                                "decrease": Perform the assignment starting with
                                                    the distributions with the most possible individual assignments
                - pool_inds                 Indices of the pool to assign (useful to set different parameters to assign different pools)
                - verbose                   Whether additional information about the assignment should be printed

    Outputs:    - dist_pools                Indices of the distributions in each pool
                - shift_pools               Indices of the shifts in each pool
                - output_asns               Possible assignments for each pool
                - output_scores             Scores of the possible assignments for each pool
                - all_labels                Individual labels for each distribution
                - output_equivs             List of equivalent distributions
    """

    # Separate topologically equivalent nuclei
    all_asns = []
    all_labels = []
    all_scores = []
    label_inds = []
    equiv_inds = []
    
    # Loop over all labels
    for asgn, l, s in zip(possible_assignments, labels, scores):
        
        these_equiv = []
        # Get equivalent nuclei (identified with a "/" in the label)
        for li in l.split("/"):
            # Get possible individual assignments
            all_asns.append(asgn)
            # Get the label
            all_labels.append(li)
            # Get the scores
            all_scores.append(s)
            # Identify the equivalent nuclei in the new arrays
            these_equiv.append(all_labels.index(li))

            # Extract the label indices
            these_inds = []
            for i, l2 in enumerate(li.split("-")):
                try:
                    these_inds.append(int(re.findall("\d+", l2)[0]))
                except:
                    these_inds.append(i)
            label_inds.append(these_inds)
        
        # Set the equivalent nuclei
        for li in l.split("/"):
            equiv_inds.append(these_equiv)
    
    all_scores = np.array(all_scores)
    label_inds = np.array(label_inds)
    shaped_exp = np.array(exp).reshape((len(exp), -1))
    
    # Initialize the array of assigned nuclei and shifts
    assigned = []
    dist_pools = []
    shift_pools = []
    output_asns = []
    output_scores = []
    output_equivs = []

    pool_ind = 0

    # Loop until all distributions are assigned
    while len(assigned) < len(all_labels):
        # Get the assignment pool
        dist_pool, shift_pool = get_assignment_pool(all_asns, assigned)
        
        # Check if the assignment pool is in the selected pools
        if pool_inds is None or pool_ind in pool_inds:

            # Display pool informations
            tmp_pp = "Assignment pool: "
            already_labs = []
            for l in [all_labels[i] for i in dist_pool]:
                tmp_pp += "{}, ".format(l)
            tmp_pp = tmp_pp[:-2]
            print(tmp_pp)

            these_asns = []
            for i in dist_pool:
                these_asns.append(all_asns[i])

            print("Corresponding shifts: {}".format(", ".join([str(exp[i]) for i in shift_pool])))
            print("Assigning {} nuclei to {} shifts".format(len(dist_pool), len(shift_pool)))
            
            if verbose:
                pp = ""
                for i in dist_pool:
                    pp += "  Possible assignments for {}: ".format(all_labels[i])
                    pp += ", ".join([str(exp[j]) for j in all_asns[i]])
                    pp += "\n"
                print(pp)
            
            print("Generating global assignments...")

            start = time.time()

            # Generate valid global assignments
            if order == "default":
                pool_label_inds = label_inds[dist_pool]

                # Get the equivalent nuclei in the current pool
                these_eqs = []
                for i in dist_pool:
                    tmp_eq = []
                    for eq in equiv_inds[i]:
                        tmp_eq.append(dist_pool.index(eq))
                    these_eqs.append(tmp_eq)

                pool_asns = generate_global_asns(these_asns, all_scores[dist_pool], len(dist_pool), len(shift_pool),
                                                 pool_label_inds, shaped_exp, these_eqs, global_asns=[], already_linked=[], rank=0,
                                                 max_asn=max_asn, r_max_asn=r_max_asn, max_excess=max_excess, disp_rank=disp_rank, t_start=start)

            elif order == "increase":

                # Reorder the distributions by decreasing number of possible assignments
                asn_len = []
                for a in these_asns:
                    asn_len.append(len(a))
                sorted_asn_inds = list(np.argsort(asn_len))
                
                sorted_dist_pool = [dist_pool[i] for i in sorted_asn_inds]
                pool_label_inds = label_inds[sorted_dist_pool]
                
                sorted_asns = []
                sorted_scores = []
                sorted_label_inds = []
                sorted_eqs = []

                # Get the equivalent nuclei in the current distribution pool
                these_eqs = []
                for i in sorted_asn_inds:
                    sorted_asns.append(these_asns[i])
                    sorted_scores.append(all_scores[dist_pool[i]])
                    sorted_label_inds.append(label_inds[dist_pool[i]])

                    tmp_eq = []
                    for eq in equiv_inds[sorted_dist_pool[i]]:
                        tmp_eq.append(sorted_dist_pool.index(eq))
                    these_eqs.append(tmp_eq)

                sorted_scores = np.array(sorted_scores)
                sorted_label_inds = np.array(sorted_label_inds)

                tmp_pool_asns = generate_global_asns(sorted_asns, sorted_scores, len(dist_pool), len(shift_pool),
                                                     pool_label_inds, shaped_exp, these_eqs, global_asns=[], already_linked=[], rank=0,
                                                     max_asn=max_asn, r_max_asn=r_max_asn, max_excess=max_excess, disp_rank=disp_rank, t_start=start)

                # Reorder the assignments
                pool_asns = []
                for a in tmp_pool_asns:
                    pool_asns.append([a[sorted_asn_inds.index(i)] for i in range(len(a))])

            elif order == "decrease":

                # Reorder the distributions by decreasing number of possible assignments
                asn_len = []
                for a in these_asns:
                    asn_len.append(len(a))
                sorted_asn_inds = list(np.argsort(asn_len)[::-1])
                
                sorted_dist_pool = [dist_pool[i] for i in sorted_asn_inds]
                pool_label_inds = label_inds[sorted_dist_pool]
                
                sorted_asns = []
                sorted_scores = []
                sorted_label_inds = []
                sorted_eqs = []

                # Get the equivalent nuclei in the current distribution pool
                these_eqs = []
                for i in sorted_asn_inds:
                    sorted_asns.append(these_asns[i])
                    sorted_scores.append(all_scores[dist_pool[i]])
                    sorted_label_inds.append(label_inds[dist_pool[i]])

                    tmp_eq = []
                    for eq in equiv_inds[sorted_dist_pool[i]]:
                        tmp_eq.append(sorted_dist_pool.index(eq))
                    these_eqs.append(tmp_eq)

                sorted_scores = np.array(sorted_scores)
                sorted_label_inds = np.array(sorted_label_inds)

                tmp_pool_asns = generate_global_asns(sorted_asns, sorted_scores, len(dist_pool), len(shift_pool),
                                                     pool_label_inds, shaped_exp, these_eqs, global_asns=[], already_linked=[], rank=0,
                                                     max_asn=max_asn, r_max_asn=r_max_asn, max_excess=max_excess, disp_rank=disp_rank, t_start=start)

                # Reorder the assignments
                pool_asns = []
                for a in tmp_pool_asns:
                    pool_asns.append([a[sorted_asn_inds.index(i)] for i in range(len(a))])
            
            else:
                raise ValueError("Unknown order: {}".format(order))

            print("{} global assignments found!".format(len(pool_asns)))
            
            # Cleanup equivalent assignments
            print("Cleaning up equivalent assignments...")
            start = time.time()
            cleaned_asns = np.unique(pool_asns, axis=0)
            stop = time.time()
            print("{} unique global assignments extracted from the {} assignments generated. Time elapsed: {:.4f} s".format(len(cleaned_asns), len(pool_asns), stop - start))
            
            # Get the scores
            print("Computing the scores for the unique global assignments...")
            start = time.time()
            cleaned_scores = [np.prod(all_scores[dist_pool, a]) for a in cleaned_asns]
            cleaned_scores /= np.sum(cleaned_scores)
            stop = time.time()
            print("Done. Time elapsed: {:.4f} s\n".format(stop - start))
            
            # Gather equivalent nuclei
            these_equiv = []
            for i in dist_pool:
                these_equiv.append([dist_pool.index(j) for j in equiv_inds[i]])
            
            # Output the pool to the global assignments
            dist_pools.append(dist_pool)
            shift_pools.append(shift_pool)
            output_equivs.append(these_equiv)
            output_asns.append(cleaned_asns)
            output_scores.append(cleaned_scores)

        assigned.extend(dist_pool)
        pool_ind += 1

    return dist_pools, shift_pools, output_asns, output_scores, all_labels, output_equivs



def write_global_probs(dist_pools, shift_pools, pool_asns, pool_scores, labels, exp, equivs, f, decimals=8, display=False):
    """
    Print all assignments generated along with their probability

    Inputs: - dist_pools    Pools of distributions
            - shift_pools   Pools of experimental shifts
            - pool_asns     Pools of generated assignments
            - pool_scores   Pools of scores
            - labels        Labels of distributions
            - exp           List of experimental shifts
            - equivs        List of equivalent distributions
            - f             File to save the output to
            - decimals      Number of decimals to write for probabilities
            - display       Whether to display the results or not
                                (WARNING: the output may be very large
                                if a large number of assignments is generated!)
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

    N_e *= fac_e
    N_e += fac_e - 1
    N_e = max(N_e, len("100.%") + max(1, decimals)) + 1

    # Initialize arrays
    already_assigned = []

    # Find the unambiguous assignments
    pp = "Unambiguous assignments:\n"
    unambiguous = False

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
                    unambiguous = True

                already_eq.extend(eq)

    pp += "\n"
    
    if not unambiguous:
        pp = ""

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
                p = str(round(scores[j] * 100, decimals)) + "%"
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



def write_individual_probs(labels, exp, scores, f, decimals=2, display=False):
    """
    Write individual probabilities of assignment to a file

    Inputs: - labels        Labels of the nuclei/distributions
            - exp           List of experimental shifts (in string format)
            - scores        Matrix of assignment scores
            - f             File to save the output to
            - decimals      Number of decimals to write for probabilities
            - display       Whether to display the result or not
    """

    # Get the maximum length for label and shift display
    N_l = 0
    for l in labels:
        N_l = max(N_l, len(l))
    N_l = max(N_l, len("Label"))

    N_e = 0
    for e in exp:
        N_e = max(N_e, len(e))
    N_e = max(N_e, len("100.%") + max(1, decimals)) + 1

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
            si_str = format(si * 100, ".{}f".format(decimals)) + "%"
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



def update_split_scores(dist_pools, shift_pools, pool_asns, pool_scores, equivs, labels, all_labels):
    """
    Update the individual assignment scores given the possible assignments and associated probabilities.
        Split the individual scores for different tuples of equivalent nuclei/distributions

    Inputs: - dist_pools    Distributions in each pool
            - shift_pools   Experimental shifts in each pool
            - pool_asns     Possible assignments for each pool
            - pool_scores   Scores associated with the possible assignments for each pool
            - equivs        List of equivalent nuclei/distributions
            - labels        List of labels
            - all_labels    List of individual labels

    Output: - scores        Updated individual scores
    """
    
    n_dist = len(labels)
    n_shifts = sum([len(sh) for sh in shift_pools])

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
        arr_shape += (n_dist, )
        for _ in range(le):
            arr_shape += (n_shifts, )

        scores[le] = np.zeros(arr_shape)

        # Loop over all assignment pools
        for ds, sh, these_asns, these_scores, eqs in zip(dist_pools, shift_pools, pool_asns, pool_scores, equivs):

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

                    # Update the score matrix
                    for a, s in zip(these_asns, these_scores):
                        inds = (d_ind, )
                        inds += tuple(a[eq])

                        scores[le][inds] += s
    return scores



def write_split_individual_probs(labels, exp, scores, f, thresh=0., decimals=2, display=False):
    """
    Print individual assignment probabilities, split by number of equivalent nuclei/distributions

    Inputs: - labels        List of labels of nuclei/distributions
            - exp           List of experimental shifts
            - scores        Score dictionary (split by length of equivalent nuclei/distributions)
            - f             File to save the output to
            - thresh        Threshold for displaying a possible assignment
            - decimals      Number of decimalsto write for probabilities
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
            this_le = l.count("/") + 1

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

        N_e = 0
        for e in these_exps:
            N_e = max(N_e, len(e))
        N_e = max(N_e, len("100.%") + max(1, decimals)) + 1

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

                si_str = format(si * 100, ".{}f".format(decimals)) + "%"

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
