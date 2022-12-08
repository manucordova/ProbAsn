####################################################################################################
###                                                                                              ###
###                             Functions for simulating spectra                                 ###
###                               Author: Manuel Cordova (EPFL)                                  ###
###                                Last modified: 03.09.2021                                     ###
###                                                                                              ###
####################################################################################################

# Import libraries
import numpy as np
import networkx as nx

# Set criterion for equivalent nodes
nm = nx.algorithms.isomorphism.categorical_node_match("elem", "X")

# Set criterion for equivalent edges
em = nx.algorithms.isomorphism.categorical_edge_match("w", -1)



def cleanup_methyl_protons(labels, Gs, envs, shifts, errs, ws, crysts, inds, atoms, bonds):
    """
    Gather methyl proton shifts into only one per methyl group
    
    Inputs:     - labels        List of labels for each distribution
                - Gs            Graph of each distribution
                - envs          Environment of each graph
                - shifts        Predicted shifts in each distribution
                - errs          Prediction errors in each distribution
                - ws            Maximum depth for each distribution
                - crysts        Crystals in each distribution
                - inds          Indices of the atoms in each distribution
                - atoms         List of atoms in the molecule
                - bonds         Bonded atoms for each atom in the molecule (by index)
    
    Outputs:    - new_labels    Cleaned list of labels for each distribution
                - new_Gs        Cleaned list of graphs
                - new_shifts    Cleaned list of predicted shifts in each distribution
                - new_errs      Cleaned list of prediction errors in each distribution
                - new_ws        Cleaned list of maximum depth for each distribution
                - new_crysts    Cleaned array of crystals in each distribution
                - new_inds      Cleaned array of indices of the atoms in each distribution
    """
    
    # Initialize new arrays
    new_labels = []
    new_Gs = []
    new_envs = []
    new_shifts = []
    new_errs = []
    new_ws = []
    new_crysts = []
    new_inds = []
    
    # Array to store already identified methyl groups
    methyls = []
    
    # Loop over all graphs
    for l, G, env, sh, er, w, cryst, ind in zip(labels, Gs, envs, shifts, errs, ws, crysts, inds):
        
        # Get the index of the central node
        i = G.nodes[0]["ind"]
        # Check that there is only one neighbour (H should be linked to only one atom
        if len(bonds[i]) == 1:
            j = bonds[i][0]
            
            # Check if there are at least three protons linked to the neighbour
            nH = [atoms[k] for k in bonds[j]].count("H")
            
            if nH >= 3:
                # If we find a new methyl proton, add it,
                #   otherwise skip it if it is part of an already known methyl
                if j not in methyls:
                    methyls.append(j)
                    new_labels.append(l)
                    new_Gs.append(G)
                    new_envs.append(env)
                    new_shifts.append(sh)
                    new_errs.append(er)
                    new_ws.append(w)
                    new_crysts.append(cryst)
                    new_inds.append(ind)
            
            # If this is not a methyl proton, add it
            else:
                new_labels.append(l)
                new_Gs.append(G)
                new_envs.append(env)
                new_shifts.append(sh)
                new_errs.append(er)
                new_ws.append(w)
                new_crysts.append(cryst)
                new_inds.append(ind)
        
        # If the proton is bonded to more than one atom, add it
        else:
            new_labels.append(l)
            new_Gs.append(G)
            new_envs.append(env)
            new_shifts.append(sh)
            new_errs.append(er)
            new_ws.append(w)
            new_crysts.append(cryst)
            new_inds.append(ind)
    
    return new_labels, new_Gs, new_envs, new_shifts, new_errs, new_ws, new_crysts, new_inds



def cleanup_methyls(labels, shifts, errs, ws, crysts, inds, hashes, atoms, bonds):
    """
    Gather methyl 2D shifts
    
    Inputs:     - labels        List of labels for each distribution
                - shifts        Predicted shifts in each distribution
                - errs          Prediction errors in each distribution
                - ws            Maximum depth for each distribution
                - crysts        Crystals in each distribution
                - inds          Indices of the atoms in each distribution
                - hashes        List of hashes for each graph
                - atoms         List of atoms in the molecule
                - bonds         Bonded atoms for each atom in the molecule (by index)
    
    Outputs:    - new_labels    Cleaned list of labels for each distribution
                - new_Gs        Cleaned list of graphs
                - new_shifts    Cleaned list of predicted shifts in each distribution
                - new_errs      Cleaned list of prediction errors in each distribution
                - new_ws        Cleaned list of maximum depth for each distribution
                - new_crysts    Cleaned array of crystals in each distribution
                - new_inds      Cleaned array of indices of the atoms in each distribution
    """
    
    # Initialize the updated lists
    new_labels = []
    new_shifts = []
    new_errs = []
    new_ws = []
    new_crysts = []
    new_inds = []
    new_hashes = []
    
    # Array to store already identified methyl groups
    methyls = []
    
    # Get element
    elem = ""
    for c in labels[0]:
        if c.isdigit():
            break
        elem += c
    
    # Loop over all distributions
    for l, sh, er, w, cryst, ind, h in zip(labels, shifts, errs, ws, crysts, inds, hashes):
    
        # Get the index of the central node
        i0 = int(l.split("-")[0].split(elem)[1]) - 1
        i = [k for k, e in enumerate(atoms) if e == elem][i0]
    
        # Check if there are at least three protons linked to the atom
        nH = [atoms[k] for k in bonds[i]].count("H")
        
        if nH >= 3:
            
            # If we find a new methyl proton, add it,
            #   otherwise skip it if it is part of an already known methyl
            if i not in methyls:
                methyls.append(i)
                new_labels.append(l)
                new_shifts.append(sh)
                new_errs.append(er)
                new_ws.append(w)
                new_crysts.append(cryst)
                new_inds.append(ind)
                new_hashes.append(h)
        
        # If this is not a methyl, add it
        else:
            new_labels.append(l)
            new_shifts.append(sh)
            new_errs.append(er)
            new_ws.append(w)
            new_crysts.append(cryst)
            new_inds.append(ind)
            new_hashes.append(h)
        
    return new_labels, new_shifts, new_errs, new_ws, new_crysts, new_inds, new_hashes



def cleanup_equivalent(labels, shifts, errs, ws, crysts, inds, hashes):
    """
    Gather equivalent graphs (identified by their shift distributions)
    
    Inputs:     - labels        List of labels of the distributions
                - shifts        List of predicted shifts in each distribution
                - errs          List of predicted errors in each distribution
                - ws            List of weights of the distributions
                - crysts        List of crystals in each distribution
                - inds          List of the atoms in each distribution
                - hashes        List of hashes for each graph
                
    Outputs:    - new_labels    Updated list of labels of the distributions
                - new_shifts    Updated list of predicted shifts in each distribution
                - new_errs      Updated list of predicted errors in each distribution
                - new_ws        Updated list of weights of the distributions
                - new_crysts    Updated list of crystals in each distribution
                - new_inds      Updated list of the atoms in each distribution
    """
    
    # Initialize the updated lists
    new_labels = []
    new_shifts = []
    new_errs = []
    new_ws = []
    new_crysts = []
    new_inds = []
    new_hashes = []
    
    # Loop over all the distributions
    for l, sh, er, w, cr, ind, h in zip(labels, shifts, errs, ws, crysts, inds, hashes):
        
        # If the distribution is already found, modify the label
        if h in new_hashes:
            i = new_hashes.index(h)
            new_labels[i] += "/{}".format(l)
        
        # Otherwise, append the distribution to the updated list
        else:
            new_labels.append(l)
            new_shifts.append(sh)
            new_errs.append(er)
            new_ws.append(w)
            new_crysts.append(cr)
            new_inds.append(ind)
            new_hashes.append(h)
    
    return new_labels, new_shifts, new_errs, new_ws, new_crysts, new_inds, new_hashes



def get_lims_1D(all_shifts, all_errs, extend=0.1, dx="rms"):
    """
    Get the limits for a predicted 1D spectrum: obtain furthest peaks ± err to determine range, extend by a factor

    Inputs: - all_shifts    List of shifts in the distributions
            - all_errs      List of predicted errors in the distributions
            - extend        How much to extend the range (fraction of the original range, applied to both sides)
            - dx            How the error is incorporated to the minimum/maximum shift:
                                "sel": use the error of the minimum and maximum peaks
                                "max": use the maximum error
                                "mean": use the mean error
                                "rms": use the rms error

    Output: - lx            Limits in the x-dimension
    """

    # Initialize limits
    lx = np.ones(2) * np.mean(all_shifts[0])

    # Get minimum and maximum peak of each distribution
    for shifts, errs in zip(all_shifts, all_errs):

        # Get indices of the minimum and maximum peaks
        imin = np.argmin(shifts)
        imax = np.argmax(shifts)

        # Add the corresponding errors to the minimum and maximum peaks
        if dx == "sel":
            min_x = shifts[imin] - errs[imin]
            max_x = shifts[imax] + errs[imax]
        # Add the maximum error to the minimum and maximum peaks
        elif dx == "max":
            m = np.max(errs)
            min_x = shifts[imin] - m
            max_x = shifts[imax] + m
        # Add the mean error to the minimum and maximum peaks
        elif dx == "mean":
            m = np.mean(errs)
            min_x = shifts[imin] - m
            max_x = shifts[imax] + m
        # Add the rms error to the minimum and maximum peaks
        elif dx == "rms":
            m = np.sqrt(np.mean(np.square(errs)))
            min_x = shifts[imin] - m
            max_x = shifts[imax] + m
        else:
            raise ValueError("Unknown dx: {}".format(dx))

        # Get the limits
        lx[0] = min(lx[0], min_x)
        lx[1] = max(lx[1], max_x)

    # Get the range
    r = lx[1] - lx[0]

    # Extend the limits by a fraction of the range
    lx[0] -= extend * r
    lx[1] += extend * r

    return lx



def get_lims_2D(all_shifts, all_errs, extend=0.1, dx="rms"):
    """
    Get the limits for a predicted 1D spectrum: obtain furthest peaks ± err to determine range, extend by a factor

    Inputs:     - all_shifts    List of all shifts in the distributions
                - all_errs      List of predicted errors in the distributions
                - extend        How much to extend the range (fraction of the original range, applied to both sides)
                - dx            How the error is incorporated to the minimum/maximum shift:
                                    "sel": use the error of the minimum and maximum peaks
                                    "max": use the maximum error
                                    "mean": use the mean error
                                    "rms": use the rms error

    Outputs:    - lx            Limits in the x-dimension
                - ly            Limits in the y-dimension
    """

    # Initialize limits
    lx = np.ones(2) * np.mean(all_shifts[0][:,0])
    ly = np.ones(2) * np.mean(all_shifts[0][:,1])


    # Get minimum and maximum peak of each distribution in each dimension
    for shifts, errs in zip(all_shifts, all_errs):
        imin_x = np.argmin(shifts[:,0])
        imax_x = np.argmax(shifts[:,0])
        imin_y = np.argmin(shifts[:,1])
        imax_y = np.argmax(shifts[:,1])

        # Add the corresponding errors to the minimum and maximum peaks
        if dx == "sel":
            min_x = shifts[imin_x, 0] - errs[imin_x, 0]
            max_x = shifts[imax_x, 0] + errs[imax_x, 0]
            min_y = shifts[imin_y, 1] - errs[imin_y, 1]
            max_y = shifts[imax_y, 1] + errs[imax_y, 1]
        # Add the maximum error to the minimum and maximum peaks
        elif dx == "max":
            mx = np.max(errs[:, 0])
            my = np.max(errs[:, 1])
            min_x = shifts[imin_x, 0] - mx
            max_x = shifts[imax_x, 0] + mx
            min_y = shifts[imin_y, 1] - my
            max_y = shifts[imax_y, 1] + my
        # Add the mean error to the minimum and maximum peaks
        elif dx == "mean":
            mx = np.mean(errs[:, 0])
            my = np.mean(errs[:, 1])
            min_x = shifts[imin_x, 0] - mx
            max_x = shifts[imax_x, 0] + mx
            min_y = shifts[imin_y, 1] - my
            max_y = shifts[imax_y, 1] + my
        # Add the rms error to the minimum and maximum peaks
        elif dx == "rms":
            mx = np.sqrt(np.mean(np.square(errs[:, 0])))
            my = np.sqrt(np.mean(np.square(errs[:, 1])))
            min_x = shifts[imin_x, 0] - mx
            max_x = shifts[imax_x, 0] + mx
            min_y = shifts[imin_y, 1] - my
            max_y = shifts[imax_y, 1] + my
        else:
            raise ValueError("Unknown dx: {} (accepted values: 'sel', 'max', 'mean', 'rms')".format(dx))

        # Get the limits
        lx[0] = min(lx[0], min_x)
        lx[1] = max(lx[1], max_x)
        ly[0] = min(ly[0], min_y)
        ly[1] = max(ly[1], max_y)

        # Get the ranges
        rx = lx[1] - lx[0]
        ry = ly[1] - ly[0]

    # Extend the limits by a fraction of the ranges
    lx[0] -= extend * rx
    lx[1] += extend * rx
    ly[0] -= extend * ry
    ly[1] += extend * ry

    return np.array([lx, ly])



def make_1D_distribution(x, shifts, errs, norm=None, max_shifts=None, seed=None):
    """
    Generate 1D distribution of chemical shifts from an array of shifts and errors

    Inputs: - x             Points in the x-axis to draw the Gaussians on
            - shifts        List of shifts in the distribution
            - errs          List of predicted errors in the distribution
            - norm          Distribution normalization to apply
                                None: no normalization
                                "max": top of the distribution set to 1
            - max_shifts    Maximum number of shifts to consider when constructing the distribution
            - seed          Seed for the random selection of shifts

    Output: - y         Value of the distribution at each point of x
    """

    # Initialize y array
    y = np.zeros_like(x)

    # If there are too many shifts, randomly select a subset of length max_shifts
    if max_shifts is not None and max_shifts < len(shifts):
        if seed is not None:
            np.random.seed(seed)
            
        inds = np.random.choice(len(shifts), max_shifts, replace=False)
        
        # Add the Gaussians
        for x0, w in zip(shifts[inds], errs[inds]):
            y += 1. / (w * np.sqrt(2. * np.pi)) * np.exp(np.square(x - x0)/(-2. * np.square(w)))

    # Otherwise, use all shifts
    else:
        # Add the Gaussians
        for x0, w in zip(shifts, errs):
            y += 1. / (w * np.sqrt(2. * np.pi)) * np.exp(np.square(x - x0)/(-2. * np.square(w)))

    # Return the non-normalized sum
    if norm is None:
        return y
    # Normalize the maximum value to one
    elif norm == "max":
        return y/np.max(y)
    else:
        raise ValueError("Unknown normalization: {}".format(norm))



def make_1D_distributions(lims, n_points, all_shifts, all_errs, norm=None, max_shifts=None, seed=None):
    """
    Generate 1D distributions of chemical shifts from arrays of shifts and errors of each distribution
    
    Inputs:     - lims          Limits of the distributions
                - n_points      Number of points in the distributions
                - all_shifts    Array of shifts for each distribution
                - all_errs      Array of predicted error for each distribution
                - norm          Distribution normalization to apply
                                    None: no normalization
                                    "max": top of each distribution set to 1
                - max_shifts    Maximum number of shifts to consider when constructing the distribution
                - seed          Seed for the random selection of shifts
    
    Outputs:    - x             Array of shielding values to plot the distributions against
                - ys            List of distributions
    """
    
    # Construct the array of shielding values
    x = np.linspace(lims[0], lims[1], n_points)
    
    # Generate the distributions
    ys = []
    for i, (sh, er) in enumerate(zip(all_shifts, all_errs)):
        print("  Constructing distribution {}/{}...".format(i+1, len(all_shifts)))
        ys.append(make_1D_distribution(x, sh, er, norm=norm, max_shifts=max_shifts, seed=seed))
        print("  Distribution constructed!\n")

    return x, ys
    
    
    
def make_2D_distribution(x, y, shifts, errs, norm=None, max_shifts=None, seed=None):
    """
    
    Inputs: - x             Array of x-values to draw the Gaussians on
            - y             Array of y-values to draw the Gaussians on
            - shifts        List of shifts in the distribution
            - errs          List of predicted errors in the distribution
            - norm          Distribution normalization to apply
                                None: no normalization
                                "max": top of the distribution set to 1
            - max_shifts    Maximum number of shifts to consider when constructing the distribution
            - seed          Seed for the random selection of shifts
    
    Output: - Z             Value of the distribution at each point of the X-Y grid
    """
    
    # Initialize Z array
    Z = np.zeros((y.shape[0], x.shape[0]))
    
    # If there are too many shifts, randomly select a subset of length max_shifts
    if max_shifts is  not None and max_shifts < len(shifts):
        if seed is not None:
            np.random.seed(seed)
        
        inds = np.random.choice(len(shifts), max_shifts, replace=False)
        
        # Add the 2D Gaussians
        for [x0, y0], [wx, wy] in zip(shifts[inds], errs[inds]):
            gx = np.exp(np.square(x - x0) / (-2. * np.square(wx))) / wx
            gy = np.exp(np.square(y - y0) / (-2. * np.square(wy))) / wy
            Z += np.outer(gy, gx) / (2. * np.pi)
    
    # Otherwise, use all shifts
    else:
        # Add the 2D Gaussians
        for [x0, y0], [wx, wy] in zip(shifts, errs):
            gx = np.exp(np.square(x - x0) / (-2. * np.square(wx))) / wx
            gy = np.exp(np.square(y - y0) / (-2. * np.square(wy))) / wy
            Z += np.outer(gy, gx) / (2. * np.pi)
    
    if norm is None:
        return Z
    elif norm == "max":
        return Z / np.max(Z)
    else:
        raise ValueError("Unknown normalization: {}".format(norm))



def make_2D_distributions(lims, n_points, all_shifts, all_errs, norm=None, max_shifts=None, seed=None):
    """
    Generate 2D distributions of chemical shifts from arrays of shifts and errors of each distribution
    
    Inputs:     - lims          Limits of the distributions
                - n_points      Number of points in the distributions
                - all_shifts    Array of shifts for each distribution
                - all_errs      Array of predicted error for each distribution
                - norm          Distribution normalization to apply
                                    None: no normalization
                                    "max": top of each distribution set to 1
                - max_shifts    Maximum number of shifts to consider when constructing the distribution
                - seed          Seed for the random selection of shifts
    
    Outputs:    - X             Grid of shielding values (first dimension) to plot the distributions against
                - Y             Grid of shielding values (second dimension) to plot the distributions against
                - Zs            List of distributions
    """
    
    # Generate grid of X and Y values
    x = np.linspace(lims[0,0], lims[0,1], n_points)
    y = np.linspace(lims[1,0], lims[1,1], n_points)
    X, Y = np.meshgrid(x, y)
    
    # Generate the distributions
    Zs = []
    for i, (sh, er) in enumerate(zip(all_shifts, all_errs)):
        print("  Constructing distribution {}/{}...".format(i+1, len(all_shifts)))
        Zs.append(make_2D_distribution(x, y, sh, er, norm=norm, max_shifts=max_shifts, seed=seed))
        print("  Distribution constructed!\n")
    
    return X, Y, Zs



def get_distribution_max_1D(x, ys):
    """
    Obtain the maximum of each distribution
    
    Inputs: - x         Array of shielding values to plot the distributions against
            - ys        List of distributions
    
    Output: - centers   Maximum of each distribution
    """
    
    # Initialize array of centers
    centers = []
    
    # Get the center of each distribution (withing the set of shielding values considered)
    for i, y in enumerate(ys):
        inds = np.where(y == np.max(y))[0]
        
        if 0 in inds or (len(y) - 1) in inds:
            msg = "    WARNING: the maximum of distribution {} is at the edge of the"
            msg += " chemical shielding range! Consider expanding the range!".format(i+1)
            print(msg)
        
        centers.append(x[inds[0]])
    
    return np.array(centers)



def get_distribution_max_2D(X, Y, Zs):
    """
    Obtain the maximum of each distribution

    Inputs: - X         Grid of shielding values of the first element to plot the distribution against
            - Y         Grid of shielding values of the second element to plot the distribution against
            - Zs        List of distributions

    Output: - centers   Maximum of the distributions
    """

    #Initialize array of centers
    centers = []

    # Get the center of each distribution (within the set X and Y values)
    for i, Z in enumerate(Zs):
    
        inds_x, inds_y = np.where(Z == np.max(Z))
        
        if 0 in inds_x or 0 in inds_y or (Z.shape[0] - 1) in inds_x or (Z.shape[1] - 1) in inds_y:
            msg = "    WARNING: the maximum of distribution {} is at the edge of the"
            msg += " chemical shielding range! Consider expanding the range!".format(i+1)
            print(msg)
        
        centers.append([X[inds_x[0], inds_y[0]], Y[inds_x[0], inds_y[0]]])

    return np.array(centers)



def compute_scores_1D(exp, shifts, errs, conv, max_shifts=None, seed=None, acc=None, N=101):
    """
    Compute the scores for every possible assignment. If the variable "acc" is set to None, the probability density
    at the experimental shift yields the score. If "acc" is set to a value, the probability between
    the experimental shift e - acc and e + acc (computed as the numerical integral) is used as the score.
    
    Inputs: - exp           List of experimental shifts
            - shifts        List of shifts in each distribution
            - errs          List of errors in each distribution
            - conv          Conversion factors [slope, offset] from shielding to shift
            - max_shifts    Maximum number of shifts to select to construct the distribution
            - seed          Seed for random selection of shifts
            - acc           Accuracy of the shifts
            - N             Number of points in each shift if an accuracy is set
    
    Output: - scores    Matrix of scores for all possible assignments
    """
    
    # Initialize array of scores
    scores = np.zeros((len(shifts), len(exp)))
    
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
    for i, (sh, er) in enumerate(zip(shifts, errs)):
        print("  Evaluating distribution {}/{}...".format(i+1, len(shifts)))
        # Compute the values of the distribution on the array x
        y = make_1D_distribution(x, sh*conv[0]+conv[1], er, max_shifts=max_shifts, seed=seed)
        
        # If an accuracy is set, get the integral
        if acc is not None:
            y2 = []
            for j in range(len(exp)):
                y2.append(np.trapz(y[j*N:(j+1)*N], x=x[j*N:(j+1)*N]))
            y = np.array(y2)
        
        # Append the scores of this distribution
        if np.sum(y) < 1e-6:
            print("    WARNING: Distribution {} does not seem to match any experimental shift".format(i+1))
        scores[i] = y / np.sum(y)
        print("  Done!\n")
    
    return scores



def compute_scores_2D(exp, shifts, errs, conv_x, conv_y, max_shifts=None, seed=None, acc_x=None, acc_y=None, N=101):
    """
    Compute the scores for every possible assignment. If the variable "acc_x/acc_y" is set to None, the probability density
    at the experimental shift yields the score. If "acc_x/acc_y" are set to numerical values, the probability between
    the experimental shift e - acc and e + acc in each dimension (computed as the numerical integral) is used as the score.

    Inputs: - exp           List of experimental shifts
            - shifts        List of shifts in each distribution
            - errs          List of errors in each distribution
            - conv_x        Conversion factors [slope, offset] from shielding to shift in the x-axis
            - conv_y        Conversion factors [slope, offset] from shielding to shift in the y-axis
            - max_shifts    Maximum number of shifts to select to construct the distribution
            - seed          Seed for random selection of shifts
            - acc_x         Accuracy of the shifts in the x dimension
            - acc_y         Accuracy of the shifts in the y dimension
            - N             Number of points in each shift (along each axis) if an accuracy is set

    Output: - scores        Matrix of scores for all possible assignments
    """

    # Initialize matrix of scores
    scores = np.zeros((len(shifts), len(exp)))

    # If no accuracy is set, set x- and y-axes as the experimental shifts
    if acc_x is None and acc_y is None:
        x = np.array(exp)[:,0]
        y = np.array(exp)[:,1]
            
    # Accuracy set only in the x-axis
    elif acc_y is None:
        x = []
        for e in exp:
            x.extend(list(np.linspace(e[0]-acc_x, e[0]+acc_x, N)))
        x = np.array(x)
        y = np.array(exp)[:,1]
    
    # Accuracy set only in the y-axis
    elif acc_x is None:
        y = []
        for e in exp:
            y.extend(list(np.linspace(e[1]-acc_y, e[1]+acc_y, N)))
        y = np.array(y)
        x = np.array(exp)[:,0]
        
    # Accuracy set in both axes
    else:
        x = []
        y = []
        for e in exp:
            x.extend(list(np.linspace(e[0]-acc_x, e[0]+acc_x, N)))
            y.extend(list(np.linspace(e[1]-acc_y, e[1]+acc_y, N)))
        x = np.array(x)
        y = np.array(y)

    # Loop over all distributions
    for i, (s, er) in enumerate(zip(shifts, errs)):
    
        print("  Evaluating distribution {}/{}...".format(i+1, len(shifts)))
        
        conv_shifts = np.zeros_like(s)
        conv_shifts[:,0] = s[:,0]*conv_x[0]+conv_x[1]
        conv_shifts[:,1] = s[:,1]*conv_y[0]+conv_y[1]
    
        # If no accuracy is set, Compute the values of the distribution on the grid of experimental shifts
        if acc_x is None and acc_y is None:
            Z = make_2D_distribution(x, y, conv_shifts, er, max_shifts=max_shifts, seed=seed)
            these_scores = np.diag(Z)
        
        # If accuracy is set only along the x-axis, integrate over the range set
        elif acc_y is None:
            these_scores = np.zeros(len(exp))
            for j in range(len(exp)):
                Z = make_2D_distribution(x[j*N:(j+1)*N], y, conv_shifts, er, max_shifts=max_shifts, seed=seed)
                these_scores[j] = np.trapz(Z[j], x=x[j*N:(j+1)*N])
        
        # If accuracy is set only along the y-axis, integrate over the range set
        elif acc_x is None:
            these_scores = np.zeros(len(exp))
            for j in range(len(exp)):
                Z = make_2D_distribution(x, y[j*N:(j+1)*N], conv_shifts, er, max_shifts=max_shifts, seed=seed)
                these_scores[j] = np.trapz(Z[:, j], x=y[j*N:(j+1)*N])
        
        # If accuracy is set along both axes, integrate over the rectangle set
        else:
            these_scores = np.zeros(len(exp))
            for j in range(len(exp)):
                Z = make_2D_distribution(x[j*N:(j+1)*N], y[j*N:(j+1)*N], conv_shifts, er, max_shifts=max_shifts, seed=seed)
                these_scores[j] = np.trapz(np.trapz(Z, x=x[j*N:(j+1)*N]), x=y[j*N:(j+1)*N])
        
        if np.sum(these_scores) < 1e-6:
            print("    WARNING: Distribution {} does not seem to match any experimental shift".format(i+1))
        scores[i] = these_scores / np.sum(these_scores)
        print("  Done!\n")
    
    return scores
