####################################################################################################
###                                                                                              ###
###                             Functions for simulating spectra                                 ###
###                               Author: Manuel Cordova (EPFL)                                  ###
###                                Last modified: 10.05.2021                                     ###
###                                                                                              ###
####################################################################################################

# Import libraries
import numpy as np
import os
import pickle as pk
import networkx as nx
import tqdm.auto as tqdm

# Import local libraries
import graph as gr

# Set criterion for equivalent nodes
nm = nx.algorithms.isomorphism.categorical_node_match("elem", "X")

# Set criterion for equivalent edges
em = nx.algorithms.isomorphism.categorical_edge_match("w", -1)



def gausses(x, peaks, widths, norm=None, Nmax=1e12, seed=None):
    """
    Draw simulated spectrum from list of peaks and widths (sum of Gaussian functions)
    
    Inputs: - x         Points in the x-axis to draw the Gaussians on
            - peaks     List of shifts
            - widths    List of widths
            - norm      Whether the resulting function should be normalized w.r.t the maximum (norm="max"),
                            the integral (norm="int"), or not normalized
            - Nmax      Maximum number of shifts to draw from. If the number is smaller than the list of shifts,
                            Nmax randomly selected shifts will be used to plot the Gaussians
            - seed      Seed for the random selection of shifts

    Output: - y         Value of the sum of Gaussians at each point of x
    """
    
    # Initialize y array
    y = np.zeros_like(x)
    
    # If there are too many peaks, randomly select Nmax peaks
    if len(peaks) > Nmax:
        if seed:
            np.random.seed(seed)
        inds = np.random.choice(range(len(peaks)), int(Nmax))
        # Add the Gaussians
        for p, w in tqdm.tqdm(zip(peaks[inds], widths[inds]), total=len(inds)):
            y += 1/w*np.exp(-np.square(x-p)/(2*np.square(w)))
    
    # Otherwise, use all peaks
    else:
        # Add the Gaussians
        for p, w in tqdm.tqdm(zip(peaks, widths), total=len(peaks)):
            y += 1/w*np.exp(-np.square(x-p)/(2*np.square(w)))
    
    # Return the non-normalized sum
    if norm is None:
        return y/np.sqrt(2*np.pi)
    # Normalize the maximum value to one
    elif norm == "max":
        return y/np.max(y)
    # Normalize the integral
    elif norm == "int":
        return y/(np.sqrt(2*np.pi)*min(Nmax, len(peaks)))



def gausses_2D(x, y, peaks_x, peaks_y, widths_x, widths_y, norm=None, Nmax=1e12, seed=None):
    """
    Draw 2D simulated spectrum from list of peaks and widths (sum of correlated Gaussian functions)
    
    Inputs:     - x         Points in the x-axis to draw the Gaussians on
                - y         Points in the y-axis to draw the Gaussians on
                - peaks_x   List of shifts in the first dimension
                - peaks_y   List of shifts in the second dimension
                - widths_x  List of widths in the first dimension
                - widths_y  List of widths in the second dimension
                - norm      Whether the resulting function should be normalized w.r.t the maximum (norm="max"),
                                the integral (norm="int"), or not normalized
                - Nmax      Maximum number of shifts to draw from. If the number is smaller than the list of shifts,
                                Nmax randomly selected shifts will be used to plot the Gaussians
                - seed      Seed for the random selection of shifts

    Outputs:    - X         Grid of X values
                - Y         Grid of Y values
                - Z         Value of the sum of Gaussians at each point of X and Y
    """
    
    # Generate grid of X and Y values
    X, Y = np.meshgrid(x, y)
    
    # Initialize Z array
    Z = np.zeros_like(X)
    
    # If there are too many points, randomy select Nmax peaks
    if len(peaks_x) > Nmax:
        if seed:
            np.random.seed(seed)
        inds = np.random.choice(range(len(peaks_x)), int(Nmax))
        # Add the 2D Gaussians
        for px, py, wx, wy in tqdm.tqdm(zip(peaks_x[inds], peaks_y[inds], widths_x[inds], widths_y[inds]), total=len(inds)):
            sig = np.diag([wx**2, wy**2])
            sig_inv = np.linalg.inv(sig)
            ds = np.stack((X-px, Y-py), axis=-1)
            G = np.exp(-0.5*np.matmul(np.matmul(ds[:, :, np.newaxis, :], sig_inv), ds[..., np.newaxis]))
            G /= 2*np.pi*wx*wy
            Z += G.squeeze()
            
    # Otherwise, use all peaks
    else:
        # Add the 2D Gaussians
        for px, py, wx, wy in tqdm.tqdm(zip(peaks_x, peaks_y, widths_x, widths_y), total=len(peaks_x)):
            sig = np.diag([wx**2, wy**2])
            sig_inv = np.linalg.inv(sig)
            ds = np.stack((X-px, Y-py), axis=-1)
            G = np.exp(-0.5*np.matmul(np.matmul(ds[:, :, np.newaxis, :], sig_inv), ds[..., np.newaxis]))
            G /= 2*np.pi*wx*wy
            Z += G.squeeze()
    
    # Return the non-normalized sum
    if norm is None:
        return X, Y, Z
    # Normalize the maximum value to one
    elif norm == "max":
        return X, Y, Z/np.max(Z)
    # Normalize the integral
    elif norm == "int":
        return X, Y, Z/min(Nmax, len(peak_x))



def opt_scaling_1D(x, y):
    """
    Optimal scaling from least square regression
    
    Inputs: - x     List of experimental shifts
            - y     List of computed shifts
    
    Output: - p     Scaling parameters [slope, offset]
    """
    
    x2 = sorted(x)
    y2 = list(reversed(sorted(y)))
    return np.polyfit(x2, y2, 1)



def opt_offset_1D(x, y):
    """
    Optimal offset from least square regression
    
    Inputs: - x     List of experimental shifts
            - y     List of computed shifts
    
    Output: - p     Scaling parameters [slope, offset]
    """
    
    x2 = np.array(sorted(x))
    y2 = list(reversed(sorted(y)))
    x2 *= -1
    
    return [-1., np.mean(y-x2)]



def get_lims_1D(all_peaks, all_errs, extend=0.1, dx="rms"):
    """
    Get the limits for a predicted 1D spectrum: obtain furthest peaks ± err to determine range, extend by a factor
    
    Inputs: - all_peaks     List of all peaks in the distributions
            - all_errs      List of predicted errors in the distribution
            - extend        How much to extend the range (fraction of the original range, extended on both sides)
            - dx            How the error is incorporated to the minimum/maximum shift: either use the error of the minimum
                                and maximum peaks (dx="sel"), the maximum error (dx="max"),
                                the mean error (dx="mean"), or the rms error (dx="rms")
    
    Output: - lx            Limits in the x-dimension
    """
    
    # Initialize limits
    lx = np.ones(2) * np.mean(all_peaks[0])
    
    # Get minimum and maximum peak of each distribution
    for peaks, errs in zip(all_peaks, all_errs):
        
        # Get indices of the minimum and maximum peaks
        imin = np.argmin(peaks)
        imax = np.argmax(peaks)
        
        # Add the corresponding errors to the minimum and maximum peaks
        if dx == "sel":
            min_x = peaks[imin] - errs[imin]
            max_x = peaks[imax] + errs[imax]
        # Add the maximum error to the minimum and maximum peaks
        elif dx == "max":
            min_x = peaks[imin] - np.max(errs)
            max_x = peaks[imax] + np.max(errs)
        # Add the mean error to the minimum and maximum peaks
        elif dx == "mean":
            min_x = peaks[imin] - np.mean(errs)
            max_x = peaks[imax] + np.mean(errs)
        # Add the rms error to the minimum and maximum peaks
        elif dx == "rms":
            min_x = peaks[imin] - np.sqrt(np.mean(np.square(errs)))
            max_x = peaks[imax] + np.sqrt(np.mean(np.square(errs)))
        else:
            raise ValueError("Unknown dx: {} (accepted values: 'sel', 'max', 'mean', 'rms')".format(dx))
        
        # Get the limits
        lx[0] = min(lx[0], min_x)
        lx[1] = max(lx[1], max_x)
    
    # Get the range
    r = lx[1] - lx[0]
    
    # Extend the limits by a fraction of the range
    lx[0] -= extend * r
    lx[1] += extend * r
    
    return lx



def get_lims_2D(all_peaks, all_errs, extend=0.1, dx="rms"):
    """
    Get the limits for a predicted 1D spectrum: obtain furthest peaks ± err to determine range, extend by a factor
    
    Inputs:     - all_peaks     List of all peaks in the distributions
                - all_errs      List of predicted errors in the distribution
                - extend        How much to extend the range
                - dx            How the error is incorporated to the minimum/maximum shift: either use the error of the minimum
                                    and maximum peaks (dx="sel"), the maximum error (dx="max"),
                                    the mean error (dx="mean"), or the rms error (dx="rms")
    
    Outputs:    - lx            Limits in the x-dimension
                - ly            Limits in the y-dimension
    """
    
    # Initialize limits
    lx = np.ones(2) * np.mean(all_peaks[0][:,0])
    ly = np.ones(2) * np.mean(all_peaks[0][:,1])
    
    
    # Get minimum and maximum peak of each distribution in each dimension
    for peaks, errs in zip(all_peaks, all_errs):
        imin_x = np.argmin(peaks[:,0])
        imax_x = np.argmax(peaks[:,0])
        imin_y = np.argmin(peaks[:,1])
        imax_y = np.argmax(peaks[:,1])
        
        # Add the corresponding errors to the minimum and maximum peaks
        if dx == "sel":
            min_x = peaks[imin_x, 0] - errs[imin_x, 0]
            max_x = peaks[imax_x, 0] + errs[imax_x, 0]
            min_y = peaks[imin_y, 1] - errs[imin_y, 1]
            max_y = peaks[imax_y, 1] + errs[imax_y, 1]
        # Add the maximum error to the minimum and maximum peaks
        elif dx == "max":
            min_x = peaks[imin_x, 0] - np.max(errs[:, 0])
            max_x = peaks[imax_x, 0] + np.max(errs[:, 0])
            min_y = peaks[imin_y, 1] - np.max(errs[:, 1])
            max_y = peaks[imax_y, 1] + np.max(errs[:, 1])
        # Add the mean error to the minimum and maximum peaks
        elif dx == "mean":
            min_x = peaks[imin_x, 0] - np.mean(errs[:, 0])
            max_x = peaks[imax_x, 0] + np.mean(errs[:, 0])
            min_y = peaks[imin_y, 1] - np.mean(errs[:, 1])
            max_y = peaks[imax_y, 1] + np.mean(errs[:, 1])
        # Add the rms error to the minimum and maximum peaks
        elif dx == "rms":
            min_x = peaks[imin_x, 0] - np.sqrt(np.mean(np.square(errs[:, 0])))
            max_x = peaks[imax_x, 0] + np.sqrt(np.mean(np.square(errs[:, 0])))
            min_y = peaks[imin_y, 1] - np.sqrt(np.mean(np.square(errs[:, 1])))
            max_y = peaks[imax_y, 1] + np.sqrt(np.mean(np.square(errs[:, 1])))
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



def cleanup_methyls(labels, Gs, shifts, errs, ws, crysts, inds, nei_inds, nei_atoms, envs):
    """
    Gather methyl proton shifts
    
    Inputs:     - labels        List of labels for each distribution
                - Gs            Graph of each distribution
                - shifts        Predicted shifts in each distribution
                - errs          Prediction errors in each distribution
                - ws            Maximum depth for each distribution
                - crysts        Crystals in each distribution
                - inds          Indices of the atoms in each distribution
                - nei_inds      Neighbouring indices for all atoms in the structure
                - nei_atoms     Neighbouring elements for all atoms in the structure
                - envs          Environment of each atom in the structure
    
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
    new_shifts = []
    new_errs = []
    new_ws = []
    new_crysts = []
    new_inds = []
    new_envs = []
    
    # Initialize array of methyls already identified (or NH3, ...)
    already_methyls = {"H":[], "C":[], "N":[], "O":[]}
    
    # Loop over all distributions/graphs
    for l, G, sh, er, w, cryst, ind, nei_ind, nei_atom in zip(labels, Gs, shifts, errs, ws, crysts, inds, nei_inds["H"], nei_atoms["H"]):
        # Check that there is only one neighbour (H should be linked to only one atom)
        if len(nei_ind) == 1 and len(nei_atom) == 1:
            e = nei_atom[0]
            i = nei_ind[0]
            # Get the environment of the neighbouring atom
            tmp = envs[e][i].split("-")
            # If there are at least three Hs, consider it as a "methyl"
            if tmp.count("H") >= 3:
                # If the attached element is not already known, keep this distribution. Otherwise, discard it
                if i not in already_methyls[e]:
                    already_methyls[e].append(i)
                    new_labels.append(l)
                    new_Gs.append(G)
                    new_shifts.append(sh)
                    new_errs.append(er)
                    new_ws.append(w)
                    new_crysts.append(cryst)
                    new_inds.append(ind)
            
            # If there is less than three Hs, keep the distribution
            else:
                new_labels.append(l)
                new_Gs.append(G)
                new_shifts.append(sh)
                new_errs.append(er)
                new_ws.append(w)
                new_crysts.append(cryst)
                new_inds.append(ind)
                
        # If the H has more than one neighbour, keep the distribution
        else:
            new_labels.append(l)
            new_Gs.append(G)
            new_shifts.append(sh)
            new_errs.append(er)
            new_ws.append(w)
            new_crysts.append(cryst)
            new_inds.append(ind)
    
    return new_labels, new_Gs, new_shifts, new_errs, new_ws, new_crysts, new_inds



def cleanup_nei_methyls(elem, nei_elem, labels, all_shifts, all_errs, ws, all_crysts, all_inds, envs):
    """
    Gather methyl 2D shifts
    
    Inputs:     - elem          Central element
                - nei_elem      Neighbour element
                - labels        Labels of the shifts
                - all_shifts    List of predicted shifts 
                - all_errs      List of predicted errors
                - ws            List of weights of the graphs
                - all_crysts    Crystals in each distribution
                - all_inds      Indices of the atoms in each distribution
                - envs          Environment of each atom in the structure

    Outputs:    - new_labels    Updated list of labels for the shifts
                - new_shifts    Updated list of predicted shifts
                - new_errs      Updated list of predicted errors
                - new_ws        Updated list of weights of the graphs
                - new_crysts    Updated list of crystals in each distribution
                - new_inds      Updated list of indices of the atoms in each distribution
    """
    
    # Initialize the updated lists
    new_labels = []
    new_shifts = []
    new_errs = []
    new_ws = []
    new_crysts = []
    new_inds = []
    
    # Array to store already identified methyl groups
    already_methyls = []
    
    # Loop over all distributions
    for l, sh, er, w, cr, ind in zip(labels, all_shifts, all_errs, ws, all_crysts, all_inds):
        
        # Get the central and neighbour indices
        c_ind = int(l.split("-")[0].split(elem)[1])-1
        n_ind = int(l.split("-")[1].split(nei_elem)[1])-1
        
        # Check whether the central atom has at least 3 H bonded to it
        if envs[elem][c_ind].split("-").count("H") >= 3:
            # Append the distribution to the updated list
            if c_ind not in already_methyls:
                already_methyls.append(c_ind)
                new_labels.append(l)
                new_shifts.append(sh)
                new_errs.append(er)
                new_ws.append(w)
                new_crysts.append(cr)
                new_inds.append(ind)
        
        # If no methyl is identified, apped the distribution to the updated list
        else:
            new_labels.append(l)
            new_shifts.append(sh)
            new_errs.append(er)
            new_ws.append(w)
            new_crysts.append(cr) 
            new_inds.append(ind)  
    
    return new_labels, new_shifts, new_errs, new_ws, new_crysts, new_inds



def cleanup_equivalent(labels, all_shifts, all_errs, ws, all_crysts, all_inds):
    """
    Gather equivalent graphs (identified by their shift distributions)
    
    Inputs:     - labels        List of labels of the distributions
                - all_shifts    List of predicted shifts in each distribution
                - all_errs      List of predicted errors in each distribution
                - ws            List of weights of the distributions
                - all_crysts    List of crystals in each distribution
                - all_inds      List of the atoms in each distribution
                
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
    
    # Loop over all the distributions
    for l, sh, er, w, cr, ind in zip(labels, all_shifts, all_errs, ws, all_crysts, all_inds):
        
        # Check if the distribution is already found
        already = False
        for i, sh2 in enumerate(new_shifts):
            if sh.shape == sh2.shape and np.all(sh == sh2):
                already = True
                break
        
        # If the distribution is already found, modify the label
        if already:
            new_labels[i] += "/{}".format(l)
            
        # Otherwise, append the new distribution to the updated list
        else:
            new_labels.append(l)
            new_shifts.append(sh)
            new_errs.append(er)
            new_ws.append(w)
            new_crysts.append(cr)
            new_inds.append(ind)
    
    return new_labels, new_shifts, new_errs, new_ws, new_crysts, new_inds
