####################################################################################################
###                                                                                              ###
###                      Functions for database handling (fetching, ...)                         ###
###                               Author: Manuel Cordova (EPFL)                                  ###
###                                Last modified: 10.05.2021                                     ###
###                                                                                              ###
####################################################################################################

# Import libraries
import numpy as np
import os
import sys
import pickle as pk
import networkx as nx
import subprocess as sbp
import time

# Import local libraries
import graph as gr



def fetch_entries(db_root, elem, envs, Gs, w_max, N_min=100, nei_elem=None, exclude=None, verbose=False):
    """
    Find the database entries corresponding to each graph, with a minimum number of instances.
        Also retrieve the crystal identifier and index of the atom associated with each database entry.
    
    Inputs:     - db_root       Root directory of the database
                - elem          Element of the central nodes of the graphs
                - envs          Environment of each graph (first coordination shell)
                - Gs            List of graphs to fetch the database for
                - w_max         Maximum depth
                - N_min         Minimum number of entries for each graph
                - nei_elem      "None" if we only want to retrieve the shifts of the central atom,
                                    otherwise element of the neighbour to extract shift distributions from
                - exclude       List of crystal identifiers to exclude
                - verbose       Whether additional information about the search should be printed
                
    Outputs:    - all_shifts    List of predicted shifts for each graph
                - all_errs      List of prediction errors for each graph
                - ws            List of maximum depth for each graph
                - all_crysts    List of the crystals corresponding to the shifts extracted
                - all_inds      List of the indices of the atoms corresponding to the shifts extracted
    """
    
    # Initialize arrays
    all_shifts = []
    all_errs = []
    ws = []
    labels = []
    all_crysts = []
    all_inds = []
    
    already_hashes = []
    
    # Get the directory
    elem_dir = db_root + elem + "/"
    
    # Loop over each graph
    for i, (G, env) in enumerate(zip(Gs, envs[elem])):
        
        start = time.time()
        
        # Get the number of neighbouring elements in the environment
        num_nei = 0
        if nei_elem is not None:
            nei_elems = env.split("-")
            num_nei = nei_elems.count(nei_elem)
            # Get the directory
            elem_dir = db_root + elem + "-" + nei_elem + "/"
        
        if not os.path.exists(elem_dir):
            raise ValueError("Directory does not exist: {}".format(elem_dir))
        
        # If there are neighbours that correspond to the element, extract the 2D shifts
        if num_nei > 0:
            
            if not os.path.exists(elem_dir + env + ".csv"):
                raise ValueError("File does not exist: {}".format(elem_dir + env + ".csv"))
            
            # Loop over all neighbours
            for j in range(1, len(nei_elems)+1):
                
                if G.nodes[j]["elem"] == nei_elem:
                    
                    this_w = w_max

                    # Generate arborescence (array of hashes)
                    arb = []
                    for w in range(2, w_max+1):
                        cut_G = gr.cut_graph(G, w)
                        cut_G.nodes[j]["elem"] = "Z"
                        arb.append(gr.generate_hash(cut_G))
                    
                    # If the arborescence was already found before, get the corresponding shifts directly
                    if ",".join(arb) in already_hashes:
                        
                        h_ind = already_hashes.index(",".join(arb))
                        already_hashes.append(",".join(arb))
                        
                        this_w = ws[h_ind]
                        
                        # Append the array of shifts and errors for this distribution
                        all_shifts.append(all_shifts[h_ind])
                        all_errs.append(all_errs[h_ind])
                        ws.append(this_w)
                        labels.append("{}{}-{}{}".format(elem, i+1, nei_elem, G.nodes[j]["ind"]+1))
                        all_inds.append(all_inds[h_ind])
                        all_crysts.append(all_crysts[h_ind])
                        
                    # Otherwise, search through the database
                    else:
                        
                        already_hashes.append(",".join(arb))
                    
                        # Initialize array of shifts and errors
                        shifts = []
                        errs = []
                        inds = []
                        crysts = []

                        # Get the entries of the corresponding graph
                        p = sbp.Popen(["grep", ",".join(arb), elem_dir + env + ".csv"], stdout=sbp.PIPE)
                        out, err = p.communicate()
                        out = out.decode("UTF-8")

                        # Extract the shifts
                        for l in out.split("\n"):
                            if len(l) > 0:
                                tmp = l.split(",")
                                if (exclude is None or tmp[0] not in exclude) and tmp[0] != "crystal":
                                    shifts.append([float(tmp[2]), float(tmp[5])])
                                    errs.append([float(tmp[3]), float(tmp[6])])
                                    inds.append([int(tmp[1]), int(tmp[4])])
                                    crysts.append(tmp[0])

                        # If there is not enough entries, reduce the depth and try again
                        while len(shifts) < N_min:
                            
                            if verbose:
                                print("  w = {}: {} instances".format(this_w, len(shifts)))

                            shifts = []
                            errs = []
                            inds = []
                            crysts = []

                            # Update the depth and the corresponding arborescence
                            this_w -= 1
                            arb = arb[:-1]

                            # Get the entries of the corresponding graph
                            p = sbp.Popen(["grep", ",".join(arb), elem_dir + env + ".csv"], stdout=sbp.PIPE)
                            out, err = p.communicate()
                            out = out.decode("UTF-8")

                            # Extract the shifts
                            for l in out.split("\n"):
                                if len(l) > 0:
                                    tmp = l.split(",")
                                    if (exclude is None or tmp[0] not in exclude) and tmp[0] != "crystal":
                                        shifts.append([float(tmp[2]), float(tmp[5])])
                                        errs.append([float(tmp[3]), float(tmp[6])])
                                        inds.append([int(tmp[1]), int(tmp[4])])
                                        crysts.append(tmp[0])

                        # Append the array of shifts and errors for this distribution
                        all_shifts.append(np.array(shifts))
                        all_errs.append(np.array(errs))
                        ws.append(this_w)
                        labels.append("{}{}-{}{}".format(elem, i+1, nei_elem, G.nodes[j]["ind"]+1))
                        all_inds.append(inds)
                        all_crysts.append(crysts)
            
            stop = time.time()
            print("Graph {}/{} found. w = {}, {} instances. Time elapsed: {:.2f} s".format(i+1, len(Gs), this_w, len(shifts), stop-start))
        
        # If the neighbouring element is not set, extract the 1D shfits
        elif nei_elem is None:
            
            this_w = w_max
        
            # Generate arborescence (array of hashes)
            arb = []
            for w in range(2, w_max+1):
                cut_G = gr.cut_graph(G, w)
                arb.append(gr.generate_hash(cut_G))
                    
            if ",".join(arb) in already_hashes:
                h_ind = already_hashes.index(",".join(arb))
                already_hashes.append(",".join(arb))
                        
                this_w = ws[h_ind]

                # Append the array of shifts and errors for this distribution
                all_shifts.append(all_shifts[h_ind])
                all_errs.append(all_errs[h_ind])
                ws.append(this_w)
                labels.append("{}{}".format(elem, i+1))
                all_inds.append(all_inds[h_ind])
                all_crysts.append(all_crysts[h_ind])
            
            else:
                        
                already_hashes.append(",".join(arb))

                # Initialize array of shifts and errors
                shifts = []
                errs = []
                inds = []
                crysts = []

                # Get the entries of the corresponding graph
                p = sbp.Popen(["grep", ",".join(arb), elem_dir + env + ".csv"], stdout=sbp.PIPE)
                out, err = p.communicate()
                out = out.decode("UTF-8")

                # Extract the shifts
                for l in out.split("\n"):
                    if len(l) > 0:
                        tmp = l.split(",")
                        if (exclude is None or tmp[0] not in exclude) and tmp[0] != "crystal":
                            shifts.append(float(tmp[2]))
                            errs.append(float(tmp[3]))
                            inds.append(int(tmp[1]))
                            crysts.append(tmp[0])

                # If there is not enough entries, reduce the depth and try again
                while len(shifts) < N_min:
                            
                    if verbose:
                        print("  w = {}: {} instances".format(this_w, len(shifts)))

                    shifts = []
                    errs = []
                    inds = []
                    crysts = []

                    # Update the depth and the corresponding arborescence
                    this_w -= 1
                    arb = arb[:-1]

                    # Get the entries of the corresponding graph
                    p = sbp.Popen(["grep", ",".join(arb), elem_dir + env + ".csv"], stdout=sbp.PIPE)
                    out, err = p.communicate()
                    out = out.decode("UTF-8")

                    # Extract the shifts
                    for l in out.split("\n"):
                        if len(l) > 0:
                            tmp = l.split(",")
                            if (exclude is  None or tmp[0] not in exclude) and tmp[0] != "crystal":
                                shifts.append(float(tmp[2]))
                                errs.append(float(tmp[3]))
                                inds.append(int(tmp[1]))
                                crysts.append(tmp[0])

                # Append the array of shifts and error for this distribution
                all_shifts.append(np.array(shifts))
                all_errs.append(np.array(errs))
                ws.append(this_w)
                labels.append("{}{}".format(elem, i+1))
                all_inds.append(inds)
                all_crysts.append(crysts)
            
            stop = time.time()
            print("Graph {}/{} found. w = {}, {} instances. Time elapsed: {:.2f} s".format(i+1, len(Gs), this_w, len(shifts), stop-start))
                                                                                               
    return all_shifts, all_errs, ws, labels, all_crysts, all_inds
    
    
    
def fetch_entries_from_hashes(db_root, elem, envs, Hs, w_max, N_min=100, nei_elem=None, exclude=None, verbose=False):
    """
    Find the database entries corresponding to each graph, with a minimum number of instances.
        Also retrieve the crystal identifier and index of the atom associated with each database entry.

    Inputs:     - db_root       Root directory of the database
                - elem          Element of the central nodes of the graphs
                - envs          Environment of each graph (first coordination shell)
                - Hs            List of graphs hashes to fetch the database for
                - w_max         Maximum depth
                - N_min         Minimum number of entries for each graph
                - nei_elem      "None" if we only want to retrieve the shifts of the central atom,
                                    otherwise element of the neighbour to extract shift distributions from
                - exclude       List of crystal identifiers to exclude
                - verbose       Whether additional information about the search should be printed
                
    Outputs:    - all_shifts    List of predicted shifts for each graph
                - all_errs      List of prediction errors for each graph
                - ws            List of maximum depth for each graph
                - all_crysts    List of the crystals corresponding to the shifts extracted
                - all_inds      List of the indices of the atoms corresponding to the shifts extracted
    """

    # Initialize arrays
    all_shifts = []
    all_errs = []
    ws = []
    labels = []
    all_crysts = []
    all_inds = []

    already_hashes = []

    # Get the directory
    elem_dir = db_root + elem + "/"

    # Loop over each graph
    for i, (H, env) in enumerate(zip(Hs, envs)):
        
        start = time.time()
        
        # Get the number of neighbouring elements in the environment
        num_nei = 0
        if nei_elem is not None:
            nei_elems = env.split("-")
            num_nei = nei_elems.count(nei_elem)
            # Get the directory
            elem_dir = db_root + elem + "-" + nei_elem + "/"
        
        if not os.path.exists(elem_dir):
            raise ValueError("Directory does not exist: {}".format(elem_dir))
        
        # If there are neighbours that correspond to the element, extract the 2D shifts
        if num_nei > 0:
            
            if not os.path.exists(elem_dir + env + ".csv"):
                raise ValueError("File does not exist: {}".format(elem_dir + env + ".csv"))
                
            this_w = w_max
            
            arb = H.split(",")
            
            # If the arborescence was already found before, get the corresponding shifts directly
            if ",".join(arb) in already_hashes:
                
                h_ind = already_hashes.index(",".join(arb))
                already_hashes.append(",".join(arb))
                
                this_w = ws[h_ind]
                
                # Append the array of shifts and errors for this distribution
                all_shifts.append(all_shifts[h_ind])
                all_errs.append(all_errs[h_ind])
                ws.append(this_w)
                labels.append("{}{}-{}{}".format(elem, i+1, nei_elem, 0))
                all_inds.append(all_inds[h_ind])
                all_crysts.append(all_crysts[h_ind])
                
            # Otherwise, search through the database
            else:
                
                already_hashes.append(",".join(arb))
            
                # Initialize array of shifts and errors
                shifts = []
                errs = []
                inds = []
                crysts = []

                # Get the entries of the corresponding graph
                p = sbp.Popen(["grep", ",".join(arb), elem_dir + env + ".csv"], stdout=sbp.PIPE)
                out, err = p.communicate()
                out = out.decode("UTF-8")

                # Extract the shifts
                for l in out.split("\n"):
                    if len(l) > 0:
                        tmp = l.split(",")
                        if (exclude is None or tmp[0] not in exclude) and tmp[0] != "crystal":
                            shifts.append([float(tmp[2]), float(tmp[5])])
                            errs.append([float(tmp[3]), float(tmp[6])])
                            inds.append([int(tmp[1]), int(tmp[4])])
                            crysts.append(tmp[0])

                # If there is not enough entries, reduce the depth and try again
                while len(shifts) < N_min:
                    
                    if verbose:
                        print("  w = {}: {} instances".format(this_w, len(shifts)))

                    shifts = []
                    errs = []
                    inds = []
                    crysts = []

                    # Update the depth and the corresponding arborescence
                    this_w -= 1
                    arb = arb[:-1]

                    # Get the entries of the corresponding graph
                    p = sbp.Popen(["grep", ",".join(arb), elem_dir + env + ".csv"], stdout=sbp.PIPE)
                    out, err = p.communicate()
                    out = out.decode("UTF-8")

                    # Extract the shifts
                    for l in out.split("\n"):
                        if len(l) > 0:
                            tmp = l.split(",")
                            if (exclude is None or tmp[0] not in exclude) and tmp[0] != "crystal":
                                shifts.append([float(tmp[2]), float(tmp[5])])
                                errs.append([float(tmp[3]), float(tmp[6])])
                                inds.append([int(tmp[1]), int(tmp[4])])
                                crysts.append(tmp[0])

                # Append the array of shifts and errors for this distribution
                all_shifts.append(np.array(shifts))
                all_errs.append(np.array(errs))
                ws.append(this_w)
                labels.append("{}{}-{}{}".format(elem, i+1, nei_elem, 0))
                all_inds.append(inds)
                all_crysts.append(crysts)
            
            stop = time.time()
            print("Graph {}/{} found. w = {}, {} instances. Time elapsed: {:.2f} s".format(i+1, len(Hs), this_w, len(shifts), stop-start))
        
        # If the neighbouring element is not set, extract the 1D shfits
        elif nei_elem is None:
            
            this_w = w_max
        
            # Generate arborescence (array of hashes)
            arb = H.split(",")
                    
            if ",".join(arb) in already_hashes:
                h_ind = already_hashes.index(",".join(arb))
                already_hashes.append(",".join(arb))
                        
                this_w = ws[h_ind]

                # Append the array of shifts and errors for this distribution
                all_shifts.append(all_shifts[h_ind])
                all_errs.append(all_errs[h_ind])
                ws.append(this_w)
                labels.append("{}{}".format(elem, i+1))
                all_inds.append(all_inds[h_ind])
                all_crysts.append(all_crysts[h_ind])
            
            else:
                        
                already_hashes.append(",".join(arb))

                # Initialize array of shifts and errors
                shifts = []
                errs = []
                inds = []
                crysts = []

                # Get the entries of the corresponding graph
                p = sbp.Popen(["grep", ",".join(arb), elem_dir + env + ".csv"], stdout=sbp.PIPE)
                out, err = p.communicate()
                out = out.decode("UTF-8")

                # Extract the shifts
                for l in out.split("\n"):
                    if len(l) > 0:
                        tmp = l.split(",")
                        if (exclude is None or tmp[0] not in exclude) and tmp[0] != "crystal":
                            shifts.append(float(tmp[2]))
                            errs.append(float(tmp[3]))
                            inds.append(int(tmp[1]))
                            crysts.append(tmp[0])

                # If there is not enough entries, reduce the depth and try again
                while len(shifts) < N_min:
                            
                    if verbose:
                        print("  w = {}: {} instances".format(this_w, len(shifts)))

                    shifts = []
                    errs = []
                    inds = []
                    crysts = []

                    # Update the depth and the corresponding arborescence
                    this_w -= 1
                    arb = arb[:-1]

                    # Get the entries of the corresponding graph
                    p = sbp.Popen(["grep", ",".join(arb), elem_dir + env + ".csv"], stdout=sbp.PIPE)
                    out, err = p.communicate()
                    out = out.decode("UTF-8")

                    # Extract the shifts
                    for l in out.split("\n"):
                        if len(l) > 0:
                            tmp = l.split(",")
                            if (exclude is  None or tmp[0] not in exclude) and tmp[0] != "crystal":
                                shifts.append(float(tmp[2]))
                                errs.append(float(tmp[3]))
                                inds.append(int(tmp[1]))
                                crysts.append(tmp[0])

                # Append the array of shifts and error for this distribution
                all_shifts.append(np.array(shifts))
                all_errs.append(np.array(errs))
                ws.append(this_w)
                labels.append("{}{}".format(elem, i+1))
                all_inds.append(inds)
                all_crysts.append(crysts)
            
            stop = time.time()
            print("Graph {}/{} found. w = {}, {} instances. Time elapsed: {:.2f} s".format(i+1, len(Hs), this_w, len(shifts), stop-start))
                                                                                               
    return all_shifts, all_errs, ws, labels, all_crysts, all_inds
