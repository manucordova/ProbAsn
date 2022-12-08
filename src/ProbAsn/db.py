####################################################################################################
###                                                                                              ###
###                      Functions for database handling (fetching, ...)                         ###
###                               Author: Manuel Cordova (EPFL)                                  ###
###                                Last modified: 03.09.2021                                     ###
###                                                                                              ###
####################################################################################################

# Import libraries
import numpy as np
import os
import sqlite3 as sl
import time

# Import local libraries
from . import graph as gr



def fetch_entries(db_file, elem, atoms, envs, Gs, max_w, N_min=10, nei_elem=None, exclude=None, verbose=False):
    """
    Find the database entries corresponding to each graph, with a minimum number of instances.
        Also retrieve the crystal identifier and index of the atom associated with each database entry.
    
    Inputs:     - db_file       Database file
                - elem          Element of the central nodes of the graphs
                - elems         List of atoms in the molecule
                - envs          Environment of each graph (first coordination shell)
                - Gs            List of graphs to fetch the database for
                - max_w         Maximum depth
                - N_min         Minimum number of entries in the database required
                - nei_elem      "None" if we only want to retrieve the shifts of the central atom,
                                    otherwise element of the neighbour to extract shift distributions from
                - exclude       List of crystal identifiers to exclude
                - verbose       Whether additional information about the search should be printed
                
    Outputs:    - all_shifts    List of predicted shifts for each graph
                - all_errs      List of prediction errors for each graph
                - ws            List of maximum depth for each graph
                - labels        List of graph labels
                - all_crysts    List of the crystals corresponding to the shifts extracted
                - all_inds      List of the indices of the atoms corresponding to the shifts extracted
                - hashes        List of hashes identifying the graphs
    """
    
    # Initialize arrays
    all_shifts = []
    all_errs = []
    ws = []
    labels = []
    all_crysts = []
    all_inds = []
    hashes = []
    
    # Check if database directory exists
    if not os.path.exists(db_file):
        raise ValueError(f"Database does not exist: {db_file}")
    # Initialize connection to the database
    con = sl.connect(db_file)
    
    # Loop over each graph
    for i, (G, env) in enumerate(zip(Gs, envs)):
        
        start = time.time()
        
        # Get the number of neighbouring elements in the environment
        num_nei = 0
        if nei_elem is not None:
            nei_elems = env.split("-")
            num_nei = nei_elems.count(nei_elem)

        # If there are neighbours that correspond to the element, extract the 2D shifts
        if num_nei > 0:
            
            # Loop over all neighbours
            for j in range(1, len(nei_elems)+1):
                
                if G.nodes[j]["elem"] == nei_elem:
                    
                    this_w = max_w

                    # Generate arborescence (array of hashes)
                    where = [f"env = '{env}'"]
                    arb = []
                    for w in range(2, max_w+1):
                        cut_G = gr.cut_graph(G, w)
                        cut_G.nodes[j]["elem"] = "Z"
                        arb.append(gr.generate_hash(cut_G))
                        where.append(f"G{w} = '{arb[-1]}'")
                    where.append("")
                    
                    # If the arborescence was already found before, get the corresponding shifts directly
                    if ",".join(arb) in hashes:
                        
                        h_ind = hashes.index(",".join(arb))
                        hashes.append(",".join(arb))
                        
                        this_w = ws[h_ind]
                        
                        # Append the array of shifts and errors for this distribution
                        all_shifts.append(all_shifts[h_ind])
                        all_errs.append(all_errs[h_ind])
                        ws.append(this_w)
                        labels.append("{}{}-{}{}".format(elem, i+1, nei_elem, atoms[:G.nodes[j]["ind"]].count(nei_elem)+1))
                        all_inds.append(all_inds[h_ind])
                        all_crysts.append(all_crysts[h_ind])
                        
                    # Otherwise, search through the database
                    else:
                        
                        hashes.append(",".join(arb))
                    
                        # Get the entries of the corresponding graph
                        while len(where) > 0:
                            where.pop(-1)
                            # If we run out of options, just return matching environments
                            if len(where) == 0:
                                with con:
                                    data = con.execute(
                                        f"""
                                        SELECT crystal, ind, shift, err, nei_ind, nei_shift, nei_err FROM {elem}_{nei_elem};
                                        """
                                    ).fetchall()
                                break
                            
                            with con:
                                data = con.execute(
                                    f"""
                                    SELECT crystal, ind, shift, err, nei_ind, nei_shift, nei_err FROM {elem}_{nei_elem} WHERE {' AND '.join(where)};
                                    """
                                ).fetchall()
                            
                            if len(data) >= N_min:
                                break

                            if verbose:
                                print("    w = {}: {} instances are not enough, reducing graph depth...".format(this_w, len(data)))

                            this_w -= 1

                        # Set arrays of shifts, errors, crystal structures and atomic indices
                        shifts = []
                        errs = []
                        crysts = []
                        inds = []
                        for cryst, ind, shift, err, nei_ind, nei_shift, nei_err in data:
                            if exclude is None or cryst not in exclude:
                                crysts.append(cryst)
                                inds.append([ind, nei_ind])
                                shifts.append([shift, nei_shift])
                                errs.append([err, nei_err])
                        
                        # Append the array of shifts and errors for this distribution
                        all_shifts.append(np.array(shifts))
                        all_errs.append(np.array(errs))
                        ws.append(this_w)
                        labels.append("{}{}-{}{}".format(elem, i+1, nei_elem, atoms[:G.nodes[j]["ind"]].count(nei_elem)+1))
                        all_inds.append(inds)
                        all_crysts.append(crysts)
            
            stop = time.time()
            print("  Graph {}/{} found. w = {}, {} instances. Time elapsed: {:.2f} s".format(i+1, len(Gs), this_w, len(all_shifts[-1]), stop-start))
        
        # If the neighbouring element is not set, extract the 1D shfits
        elif nei_elem is None:
            this_w = max_w
        
            # Generate arborescence (array of hashes)
            where = [f"env = '{env}'"]
            arb = []
            for w in range(2, max_w+1):
                cut_G = gr.cut_graph(G, w)
                arb.append(gr.generate_hash(cut_G))
                where.append(f"G{w} = '{arb[-1]}'")
            where.append("")
            
            # If the arborescence was already found, reuse the previously extracted shifts to save time
            if ",".join(arb) in hashes:
            
                h_ind = hashes.index(",".join(arb))
                hashes.append(",".join(arb))
                        
                this_w = ws[h_ind]

                # Append the array of shifts and errors for this distribution
                all_shifts.append(all_shifts[h_ind])
                all_errs.append(all_errs[h_ind])
                ws.append(this_w)
                labels.append("{}{}".format(elem, i+1))
                all_inds.append(all_inds[h_ind])
                all_crysts.append(all_crysts[h_ind])
            
            else:
                hashes.append(",".join(arb))

                # Get the entries of the corresponding graph
                while len(where) > 0:
                    where.pop(-1)
                    # If we run out of options, just return matching environments
                    if len(where) == 0:
                        with con:
                            data = con.execute(
                                f"""
                                SELECT crystal, ind, shift, err FROM {elem};
                                """
                            ).fetchall()
                        break
                    
                    with con:
                        data = con.execute(
                            f"""
                            SELECT crystal, ind, shift, err FROM {elem} WHERE {' AND '.join(where)};
                            """
                        ).fetchall()
                    
                    if len(data) >= N_min:
                        break

                    if verbose:
                        print("    w = {}: {} instances are not enough, reducing graph depth...".format(this_w, len(data)))

                    this_w -= 1

                # Set arrays of shifts, errors, crystal structures and atomic indices
                shifts = []
                errs = []
                crysts = []
                inds = []
                for cryst, ind, shift, err in data:
                    if exclude is None or cryst not in exclude:
                        crysts.append(cryst)
                        inds.append(ind)
                        shifts.append(shift)
                        errs.append(err)
          
                # Append the array of shifts and error for this distribution
                all_shifts.append(np.array(shifts))
                all_errs.append(np.array(errs))
                ws.append(this_w)
                labels.append("{}{}".format(elem, i+1))
                all_inds.append(inds)
                all_crysts.append(crysts)
            
            stop = time.time()
            print("  Graph {}/{} found. w = {}, {} instances. Time elapsed: {:.2f} s".format(i+1, len(Gs), this_w, len(all_shifts[-1]), stop-start))
               
        else:
            print("  Graph {}/{} has no neighbouring {}.".format(i+1, len(Gs), nei_elem))
    
    return all_shifts, all_errs, ws, labels, all_crysts, all_inds, hashes
