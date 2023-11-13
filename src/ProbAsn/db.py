###############################################################################
#                                                                             #
#               Functions for database handling (fetching, ...)               #
#                        Author: Manuel Cordova (EPFL)                        #
#                          Last modified: 22.09.2023                          #
#                                                                             #
###############################################################################

# Import libraries
import numpy as np
import sqlite3 as sl
import time

# Import local libraries
from . import graph as gr


def fetch_entries(
    db_dir,
    elem,
    atoms,
    envs,
    Gs,
    max_w=6,
    n_min=10,
    nei_elem=None,
    exclude=None,
    verbose=False
):
    """Find the database entries corresponding to each graph.

    Parameters
    ----------
    db_dir : str
        Path to the directory containing the database.
    elem : str
        Element of the central nodes of the graphs.
    atoms : list
        List of atoms in the molecule.
    envs : list
        List of environments corresponding to each graph.
    Gs : list
        List of graphs to fetch the database for.
    max_w : int, default=6
        Maximum graph depth.
    n_min : int
        Minimum number of entries in the database requested.
    nei_elem : None or str, default=None
        `None` if we only want to retrieve the shift of the central atom,
        Otherwise element of the neighbour
        to extract the shift distributions from.
    exclude : None or list, default=None
        List of crystal identifiers to exclude.
    verbose : bool, default=False
        Print additional information about the search.

    Returns
    -------
    all_shifts : list
        List of shifts for each graph.
    all_errs : list
        List of errors on the shift for each graph.
    ws : list
        List of depths for each graph.
    labels : list
        List of labels corresponding to each graph.
    all_crysts : list
        List of crystal identifiers corresponding
        to the shifts extracted for each graph.
    all_inds : list
        List of the indices of the atoms
        corresponding to the shifts extracted for each graph.
    hashes : list
        List of Weisfeiler-Lehman hashes corresponding to each graph.
    """

    # Initialize arrays
    all_shifts = []
    all_errs = []
    ws = []
    labels = []
    all_crysts = []
    all_inds = []
    hashes = []

    # Select database to search
    if nei_elem is None:
        con = sl.connect(f"{db_dir}ProbAsn_{elem}.db")
    else:
        con = sl.connect(f"{db_dir}ProbAsn_{elem}-{nei_elem}.db")

    # Loop over each graph
    for i, (G, env) in enumerate(zip(Gs, envs)):

        start = time.time()

        # Get the number of neighbouring elements in the environment
        num_nei = 0
        if nei_elem is not None:
            nei_elems = env.split("-")
            num_nei = nei_elems.count(nei_elem)

        # If there are neighbours that correspond to the element,
        # extract the 2D shifts
        if num_nei > 0:

            # Loop over all neighbours
            for j in range(1, len(nei_elems)+1):

                if G.nodes[j]["elem"] == nei_elem:

                    nei_i = atoms[:G.nodes[j]["ind"]].count(nei_elem)+1
                    this_w = max_w

                # Generate arborescence (array of hashes) and SQL conditions
                    where = [f"env = '{env}'"]
                    arb = []
                    for w in range(2, max_w+1):
                        cut_G = gr.cut_graph(G, w)
                        cut_G.nodes[j]["elem"] = "Z"
                        arb.append(gr.generate_hash(cut_G))
                        where.append(f"G{w} = '{arb[-1]}'")
                    where.append("")

                    # If the arborescence was already found before,
                    # get the corresponding shifts directly
                    if ",".join(arb) in hashes:

                        h_ind = hashes.index(",".join(arb))
                        hashes.append(",".join(arb))

                        this_w = ws[h_ind]

                        # Append the array of shifts and errors
                        # for this distribution
                        all_shifts.append(all_shifts[h_ind])
                        all_errs.append(all_errs[h_ind])
                        ws.append(this_w)
                        labels.append(f"{elem}{i+1}-{nei_elem}{nei_i}")
                        all_inds.append(all_inds[h_ind])
                        all_crysts.append(all_crysts[h_ind])

                    # Otherwise, search through the database
                    else:

                        hashes.append(",".join(arb))

                        # Get the entries of the corresponding graph
                        while len(where) > 0:
                            where.pop(-1)
                            # If we run out of options,
                            # just return all environments
                            if len(where) == 0:
                                with con:
                                    data = con.execute(
                                        f"""
                                        SELECT
                                        crystal,
                                        ind,
                                        shift,
                                        err,
                                        nei_ind,
                                        nei_shift,
                                        nei_err
                                        FROM {elem}_{nei_elem};
                                        """
                                    ).fetchall()
                                break

                            with con:
                                data = con.execute(
                                    f"""
                                    SELECT
                                    crystal,
                                    ind,
                                    shift,
                                    err,
                                    nei_ind,
                                    nei_shift,
                                    nei_err
                                    FROM {elem}_{nei_elem}
                                    WHERE {' AND '.join(where)};
                                    """
                                ).fetchall()

                            if len(data) >= n_min:
                                break

                            if verbose:
                                pp = f"    w = {this_w}: {len(data)} "
                                pp += "instances are not enough, "
                                pp += "reducing graph depth..."
                                print(pp)

                            this_w -= 1

                        # Set arrays of shifts, errors, crystal structures
                        # and atomic indices
                        shifts = []
                        errs = []
                        crysts = []
                        inds = []
                        for (
                            cryst,
                            ind,
                            shift,
                            err,
                            nei_ind,
                            nei_shift,
                            nei_err
                        ) in data:
                            if exclude is None or cryst not in exclude:
                                crysts.append(cryst)
                                inds.append([ind, nei_ind])
                                shifts.append([shift, nei_shift])
                                errs.append([err, nei_err])

                        # Append the array of shifts and errors
                        # for this distribution
                        all_shifts.append(np.array(shifts))
                        all_errs.append(np.array(errs))
                        ws.append(this_w)
                        labels.append(f"{elem}{i+1}-{nei_elem}{nei_i}")
                        all_inds.append(inds)
                        all_crysts.append(crysts)

            stop = time.time()
            pp = f"  Graph {i+1}/{len(Gs)} found. w = {this_w}, "
            pp += f"{len(all_shifts[-1])} instances. "
            pp += f"Time elapsed: {stop-start:.2f} s."
            print(pp)

        # If the neighbouring element is not set, extract the 1D shifts
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

            # If the arborescence was already found,
            # reuse the previously extracted shifts to save time
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
                    # If we run out of options,
                    # just return matching environments
                    if len(where) == 0:
                        with con:
                            data = con.execute(
                                f"""
                                SELECT
                                crystal,
                                ind,
                                shift,
                                err
                                FROM {elem};
                                """
                            ).fetchall()
                        break

                    with con:
                        data = con.execute(
                            f"""
                            SELECT
                            crystal,
                            ind,
                            shift,
                            err
                            FROM {elem}
                            WHERE {' AND '.join(where)};
                            """
                        ).fetchall()

                    if len(data) >= n_min:
                        break

                    if verbose:
                        pp = f"    w = {this_w}: {len(data)} instances "
                        pp += "are not enough, reducing graph depth..."
                        print(pp)

                    this_w -= 1

                # Set arrays of shifts, errors,
                # crystal structures and atomic indices
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
            pp = f"  Graph {i+1}/{len(Gs)} found. "
            pp += f"w = {this_w}, {len(all_shifts[-1])} instances. "
            pp += f"Time elapsed: {stop-start:.2f} s."
            print(pp)

        else:
            print(f"  Graph {i+1}/{len(Gs)} has no neighbouring {nei_elem}.")

    return all_shifts, all_errs, ws, labels, all_crysts, all_inds, hashes
