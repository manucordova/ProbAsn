###############################################################################
#                                                                             #
#                      Functions for simulating spectra                       #
#                        Author: Manuel Cordova (EPFL)                        #
#                          Last modified: 04.10.2023                          #
#                                                                             #
###############################################################################

# Import libraries
import numpy as np
import networkx as nx

# Set criterion for equivalent nodes
nm = nx.algorithms.isomorphism.categorical_node_match("elem", "X")

# Set criterion for equivalent edges
em = nx.algorithms.isomorphism.categorical_edge_match("w", -1)


def cleanup_methyl_protons(
    Gs,
    atoms,
    bonds,
    arrs=[]
):
    """Gather methyl proton shifts into only one per methyl group.

    Parameters
    ----------
    Gs : array_like
        List of graphs corresponding to each distribution.
    atoms : list
        List of atomic species in the molecule.
    bonds : list
        List of bonded atoms for each atom in the moecule (by index).
    arrs : list, default=[]
        List of arrays to apply the selection to.

    Returns
    -------
    sel_Gs : list
        Cleaned list of graphs.
    sel_arrs : list
        List of cleaned arrays.
    """

    # Initialize new arrays
    sel_Gs = []
    sel_arrs = []
    n_arrs = len(arrs)
    for i in range(n_arrs):
        sel_arrs.append([])

    # Array to store already identified methyl groups
    methyls = []

    # Loop over all graphs
    for ai, G in enumerate(Gs):

        # Get the index of the central node
        i = G.nodes[0]["ind"]
        # Check that there is only one neighbour
        # (H should be linked to only one atom)
        if len(bonds[i]) == 1:
            j = bonds[i][0]

            # Check if there are at least three protons linked to the neighbour
            nH = [atoms[k] for k in bonds[j]].count("H")

            if nH >= 3:
                # If we find a new methyl proton, add it,
                # otherwise skip it if it is part of an already known methyl
                if j not in methyls:
                    methyls.append(j)
                    sel_Gs.append(G)
                    for i in range(n_arrs):
                        sel_arrs[i].append(arrs[i][ai])

            # If this is not a methyl proton, add it
            else:
                sel_Gs.append(G)
                for i in range(n_arrs):
                    sel_arrs[i].append(arrs[i][ai])

        # If the proton is bonded to more than one atom, add it
        else:
            sel_Gs.append(G)
            for i in range(n_arrs):
                sel_arrs[i].append(arrs[i][ai])

    return sel_Gs, sel_arrs


def cleanup_methyls(
    labels,
    atoms,
    bonds,
    arrs=[]
):
    """Gather methyl 2D shifts.

    Parameters
    ----------
    labels : list
        List of labels for each distribution.
    atoms : list
        List of atomic species in the molecule.
    bonds : list
        List of bonded atoms for each atom in the moecule (by index).
    arrs : list, default=[]
        List of arrays to apply the selection to.

    Returns
    -------
    sel_labels : list
        Cleaned list of labels.
    sel_arrs : list
        List of cleaned arrays.
    """

    # Initialize the updated lists
    sel_labels = []
    sel_arrs = []
    n_arrs = len(arrs)
    for i in range(n_arrs):
        sel_arrs.append([])

    # Array to store already identified methyl groups
    methyls = []

    # Get element
    elem = ""
    for c in labels[0]:
        if c.isdigit():
            break
        elem += c

    # Loop over all distributions
    for li, label in enumerate(labels):

        # Get the index of the central node
        i0 = int(label.split("-")[0].split(elem)[1]) - 1
        i = [k for k, e in enumerate(atoms) if e == elem][i0]

        # Check if there are at least three protons linked to the atom
        nH = [atoms[k] for k in bonds[i]].count("H")

        if nH >= 3:

            # If we find a new methyl proton, add it,
            #   otherwise skip it if it is part of an already known methyl
            if i not in methyls:
                methyls.append(i)
                sel_labels.append(label)
                for i in range(n_arrs):
                    sel_arrs[i].append(arrs[i][li])

        # If this is not a methyl, add it
        else:
            methyls.append(i)
            sel_labels.append(label)
            for i in range(n_arrs):
                sel_arrs[i].append(arrs[i][li])

    return sel_labels, sel_arrs


def cleanup_equivalent(
    labels,
    hashes,
    arrs=[]
):
    """Gather equivalent graphs (identified by their shift distributions).

    Parameters
    ----------
    labels : list
        List of labels for each distribution.
    hashes : list
        List of hashes for each distribution.
    arrs : list, default=[]
        List of arrays to apply the selection to.

    Returns
    -------
    sel_labels : list
        Cleaned list of labels.
    sel_hashes : list
        Cleaned list of hashes.
    sel_arrs : list
        List of cleaned arrays.
    """

    # Initialize the updated lists
    sel_labels = []
    sel_hashes = []
    sel_arrs = []
    n_arrs = len(arrs)
    for i in range(n_arrs):
        sel_arrs.append([])

    # Loop over all the distributions
    for li, (label, hash) in enumerate(zip(labels, hashes)):

        # If the distribution is already found, modify the label
        if hash in sel_hashes:
            i = sel_hashes.index(hash)
            sel_labels[i] += f"/{label}"

        # Otherwise, append the distribution to the updated list
        else:
            sel_labels.append(label)
            sel_hashes.append(hash)
            for i in range(n_arrs):
                sel_arrs[i].append(arrs[i][li])

    return sel_labels, sel_hashes, sel_arrs


def get_lims_1D(all_shifts, all_errs, conv, extend=0.1, dx="rms"):
    """Get the limits for a predicted 1D spectrum.

    Parameters
    ----------
    all_shifts : list
        List lists of shifts.
    all_errs : list
        List lists of prediction errors.
    conv : list
        Shielding to shift conversion parameters.
    extend : float, default=0.1
        How much to extend the range, fraction of the original range,
        applied to both sides of the spectrum.
    dx : str, default="rms"
        How the error is incorporated to the minimum/maximum shift.
        "sel" to use the error of the minimum and maximum peaks,
        "max" to use the maximum error, "mean" to use the mean error,
        "rms" to use the root-mean-square error.

    Returns
    -------
    lx : list
        Limits of the spectrum.
    """

    # Initialize limits
    lx = np.ones(2) * np.mean(all_shifts[0])

    # Get minimum and maximum peak of each distribution
    for shifts, errs in zip(all_shifts, all_errs):

        # Get indices of the minimum and maximum peaks
        imin = np.argmin(shifts)
        imax = np.argmax(shifts)

        # Add the corresponding errors to the minimum and maximum peaks
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
            pp = f"Unknown dx: {dx} "
            pp += "(accepted values: 'sel', 'max', 'mean', 'rms')"
            raise ValueError(pp)

        # Get the limits
        lx[0] = min(lx[0], min_x)
        lx[1] = max(lx[1], max_x)

    # Get the range
    r = lx[1] - lx[0]

    # Extend the limits by a fraction of the range
    lx[0] -= extend * r
    lx[1] += extend * r

    return np.sort(lx * conv[0] + conv[1])


def get_lims_2D(
    all_shifts,
    all_errs,
    conv_x,
    conv_y,
    dqsq=False,
    extend=0.1,
    dx="rms"
):
    """Get the limits for a predicted 2D spectrum.

    Parameters
    ----------
    all_shifts : list
        List lists of shifts.
    all_errs : list
        List lists of prediction errors.
    conv_x : list
        Shielding to shift conversion parameters in the x-axis.
    conv_y : list
        Shielding to shift conversion parameters in the y-axis.
    dqsq : bool, default=false
        Whether or not the y-dimension is double-quantum
    extend : float, default=0.1
        How much to extend the range, fraction of the original range,
        applied to both sides of the spectrum.
    dx : str, default="rms"
        How the error is incorporated to the minimum/maximum shift.
        "sel" to use the error of the minimum and maximum peaks,
        "max" to use the maximum error, "mean" to use the mean error,
        "rms" to use the root-mean-square error.

    Returns
    -------
    lims : list
        List of limits in the two dimensions of the spectrum.
    """

    # Initialize limits
    lx = np.ones(2) * np.mean(all_shifts[0][:, 0])
    ly = np.ones(2) * np.mean(all_shifts[0][:, 1])

    # Get minimum and maximum peak of each distribution in each dimension
    for shifts, errs in zip(all_shifts, all_errs):
        imin_x = np.argmin(shifts[:, 0])
        imax_x = np.argmax(shifts[:, 0])
        imin_y = np.argmin(shifts[:, 1])
        imax_y = np.argmax(shifts[:, 1])

        # Add the corresponding errors to the minimum and maximum peaks
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
            pp = f"Unknown dx: {dx} "
            pp += "(accepted values: 'sel', 'max', 'mean', 'rms')"
            raise ValueError(pp)

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

    if dqsq:
        lims = np.array([
            np.sort(lx*conv_x[0]+conv_x[1]),
            np.sort(ly*conv_y[0]+conv_y[1])+np.sort(lx*conv_x[0]+conv_x[1])
        ])
    else:
        lims = np.array([
            np.sort(lx*conv_x[0]+conv_x[1]),
            np.sort(ly*conv_y[0]+conv_y[1])
        ])

    return lims


def make_1D_distribution(
    x,
    shifts,
    errs,
    norm=None,
    max_shifts=None
):
    """Generate a 1D distribution of chemical shifts from shifts and errors.

    Parameters
    ----------
    x : array_like
        Points in the x-axis to draw the distribution on.
    shifts : array_like
        List of shifts in the distribution.
    errs : array_like
        List of prediction errors in the distribution.
    norm : None or str, default=None
        Normalization to apply. `None` for no normalization,
        "max" to set the top of the distribution to 1.
    max_shifts: None or int, default=None
        Maximum number of shifts to consider when constructing
        the distribution. If `None`, use all shifts.

    Returns
    -------
    y : Numpy ndarray
        Value of the distribution at each point of x.
    """

    # If there are too many shifts,
    # randomly select a subset of length max_shifts
    if max_shifts is not None and max_shifts < len(shifts):
        inds = np.random.choice(len(shifts), max_shifts, replace=False)

    else:
        inds = np.arange(len(shifts))

    # Add the Gaussians
    y = np.zeros_like(x)
    for x0, w in zip(shifts[inds], errs[inds]):
        y += np.exp(
            np.square(x - x0)/(-2. * np.square(w))
        ) / (
            w * np.sqrt(2. * np.pi)
        )

    # Normalize the maximum value to one
    if norm == "max":
        y /= np.max(y)
    elif norm is not None:
        raise ValueError(f"Unknown normalization: {norm}")

    return y


def make_1D_distributions(
    lims,
    n_points,
    all_shifts,
    all_errs,
    conv,
    norm=None,
    max_shifts=None
):
    """Generate 1D distributions of chemical shifts from shifts and errors.

    Parameters
    ----------
    lims : array_like
        Limits of the distributions.
    n_points : int
        Number of points in the distribution.
    all_shifts : array_like
        List of the array of shifts for each distribution.
    all_errs : array_like
        List of the array of predicted uncertainties for each distribution.
    conv : array_like
        Shielding to shift conversion.
    norm : None or str, default=None
        Distribution normalization to apply.
        "max": set the top of the distributions to one.
    max_shifts : None or int, default=None
        Maximum number of shifts to consider to construct one distribution.

    Returns
    -------
    x : Numpy ndarray
        Array of shift values to plot the distributions on.
    ys : list
        List of distributions generated.
    """

    # Construct the array of shielding values
    x = np.linspace(lims[0], lims[1], n_points)

    # Generate the distributions
    ys = []
    for i, (sh, er) in enumerate(zip(all_shifts, all_errs)):
        print(f"  Constructing distribution {i+1}/{len(all_shifts)}...")
        ys.append(
            make_1D_distribution(
                x,
                sh*conv[0]+conv[1],
                er*np.abs(conv[0]),
                norm=norm,
                max_shifts=max_shifts
            )
        )
        print("  Distribution constructed!\n")

    return x, ys


def make_2D_distribution(
    x,
    y,
    shifts,
    errs,
    norm=None,
    max_shifts=None
):
    """Generate a 2D distribution of chemical shifts from shifts and errors.

    Parameters
    ----------
    x : array_like
        Points in the x-axis to draw the distribution on.
    y : array_like
        Points in the y-axis to draw the distribution on.
    shifts : array_like
        List of 2D shifts in the distribution.
    errs : array_like
        List of 2D prediction errors in the distribution.
    norm : None or str, default=None
        Normalization to apply. `None` for no normalization,
        "max" to set the top of the distribution to 1.
    max_shifts: None or int, default=None
        Maximum number of shifts to consider when constructing
        the distribution. If `None`, use all shifts.

    Returns
    -------
    zz : Numpy ndarray
        Value of the distribution at each point of the xy grid.
    """

    # Initialize zz array
    zz = np.zeros((y.shape[0], x.shape[0]))

    # If there are too many shifts,
    # randomly select a subset of length max_shifts
    if max_shifts is not None and max_shifts < len(shifts):
        inds = np.random.choice(len(shifts), max_shifts, replace=False)

    else:
        inds = np.arange(len(shifts))

    # Add the 2D Gaussians
    for [x0, y0], [wx, wy] in zip(shifts[inds], errs[inds]):
        gx = np.exp(np.square(x - x0) / (-2. * np.square(wx))) / wx
        gy = np.exp(np.square(y - y0) / (-2. * np.square(wy))) / wy
        zz += np.outer(gy, gx) / (2. * np.pi)

    if norm == "max":
        zz /= np.max(zz)
    elif norm is not None:
        raise ValueError(f"Unknown normalization: {norm}")

    return zz


def make_2D_distributions(
    lims,
    n_points,
    all_shifts,
    all_errs,
    conv_x,
    conv_y,
    dqsq=False,
    norm=None,
    max_shifts=None
):
    """Generate 2D distributions of chemical shifts from shifts and errors.

    Parameters
    ----------
    lims : array_like
        Limits of the distributions.
    n_points : int
        Number of points in each dimension of the distribution.
    all_shifts : array_like
        List of the array of 2D shifts for each distribution.
    all_errs : array_like
        List of the array of 2D predicted uncertainties for each distribution.
    conv_x : array_like
        Shielding to shift conversion in the x-dimension.
    conv_y : array_like
        Shielding to shift conversion in the y-dimension.
    dqsq : bool, default=False
        Whether or not the second dimension is second quantum.
    norm : None or str, default=None
        Distribution normalization to apply.
        "max": set the top of the distributions to one.
    max_shifts : None or int, default=None
        Maximum number of shifts to consider to construct one distribution.

    Returns
    -------
    xx : Numpy ndarray
        Array of shift values in the x-dimension to plot the distributions on.
    yy : Numpy ndarray
        Array of shift values in the y-dimension to plot the distributions on.
    zzs : list
        List of distributions generated.
    """

    # Generate grid of X and Y values
    x = np.linspace(lims[0, 0], lims[0, 1], n_points)
    y = np.linspace(lims[1, 0], lims[1, 1], n_points)
    xx, yy = np.meshgrid(x, y)

    # Generate the distributions
    zzs = []
    for i, (sh, er) in enumerate(zip(all_shifts, all_errs)):
        print(f"  Constructing distribution {i+1}/{len(all_shifts)}...")
        sh2 = sh.copy()
        er2 = er.copy()
        sh2[:, 0] = sh2[:, 0]*conv_x[0]+conv_x[1]
        sh2[:, 1] = sh2[:, 1]*conv_y[0]+conv_y[1]
        er2[:, 0] *= np.abs(conv_x[0])
        er2[:, 1] *= np.abs(conv_y[0])
        if dqsq:
            sh2[:, 1] += sh2[:, 0]
            er2[:, 1] = np.sqrt(np.square(er2[:, 0]) + np.square(er2[:, 1]))
        zzs.append(
            make_2D_distribution(
                x,
                y,
                sh2,
                er2,
                norm=norm,
                max_shifts=max_shifts
            )
        )
        print("  Distribution constructed!\n")

    return xx, yy, zzs


def get_distribution_max_1D(x, ys):
    """Obtain the maximum of 1D distributions.

    Parameters
    ----------
    x : array_like
        Array of shielding values to plot the distributions against.
    ys : list
        List of distributions.

    Returns
    -------
    centers : Numpy ndarray
        Array of the maximum of each distribution.
    """

    # Initialize array of centers
    centers = []

    # Get the center of each distribution
    # (withing the set of shift values considered)
    for i, y in enumerate(ys):
        ind = np.argmax(y)

        if ind == 0 or ind == len(y)-1:
            msg = f"    WARNING: the maximum of distribution {i+1}"
            msg += " is at the edge of the chemical shielding range!"
            msg += " Consider expanding the range!"
            print(msg)

        centers.append(x[ind])

    return np.array(centers)


def get_distribution_max_2D(xx, yy, zzs):
    """Obtain the maximum of 2D distributions.

    Parameters
    ----------
    xx : Numpy ndarray
        2D array of shielding values in the first dimension
        to plot the distributions against.
    yy : Numpy ndarray
        2D array of shielding values in the second dimension
        to plot the distributions against.
    zzs : list
        List of distributions.

    Returns
    -------
    centers : Numpy ndarray
        Array of the maximum of each distribution.
    """

    # Initialize array of centers
    centers = []

    # Get the center of each distribution (within the set X and Y values)
    for i, zz in enumerate(zzs):

        ind_x, ind_y = np.unravel_index(np.argmax(zz), zz.shape)

        if (
            ind_x == 0 or
            ind_y == 0 or
            ind_x == zz.shape[0]-1 or
            ind_y == zz.shape[1]-1
        ):
            msg = f"    WARNING: the maximum of distribution {i+1}"
            msg += " is at the edge of the chemical shielding range!"
            msg += " Consider expanding the range!"
            print(msg)

        centers.append([xx[ind_x, ind_y], yy[ind_x, ind_y]])

    return np.array(centers)


def compute_scores_1D(
    exp,
    shifts,
    errs,
    conv,
    max_shifts=None,
    acc=None,
    n_points=101
):
    """Compute individual assignment scores for 1D distributions.

    Parameters
    ----------
    exp : array_like
        Array of experimental shifts.
    shifts : array_like
        Array of shifts in each distribution.
    errs : array_like
        Array of prediction errors in each distribution.
    conv : array_like
        Shielding to shift conversion.
    max_shifts : None or int, default=None
        If set, maximum number of shifts to select
        to construct the distribution.
    acc : None or float, default=None
        If set, accuracy of the shift predictions.
    n_points : int, default=101
        Number of points in each shift (if `acc` is set).

    Returns
    -------
    scores : Numpy ndarray
        Array of individual assignment scores.
    """

    # Initialize array of scores
    scores = np.zeros((len(shifts), len(exp)))

    # If no accuracy is set, take the shifts as elements of the array x
    if acc is None:
        x = np.array(exp)
    # Otherwise, append the array of n_points element
    # between e - acc and e + acc to x, for each shift e
    else:
        x = []
        for e in exp:
            x.extend(list(np.linspace(e-acc, e+acc, n_points)))
        x = np.array(x)

    # Loop over all distributions
    for i, (sh, er) in enumerate(zip(shifts, errs)):
        print(f"  Evaluating distribution {i+1}/{len(shifts)}...")
        # Compute the values of the distribution on the array x
        y = make_1D_distribution(
            x,
            sh*conv[0]+conv[1],
            er,
            max_shifts=max_shifts
        )

        # If an accuracy is set, get the integral
        if acc is not None:
            y2 = []
            for j in range(len(exp)):
                y2.append(
                    np.trapz(
                        y[j*n_points:(j+1)*n_points],
                        x=x[j*n_points:(j+1)*n_points]
                    )
                )
            y = np.array(y2)

        # Append the scores of this distribution
        if np.sum(y) < 1e-6:
            msg = f"    WARNING: Distribution {i+1} does not"
            msg += " seem to match any experimental shift"
            print(msg)

        scores[i] = y / np.sum(y)
        print("  Done!\n")

    return scores


def compute_scores_2D(
    exp,
    shifts,
    errs,
    conv_x,
    conv_y,
    dqsq=False,
    max_shifts=None,
    seed=None,
    acc_x=None,
    acc_y=None,
    n_points=101
):
    """Compute individual assignment scores for 2D distributions.

    Parameters
    ----------
    exp : array_like
        Array of experimental shifts.
    shifts : array_like
        Array of 2D shifts in each distribution.
    errs : array_like
        Array of 2D prediction errors in each distribution.
    conv_x : array_like
        Shielding to shift conversion in the first dimension.
    conv_y : array_like
        Shielding to shift conversion in the second dimension.
    dqsq : bool, default=False
        Whether or not the second dimension is double quantum.
    max_shifts : None or int, default=None
        If set, maximum number of shifts to select
        to construct the distribution.
    acc_x : None or float, default=None
        If set, accuracy of the shift predictions in the first dimension.
    acc_y : None or float, default=None
        If set, accuracy of the shift predictions in the second dimension.
    n_points : int, default=101
        Number of points in each shift (if `acc` is set), in each dimension.

    Returns
    -------
    scores : Numpy ndarray
        Array of individual assignment scores.
    """

    # Initialize matrix of scores
    scores = np.zeros((len(shifts), len(exp)))

    # If no accuracy is set, set x- and y-axes as the experimental shifts
    if acc_x is None and acc_y is None:
        x = np.array(exp)[:, 0]
        y = np.array(exp)[:, 1]

    # Accuracy set only in the x-axis
    elif acc_y is None:
        x = []
        for e in exp:
            x.extend(list(np.linspace(e[0]-acc_x, e[0]+acc_x, n_points)))
        x = np.array(x)
        y = np.array(exp)[:, 1]

    # Accuracy set only in the y-axis
    elif acc_x is None:
        y = []
        for e in exp:
            y.extend(list(np.linspace(e[1]-acc_y, e[1]+acc_y, n_points)))
        y = np.array(y)
        x = np.array(exp)[:, 0]

    # Accuracy set in both axes
    else:
        x = []
        y = []
        for e in exp:
            x.extend(list(np.linspace(e[0]-acc_x, e[0]+acc_x, n_points)))
            y.extend(list(np.linspace(e[1]-acc_y, e[1]+acc_y, n_points)))
        x = np.array(x)
        y = np.array(y)

    # Loop over all distributions
    for i, (s, er) in enumerate(zip(shifts, errs)):

        print("  Evaluating distribution {}/{}...".format(i+1, len(shifts)))

        sh2 = np.zeros_like(s)
        sh2[:, 0] = s[:, 0]*conv_x[0]+conv_x[1]
        sh2[:, 1] = s[:, 1]*conv_y[0]+conv_y[1]
        er2 = er.copy()
        er2[:, 0] *= np.abs(conv_x[0])
        er2[:, 1] *= np.abs(conv_y[0])
        if dqsq:
            sh2[:, 1] += sh2[:, 0]
            er2[:, 1] = np.sqrt(np.square(er2[:, 0])+np.square(er2[:, 1]))

        # If no accuracy is set, Compute the values of the distribution
        # on the grid of experimental shifts
        if acc_x is None and acc_y is None:
            zz = make_2D_distribution(x, y, sh2, er2, max_shifts=max_shifts)
            these_scores = np.diag(zz)

        # If accuracy is set only along the x-axis,
        # integrate over the range set
        elif acc_y is None:
            these_scores = np.zeros(len(exp))
            for j in range(len(exp)):
                zz = make_2D_distribution(
                    x[j*n_points:(j+1)*n_points],
                    y,
                    sh2,
                    er2,
                    max_shifts=max_shifts
                )
                these_scores[j] = np.trapz(
                    zz[j],
                    x=x[j*n_points:(j+1)*n_points]
                )

        # If accuracy is set only along the y-axis,
        # integrate over the range set
        elif acc_x is None:
            these_scores = np.zeros(len(exp))
            for j in range(len(exp)):
                zz = make_2D_distribution(
                    x,
                    y[j*n_points:(j+1)*n_points],
                    sh2,
                    er2,
                    dqsq=dqsq,
                    max_shifts=max_shifts
                )
                these_scores[j] = np.trapz(
                    zz[:, j],
                    x=y[j*n_points:(j+1)*n_points]
                )

        # If accuracy is set along both axes, integrate over the rectangle set
        else:
            these_scores = np.zeros(len(exp))
            for j in range(len(exp)):
                zz = make_2D_distribution(
                    x[j*n_points:(j+1)*n_points],
                    y[j*n_points:(j+1)*n_points],
                    sh2,
                    er2,
                    dqsq=dqsq,
                    max_shifts=max_shifts
                )
                these_scores[j] = np.trapz(
                    np.trapz(zz, x=x[j*n_points:(j+1)*n_points]),
                    x=y[j*n_points:(j+1)*n_points]
                )

        if np.sum(these_scores) < 1e-6:
            msg = f"    WARNING: Distribution {i+1} does not seem"
            msg += " to match any experimental shift"
            print(msg)

        scores[i] = these_scores / np.sum(these_scores)
        print("  Done!\n")

    return scores
