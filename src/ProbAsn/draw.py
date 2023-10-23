###############################################################################
#                                                                             #
#            Functions for plotting (distributions, graphs, ...)              #
#                      Author: Manuel Cordova (EPFL)                          #
#                       Last modified: 03.09.2021                             #
#                                                                             #
###############################################################################


# Import libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from matplotlib.ticker import MaxNLocator

# Import local libraries
from . import sim

# Define color map
cdict = {
    "red": [
        [0.0, 1.00, 1.00],
        [0.5, 1.00, 1.00],
        [1.0, 0.85, 0.85]
    ],
    "green": [
        [0.0, 1.00, 1.00],
        [0.5, 0.77, 0.77],
        [1.0, 0.37, 0.37]
    ],
    "blue": [
        [0.0, 1.00, 1.00],
        [0.5, 0.31, 0.31],
        [1.0, 0.05, 0.05]
    ]
}

WOrBr = clrs.LinearSegmentedColormap('WOrBr', cdict)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """Truncate a matplotlib colormap.

    Parameters
    ----------
    cmap : matplotlib colormap
        Original colormap.
    minval : float, default=0.0
        Starting color value.
    maxval : float, default=1.0
        End color value.
    n : int
        Number of levels in the colormap.

    Returns
    -------
    new_cmap : matplotlib colormap
        Updated colormap.
    """

    if minval < 0. or minval > 1.:
        raise ValueError(
            f"minval outside of range: {minval} (should be within [0, 1])"
        )

    if maxval < 0. or maxval > 1.:
        raise ValueError(
            f"maxval outside of range: {maxval} (should be within [0, 1])"
        )

    # Generate the colormap from a to b
    new_cmap = clrs.LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n))
    )

    return new_cmap


def place_annot_1D(
    tops,
    lims,
    labels,
    f,
    ax,
    fontsize=16,
    dmin=0.01,
    h=1.1,
    c="k",
    n_valid=int(1e4)
):
    """Place labels on a 1D plot of distributions.

    Parameters
    ----------
    tops : list
        Position of the top of each distribution.
    lims : list
        Limits of the plot.
    labels : list
        List of labels to write (in the same order as `tops`).
    f : matplotlib Figure
        Figure handle.
    ax : matplotlib Axes
        Axis handle.
    fontsize : int, default=16
        Font size of the text.
    dmin : float, default=0.01
        Minimum distance between labels,
        as a fraction of the total plotting range.
    h : float, default=1.1
        Height of the bottom of the labels.
    c : str, default="k"
        Color scheme to use. "k" for black labels,
        "C" to match the label color to its corresponding distribution.
    n_valid : int, default=10000
        Number of iterations before aborting the label placement.

    Returns
    -------
    valid : bool
        Whether or not the placement could be done successfully.
    """

    # Initialize the maximum height so that the labels are inside the plot
    max_h = 0.

    # Sort the tops (decreasing shift, left to right on the final plot)
    sorted_inds = np.argsort(tops)[::-1]
    sorted_tops = [tops[i] for i in sorted_inds]
    sorted_labels = [labels[i] for i in sorted_inds]

    # Get the absolute minimum distance between labels
    dx = dmin*np.abs(lims[1]-lims[0])

    # Get the renderer
    r = f.canvas.get_renderer()

    # Get the size of each label to print over the distributions,
    # get the height of the plot. Initialize the array of sizes of the labels
    sizes = []
    # Loop over all labels
    for label in sorted_labels:
        # Print the label text in the plot
        t = ax.text(
            0,
            0,
            label.replace("/", "/\n"),
            ha="center",
            va="bottom",
            size=fontsize
        )

        # Get the size of the label
        box = t.get_tightbbox(r)
        box = box.transformed(ax.transData.inverted())
        sizes.append(np.abs(box.x1-box.x0))

        # If the label is above the maximum height, update the height
        if np.abs(box.y1 - box.y0) + h > max_h:

            max_h = np.abs(box.y1 - box.y0) + h

        # Discard the label text
        t.remove()

    # Set the plot height limit
    ax.set_ylim([-0.1, max_h])

    # Initialize the positions of the labels
    places = sorted_tops.copy()

    # Place the labels and move them until they don't overlap anymore
    valid = False
    n = 0
    while not valid:
        n += 1
        if n > n_valid:
            msg = "No valid placement found. "
            msg += "Increasing the width of the plot..."
            print(msg)
            return False

        valid = True
        # Loop over all neighbouring label pairs
        for i in range(1, len(places)):
            # If the distance between the two labels
            # is too small, spread these two labels
            if places[i-1] - (sizes[i-1]/2) < places[i] + (sizes[i]/1) + dx:
                places[i-1] += dx/2
                places[i] -= dx/2
                valid = False

            # If label i-1 goes outside the plot range (horizontally),
            # bring it back
            if places[i-1] - (sizes[i-1]/2) < lims[0]:
                places[i-1] += dx/2
                valid = False
            if places[i-1] + (sizes[i-1]/2) > lims[1]:
                places[i-1] -= dx/2
                valid = False

            # If label i goes outside the plot range (horizontally),
            # bring it back
            if places[i] - (sizes[i]/2) < lims[0]:
                places[i] += dx/2
                valid = False
            if places[i] + (sizes[i]/2) > lims[1]:
                places[i] -= dx/2
                valid = False

    # Print the labels in the plot,
    # connect them to their corresponding distribution
    for i, (p, t, label) in enumerate(zip(places, sorted_tops, sorted_labels)):
        if c == "k":
            ax.text(
                p,
                h,
                label.replace("/", "/\n"),
                ha="center",
                va="bottom",
                size=fontsize
            )
            ax.plot([t, p], [1., h], "k")
        elif c == "C":
            ax.text(
                p,
                h,
                label.replace("/", "/\n"),
                ha="center",
                va="bottom",
                color=f"C{i}",
                size=fontsize
            )
            ax.plot([t, p], [1., h], "C{}".format(i))
        else:
            raise ValueError("Unknown color scheme: {}".format(c))

    return True


def draw_1D_distribution_and_hist(
    x,
    y,
    shifts,
    conv,
    w,
    elem,
    fsize=(4, 3),
    fontsize=12,
    n_bins=50,
    ext=5.,
    f=None,
    display=False,
    custom=False
):
    """Plot a 1D chemical shift distribution.

    Parameters
    ----------
    x : array_like
        Array of chemical shielding values to plot.
    y : array_like
        Array of intensities of the distribution to plot.
    shifts : array_like
        Array of shifts from which the distribution is constructed.
    conv : array_like
        Conversion parameters (slope, offset) from shielding to shift.
    w : int
        Graph depth.
    elem : str
        Element for which the distributions of shifts are constructed.
    fsize : tuple, default=(4, 3)
        Figure size.
    fontsize : int, default=12
        Font size for labels.
    n_bins : int, default=50
        Number of bins to display in the histogram.
    ext : float, default=5.0
        Number of standard deviations to expand the plotting range.
    f : str or None, default=None
        File to save the figure to.
    display : bool, default=False
        Whether or not to display the figure.
    custom : bool, default=False
        Whether or not a custom distribution is set.

    Returns
    -------
    msg : str
        Output message.
    """

    plt.rcParams.update({"font.size": fontsize})
    isotopes = {
        "H": "$^1$H",
        "C": "$^{13}$C",
        "N": "$^{15}$N",
        "O": "$^{17}$O"
    }

    # Initialize figure handle
    fig = plt.figure(figsize=fsize)
    ax = fig.add_subplot(1, 1, 1)

    # Get center and width of the shift distribution
    mu0 = np.mean(shifts)
    sig0 = np.std(shifts)

    # Custom distribution
    if custom:
        locs = np.where(y > 1e-2)[0]
        sm = x[locs[0]]
        sM = x[locs[-1]]
        ax.plot([
            shifts[0]*conv[0]+conv[1],
            shifts[0]*conv[0]+conv[1]],
            [0., 1.],
            "k"
        )

        # Plot the distribution function
        ax.plot(x, y)

    else:

        # Get range for plotting
        sm = mu0 - ext * sig0
        sM = mu0 + ext * sig0

        # Plot the data histogram
        hs, _, _ = ax.hist(
            shifts*conv[0]+conv[1],
            bins=n_bins,
            range=[
                sM*conv[0]+conv[1],
                sm*conv[0]+conv[1]
            ]
        )

        # Plot the distribution function
        ax.plot(x, y * np.max(hs))

    # Set axes labels and limits
    ax.set_xlabel(isotopes[elem] + " chemical shift [ppm]")
    ax.set_ylabel("Number of instances")

    ax.set_xlim(sm*conv[0]+conv[1], sM*conv[0]+conv[1])
    ax.yaxis.set_major_locator(MaxNLocator(nbins="auto", integer=True))

    # Write the depth of the graph and the number of instances
    msg = f"w = {w}, N = {len(shifts)}\nµ= {mu0*conv[0]+conv[1]:.2f}\n"
    msg += f"$\\sigma$ = {sig0*np.abs(conv[0]):.2f}"
    ax.text(0.01, 0.98, msg, va="top", transform=ax.transAxes)

    fig.tight_layout()

    # Save the plot
    if f is not None:
        fig.savefig(f)

    # Show the plot
    if display:
        plt.show()

    plt.close()

    msg = f"w = {w}, N = {len(shifts)}\n"
    msg += f"mu = {mu0*conv[0]+conv[1]:.2f}, "
    msg += f"sigma = {sig0*np.abs(conv[0]):.2f}"

    return msg


def draw_2D_distribution_and_hist(
    xx,
    yy,
    zz,
    shifts,
    conv_x,
    conv_y,
    w,
    elem,
    nei_elem,
    fsize=(4.5, 3),
    fontsize=12,
    n_bins=50,
    levels=[0.1, 0.5, 0.9],
    ext=5,
    f=None,
    display=False,
    dqsq=False,
    custom=False
):
    """Plot a 2D chemical shift distribution.

    Parameters
    ----------
    xx : Numpy ndarray
        Grid of x values.
    yy : Numpy ndarray
        Grid of y values.
    zz : Numpy ndarray
        Distribution on the XY grid.
    shifts : Numpy array
        Array of pairs of shifts in the distribution.
    conv_x : array_like
        Conversion parameters (slope, offset)
        from shielding to shift along the x axis.
    conv_y : array_like
        Conversion parameters (slope, offset)
        from shielding to shift along the y axis.
    w : int
        Graph depth.
    elem : str
        Element on the x axis for which
        the distributions of shifts are constructed.
    nei_elem : str
        Element on the y axis for which
        the distributions of shifts are constructed.
    fsize : tuple, default=(4.5, 3)
        Figure size.
    fontsize : int, default=12
        Font size for labels.
    n_bins : int, default=50
        Number of bins to display in the histogram.
    levels : array_like, default=[0.1, 0.5, 0.9]
        List of contour levels to plot.
    ext : float, default=5.0
        Number of standard deviations to expand the plotting range.
    f : str or None, default=None
        File to save the figure to.
    display : bool, default=False
        Whether or not to display the figure.
    dqsq : bool, default=False
        Whether or not the second dimension is double quantum.
    custom : bool, default=False
        Whether or not a custom distribution is set.

    Returns
    -------
    msg : str
        Output message.
    """

    plt.rcParams.update({"font.size": fontsize})
    isotopes = {
        "H": "$^1$H",
        "C": "$^{13}$C",
        "N": "$^{15}$N",
        "O": "$^{17}$O"
    }

    # Initialize figure and axis handles
    fig = plt.figure(figsize=fsize)
    ax = fig.add_subplot(1, 1, 1)

    # Initialize colormap for the histogram
    cm = truncate_colormap(plt.get_cmap("Reds"), 0.3, 0.9)

    # Custom distribution
    if custom:
        locs_y, locs_x = np.where(zz > 1e-2)
        sm_x = xx[0, np.min(locs_x)]
        sM_x = xx[0, np.max(locs_x)]
        sm_y = yy[np.min(locs_y), 0]
        sM_y = yy[np.max(locs_y), 0]
        ax.plot(
            [
                shifts[0][0]*conv_x[0]+conv_x[1],
                shifts[0][0]*conv_x[0]+conv_x[1]
            ],
            [
                shifts[0][1]*conv_y[0]+conv_y[1],
                shifts[0][1]*conv_y[0]+conv_y[1]
            ],
            "ks"
        )

    else:
        # Get centers and widths of the shift distribution
        mu0_x = np.mean(shifts[:, 0])
        mu0_y = np.mean(shifts[:, 1])

        sig0_x = np.std(shifts[:, 0])
        sig0_y = np.std(shifts[:, 1])

        # Get ranges for plotting
        sm_x = mu0_x - ext * sig0_x
        sM_x = mu0_x + ext * sig0_x
        sm_y = mu0_y - ext * sig0_y
        sM_y = mu0_y + ext * sig0_y

        # Plot data histogram
        if dqsq:
            cs = ax.hist2d(
                shifts[:, 0] * conv_x[0] + conv_x[1],
                (
                    shifts[:, 0] * conv_x[0] + conv_x[1]
                ) + (
                    shifts[:, 1] * conv_y[0] + conv_y[1]
                ),
                bins=n_bins,
                range=[
                    [
                        sM_x * conv_x[0] + conv_x[1],
                        sm_x * conv_x[0] + conv_x[1]
                    ],
                    [
                        (
                            sM_x * conv_x[0] + conv_x[1]
                        ) + (
                            sM_y * conv_y[0] + conv_y[1]
                        ),
                        (
                            sm_x * conv_x[0] + conv_x[1]
                        ) + (
                            sm_y * conv_y[0] + conv_y[1]
                        )
                    ]
                ],
                cmin=0.5,
                cmap=cm
            )
        else:
            cs = ax.hist2d(
                shifts[:, 0] * conv_x[0] + conv_x[1],
                shifts[:, 1] * conv_y[0] + conv_y[1],
                bins=n_bins,
                range=[
                    [
                        sM_x * conv_x[0] + conv_x[1],
                        sm_x * conv_x[0] + conv_x[1]
                    ],
                    [
                        sM_y * conv_y[0] + conv_y[1],
                        sm_y * conv_y[0] + conv_y[1]
                    ]
                ],
                cmin=0.5,
                cmap=cm
            )

        cbar = fig.colorbar(cs[3], label="Number of instances")
        cbar.ax.yaxis.set_major_locator(
            MaxNLocator(nbins="auto", integer=True)
        )

    # Plot distribution function
    cm = truncate_colormap(plt.get_cmap("Blues"), 0.3, 0.9)
    ax.contour(xx, yy, zz, levels, cmap=cm)

    # Set labels and axes limits
    ax.set_xlabel(isotopes[elem] + " chemical shift [ppm]")
    ax.set_ylabel(isotopes[nei_elem] + " chemical shift [ppm]")

    ax.set_xlim(
        sm_x * conv_x[0] + conv_x[1],
        sM_x * conv_x[0] + conv_x[1]
    )

    if dqsq:
        ax.set_ylim(
            (
                sm_x * conv_x[0] + conv_x[1]
            ) + (
                sm_y * conv_y[0] + conv_y[1]
            ),
            (
                sM_x * conv_x[0] + conv_x[1]
            ) + (
                sM_y * conv_y[0] + conv_y[1]
            )
        )

    else:
        ax.set_ylim(
            sm_y * conv_y[0] + conv_y[1],
            sM_y * conv_y[0] + conv_y[1]
        )

    # Write the depth of the graph and the number of instances
    if dqsq:

        mu_x = mu0_x * conv_x[0] + conv_x[1]
        mu_y = (
            mu0_x * conv_x[0] + conv_x[1]
        ) + (
            mu0_y * conv_y[0] + conv_y[1]
        )
        sig_x = sig0_x * np.abs(conv_x[0])
        sig_y = (
            sig0_x * np.abs(conv_x[0])
        ) + (
            sig0_y * np.abs(conv_y[0])
        )

        msg = f"w = {w}, N = {len(shifts)}\n"
        msg += f"$\\mu_x$ = {mu_x:.2f}\n"
        msg += f"$\\sigma_x$ = {sig_x:.2f}\n"
        msg += f"$\\mu_y$ = {mu_y:.2f}\n"
        msg += f"$\\sigma_y$ = {sig_y:.2f}"
        ax.text(
            0.01,
            0.98,
            msg,
            va="top",
            transform=ax.transAxes
        )

    else:

        mu_x = mu0_x * conv_x[0] + conv_x[1]
        mu_y = mu0_y * conv_y[0] + conv_y[1]
        sig_x = sig0_x * np.abs(conv_x[0])
        sig_y = sig0_y * np.abs(conv_y[0])

        msg = f"w = {w}, N = {len(shifts)}\n"
        msg += f"$\\mu_x$ = {mu_x:.2f}\n"
        msg += f"$\\sigma_x$ = {sig_x:.2f}\n"
        msg += f"$\\mu_y$ = {mu_y:.2f}\n"
        msg += f"$\\sigma_y$ = {sig_y:.2f}"
        ax.text(
            0.01,
            0.98,
            msg,
            va="top",
            transform=ax.transAxes
        )

    fig.tight_layout()

    # Save the plot
    if f is not None:
        fig.savefig(f)

    # Display the plot
    if display:
        plt.show()

    plt.close()

    return


def draw_1D_distributions(
    x,
    ys,
    labels,
    elem,
    lims=None,
    exps=[],
    f=None,
    fsize=(6, 3),
    fontsize=12,
    dmin=0.001,
    c="C",
    display=False
):
    """Plot the 1D chemical shift distributions for a molecule.

    Parameters
    ----------
    x : array_like
        Array of chemical shift values to plot.
    ys : array_like
        List of distributions to plot against `x`.
    labels : array_like
        List of labels for each distribution.
    elem : str
        Element for which the distributions are constructed.
    lims : array_like or None, default=None
        Plotting range limits.
    exps : array_like, default=[]
        List of experimental shifts.
    f : str
        File path to save the figure to.
    fsize : tuple, default=(6, 3)
        Figure size.
    fontsize : int, default=12
        Font size for labels.
    dmin : float, default=0.001
        Minimum distance between labels,
        as a fraction of the total plotting range.
    c : str, default="C"
        Color scheme to use. "k" for black labels,
        "C" to match the label color to its corresponding distribution.
    display : bool, default=False
        Whether or not to display the figure.
    """

    plt.rcParams.update({"font.size": fontsize})
    isotopes = {
        "H": "$^1$H",
        "C": "$^{13}$C",
        "N": "$^{15}$N",
        "O": "$^{17}$O"
    }

    tops = sim.get_distribution_max_1D(x, ys)

    sorted_inds = np.argsort(tops)[::-1]

    if fsize == "auto":
        fsize = (6, 3)

    success = False
    while not success:

        # Initialize figure handle
        fig = plt.figure(figsize=fsize)
        ax = fig.add_subplot(1, 1, 1)

        # Plot the distributions
        for i in sorted_inds:
            ax.plot(x, ys[i])

        ax.set_xlabel(isotopes[elem] + " chemical shift [ppm]")

        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_xlim(lims[1], lims[0])
        ax.set_yticks([])

        # Place annotations on the plot
        success = place_annot_1D(
            tops,
            lims,
            labels,
            fig,
            ax,
            fontsize=fontsize-2,
            dmin=dmin,
            c=c
        )

        if not success:
            tmp = list(fsize)
            tmp[0] += 2
            fsize = tuple(tmp)

    # Plot the experimental shifts
    if exps is not None:
        for e in exps:
            ax.plot([e, e], [-0.05, 0], "k")

    fig.tight_layout()

    # Save the plot
    if f is not None:
        fig.savefig(f)

    # Show the plot
    if display:
        plt.show()

    plt.close()

    return


def draw_2D_distributions(
    xx,
    yy,
    zzs,
    labels,
    elem,
    nei_elem,
    lims=None,
    exps=[],
    levels=[0.1, 0.5, 0.9],
    f=None,
    fsize=(5, 5),
    fontsize=12,
    ncol=4,
    dqsq=False,
    display=False
):
    """Plot 2D distributions of chemical shifts.

    Parameters
    ----------
    xx : Numpy ndarray
        Grid of x values.
    yy : Numpy ndarray
        Grid of y values.
    zzs : array_like
        List of istributions on the XY grid.
    labels : array_like
        List of labels for each distribution.
    elem : str
        Element on the x axis for which the distributions are constructed.
    nei_elem : str
        Element on the y axis for which the distributions are constructed.
    lims : array_like or None, default=None
        Plotting range limits.
    exps : array_like, default=[]
        List of experimental shifts.
    levels : array_like, default=[0.1, 0.5, 0.9]
        List of contour levels to plot.
    f : str
        File path to save the figure to.
    fsize : tuple, default=(5, 5)
        Figure size.
    fontsize : int, default=12
        Font size for labels.
    ncol : int, default=4
        Number of columns of legends
    dqsq : bool, default=False
        Whether or not the second dimension is double quantum.
    display : bool, default=False
        Whether or not to display the figure.
    """

    plt.rcParams.update({"font.size": fontsize})
    isotopes = {
        "H": "$^1$H",
        "C": "$^{13}$C",
        "N": "$^{15}$N",
        "O": "$^{17}$O"
    }
    cmaps = ["Blues", "Reds", "Greens", "Purples", "Oranges"]

    tops = sim.get_distribution_max_2D(xx, yy, zzs)

    sorted_inds = np.argsort(tops[:, 0])[::-1]

    cs = []

    # Initialize figure handle
    fig = plt.figure(figsize=fsize)
    ax = fig.add_subplot(1, 1, 1)

    k = 0
    # Loop over all distributions
    for i in sorted_inds:

        # Get the color of the distribution
        if k >= len(cmaps):
            k = 0

        # Plot the distribution
        cm = truncate_colormap(plt.get_cmap(cmaps[k]), 0.3, 0.9)
        ax.contour(xx, yy, zzs[i], levels, cmap=cm)
        ax.plot(tops[i, 0], tops[i, 1], ".", color=cm(1.))
        cs.append(cm(0.5))

        k += 1

    # Set labels and ax limits
    ax.set_xlabel(isotopes[elem] + " chemical shift [ppm]")
    ax.set_ylabel(isotopes[nei_elem] + " chemical shift [ppm]")

    ax.set_xlim(lims[0, 1], lims[0, 0])
    ax.set_ylim(lims[1, 1], lims[1, 0])

    # Generate legends
    hs = []
    for c in cs:
        hs.append(mpl.lines.Line2D([0], [0], color=c))

    print_labels = []
    for i in sorted_inds:
        print_labels.append(labels[i].replace("/", "/\n"))

    ax.legend(
        hs,
        print_labels,
        bbox_to_anchor=(0., 1.01, 1., 0.01),
        loc=3,
        ncol=ncol,
        mode="expand",
        borderaxespad=0.,
        prop={"size": fontsize - 2}
    )

    if exps is not None:
        for exp in exps:
            ax.plot(exp[0], exp[1], "wo")
            ax.plot(exp[0], exp[1], "k.")

    fig.tight_layout()

    # Save the plot
    if f is not None:
        fig.savefig(f)

    # Display the plot
    if display:
        plt.show()

    plt.close()

    return


def load_prob_file(f):
    """Load a probabilities file and extract experimental shifts,
    nuclei labels and assignment probabilities.

    Parameters
    ----------
    f : str
        Path to the input file.

    Returns
    -------
    exps : dict
        Dictionary of experimental shifts.
    labels : dict
        Dictionary of nuclei labels.
    probs : dict
        Dictionary of assignment probabilities.
    """

    # Initialize dictionaries of experimental shifts,
    # nuclei labels and assignment probabilities
    exps = {}
    labels = {}
    probs = {}

    ind = 0

    # Load the file
    with open(f, "r") as F:
        lines = F.read().split("\n")

    for i, l in enumerate(lines):
        if "Label" in l:

            # Initialize the arrays of labels and the matrix of probabilities
            labels[ind] = []
            probs[ind] = []

            # Load the experimental shifts line
            tmp = l.split()
            exps[ind] = tmp[1:]

            # Load the labels and probabilities
            j = 1
            while len(lines[i+j]) > 0:

                tmp = lines[i+j].replace("%", "").split()

                labels[ind].append(tmp[0])

                probs[ind].append([float(k) for k in tmp[1:]])

                j += 1

            probs[ind] = np.array(probs[ind])

            ind += 1

    return exps, labels, probs


def print_probabilities(f, fontsize=12, display=True, cmap=None):
    """Print probabilistic assignment maps.

    Parameters
    ----------
    f : str
        Path to the input file.
    fontsize : int, default=12
        Font size for labels.
    display : bool, default=True
        Whether or not to display the plot.
    cmap : matplotlib Colormap
        Colormap to use.
    """

    plt.rcParams.update({"font.size": fontsize})

    # Get experimental shifts, nuclei labels and assignment probabilities
    exps, labels, probs = load_prob_file(f)

    # Split by number of equivalent nuclei/distributions
    for k in exps.keys():

        # Sort the experimental shifts by decreasing value
        num_exps = [
            float(label.split("/")[0].split("\\")[0]) for label in exps[k]
        ]
        sorted_exp_inds = np.argsort(num_exps)[::-1]
        sorted_exps = [exps[k][i] for i in sorted_exp_inds]

        # Get "Center of mass" of assignments
        coms = []

        # Loop over all labels
        for p in probs[k]:
            com = 0.

            for i, j in enumerate(sorted_exp_inds):
                com += i * p[j]

            com /= np.sum(p)
            coms.append(com)

        # Get label ordering
        sorted_lab_inds = np.argsort(coms)
        sorted_labs = [labels[k][i] for i in sorted_lab_inds]

        # Rearrange the probability matrix according to the changes made to
        # the order of experimental shifts and labels
        sorted_probs = np.zeros_like(probs[k])

        for i, i2 in enumerate(sorted_lab_inds):
            for j, j2 in enumerate(sorted_exp_inds):

                sorted_probs[i, j] = probs[k][i2, j2]

        # Get the figure size (proportional to the number of
        # experimental shifts and labels)
        le = 0
        ll = 0
        for e in exps[k]:
            le = max(le, len(e))
        for label in labels[k]:
            ll = max(ll, len(label))

        lx = (ll + 3) * fontsize / 100.
        ly = (le + 2) * fontsize / 100.

        lx += (len(exps[k]) + 2) * 0.5
        ly += len(labels[k]) * 0.5

        # Initialize figure handle
        fig = plt.figure(figsize=(lx, ly))
        ax = fig.add_subplot()

        # Plot the probability map
        if cmap is None:
            cmap = WOrBr

        c = ax.pcolormesh(
            sorted_probs,
            cmap=cmap,
            vmin=0.,
            vmax=100.,
            edgecolors=(0.9, 0.9, 0.9),
            linewidths=1
        )

        fig.colorbar(c, label="Probability [%]")

        x_ticks = np.array(range(sorted_probs.shape[1])) + 0.5
        y_ticks = np.array(range(sorted_probs.shape[0])) + 0.5

        # Set tick labels of x (experimental shifts) and y (nuclei labels) axes
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

        ax.set_xticklabels(
            sorted_exps,
            rotation=90,
            ha="right",
            va="center_baseline",
            rotation_mode="anchor"
        )
        ax.set_yticklabels(sorted_labs)

        # Set axis labels
        ax.set_xlabel("Chemical shift [ppm]")
        ax.set_ylabel("Label")

        # Save the figure
        fig.tight_layout()
        fig.savefig(f.replace(".dat", "_{}.pdf".format(k)))

        if display:
            plt.show()

        plt.close()

    return
