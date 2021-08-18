####################################################################################################
###                                                                                              ###
###                     Functions for plotting (distributions, graphs, ...)                      ###
###                               Author: Manuel Cordova (EPFL)                                  ###
###                                Last modified: 10.05.2021                                     ###
###                                                                                              ###
####################################################################################################

# Import libraries
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import scipy.optimize as op
import copy
import tqdm.auto as tqdm



# Define color map
cdict = {"red": [[0.0, 1.00, 1.00],
                [0.5, 1.00, 1.00],
                [1.0, 0.85, 0.85]],
        "green": [[0.0, 1.00, 1.00],
                  [0.5, 0.77, 0.77],
                  [1.0, 0.37, 0.37]],
        "blue": [[0.0, 1.00, 1.00],
                 [0.5, 0.31, 0.31],
                 [1.0, 0.05, 0.05]]
}
WOrBr = clrs.LinearSegmentedColormap('WOrBr', cdict)



def place_annot_1D(tops, lims, conv, labels, f, ax, dMin=0.01, h=1.1, c="k", N_valid=100000):
    """
    Place labels on a 1D plot of distributions
    
    Inputs: - tops      Position of the top of each distribution (shielding)
            - lims      Limits of the plot (shielding)
            - conv      Conversion parameters for shielding to shift
            - labels    Labels to write (in order of the tops)
            - f         Figure handle
            - ax        Axis (subplot) handle
            - dMin      Minimum distance between labels (fraction of the total range of shifts)
            - h         Height for the bottom of the labels
            - c         Color scheme to use
            - N_valid   Number of iterations before aborting the placement
    """
    
    # Initialize the maximum height so that the labels are inside the plot
    max_h = 0.
    
    # Sort the tops (increasing shielding, left to right on the final plot)
    sorted_inds = np.argsort(tops)
    sorted_tops = [tops[i] for i in sorted_inds]
    sorted_labels = [labels[i] for i in sorted_inds]
    
    # Get the absolute minimum distance between labels
    dx = dMin*np.abs(lims[1]-lims[0])
    
    # Get the renderer
    r = f.canvas.get_renderer()
    
    # Get the size of each label to print over the distributions, get the height of the plot
    change = True
    while change:
        change = False
        # Initialize the array of sizes of the labels
        sizes = []
        # Loop over all labels
        for l in sorted_labels:
            # Print the label text in the plot
            t = ax.text(0, 0, l.replace("/", "/\n"), ha="center", va="bottom")
            
            # Get the size of the label
            box = t.get_tightbbox(r)
            box = box.transformed(ax.transData.inverted())
            sizes.append(np.abs(box.x1-box.x0))
            
            # If the label is above the maximum height, update the height and start again
            if np.abs(box.y1 - box.y0) + h > max_h:
                max_h = np.abs(box.y1 - box.y0) + h + 0.01
                change = True
                
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
        if n > N_valid:
            raise ValueError("No valid placement found. Increase the width of the plot.")
        valid = True
        # Loop over all neighbouring label pairs
        for i in range(1, len(places)):
            # If the distance between the two labels is too small, spread these two labels
            if places[i-1] + (sizes[i-1]/2) > places[i] - (sizes[i]/1) - dx:
                places[i-1] -= dx/2
                places[i] += dx/2
                valid = False
            
            # If label i-1 goes outside the plot range (horizontally), bring it back
            if places[i-1] - (sizes[i-1]/2) < lims[0]:
                places[i-1] += dx/2
                valid = False
            if places[i-1] + (sizes[i-1]/2) > lims[1]:
                places[i-1] -= dx/2
                valid = False
            
            # If label i goes outside the plot range (horizontally), bring it back
            if places[i] - (sizes[i]/2) < lims[0]:
                places[i] += dx/2
                valid = False
            if places[i] + (sizes[i]/2) > lims[1]:
                places[i] -= dx/2
                valid = False
    
    unsorted_inds = np.argsort(sorted_inds)
    # Print the labels in the plot, connect them to their corresponing distribution
    for i, (p, t, l) in enumerate(zip([places[k] for k in unsorted_inds], tops, labels)):
        if c == "k":
            ax.text(p*conv[0]+conv[1], h, l.replace("/", "/\n"), ha="center", va="bottom")
            ax.plot([t*conv[0]+conv[1], p*conv[0]+conv[1]], [1., h], "k")
        elif c == "C":
            ax.text(p*conv[0]+conv[1], h, l.replace("/", "/\n"), ha="center", va="bottom", color="C{}".format(i))
            ax.plot([t*conv[0]+conv[1], p*conv[0]+conv[1]], [1., h], "C{}".format(i))
        else:
            raise ValueError("Unknown color scheme: {}".format(c))

    return



def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Truncate a matplotlib colormap
    
    Inputs: - cmap          Original colormap
            - minval        Starting color value [0, 1]
            - maxval        End color value [0, 1]
            - n             Number of levels in the colormap
    
    Output: - new_cmap      Updated colormap
    """

    # Generate the colormap from a to b
    new_cmap = clrs.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval), cmap(np.linspace(minval, maxval, n)))
    
    return new_cmap



def place_annot_2D(X, Y, Zs, tops, lims, conv_x, conv_y, labels, f, ax, cs, dMin=0.01, h=0.1, size=14, seed=None, Nmax=1000, Nmax_valid=1000):
    """
    Place labels on a 2D plot of distributions
    
    Inputs: - X             Grid of x values
            - Y             Grid of y values
            - Zs            List of distributions on the XY mesh
            - tops          Position of the top of each distribution (shielding)
            - lims          Limits of the plot (shielding)
            - conv_x        Conversion parameters for shielding to shift in the x direction
            - conv_y        Conversion parameters for shielding to shift in the y direction
            - labels        Labels to write (in order of the tops)
            - f             Figure handle
            - ax            Axis (subplot) handle
            - cs            List of colors to write the annotations in
            - dMin          Minimum distance between labels (fraction of the total range of shifts)
            - h             Maximum value of the distributions where a label can be placed
            - size          Font size for the labels
            - seed          Seed to generate random numbers in a deterministic way
            - Nmax          Maximum number of iterations to try to place labels
            - Nmax_valid    Maximum number of iterations to take a valid initial guess for label placement
    """
    
    # Get the chemical shift ranges
    rx = lims[0][1]-lims[0][0]
    ry = lims[1][1]-lims[1][0]
    
    # Initialize the renderer
    r = f.canvas.get_renderer()
    
    # Seed the random number generateion
    if seed is not None:
        np.random.seed(seed)
    
    failed = True
    for it in tqdm.tqdm(range(Nmax)):
        failed = False
        # Generate slightly displaced placements compared to the tops of the distributions
        places = []
        for t, l in zip(tops, labels):
            this_place = np.copy(t)

            # Randomly displace the placement
            v = np.array([np.random.normal(), np.random.normal()])
            v /= np.linalg.norm(v)

            this_place[0] += v[0]*(lims[0][1]-lims[0][0])*dMin
            this_place[1] += v[1]*(lims[1][1]-lims[1][0])*dMin

            # Get the value of the distribution at the placement
            i = np.argmin(np.abs(Y[:,0] - this_place[1]))
            j = np.argmin(np.abs(X[0] - this_place[0]))

            # Make the position valid
            valid = False
            for it2 in range(Nmax_valid):
                valid = True
                for t, Z in zip(tops, Zs):
                    # If the position of the label overlaps with any distribution, move the label away from the center of the distribution
                    if Z[i, j] > h:
                        valid = False
                        
                        v2 = this_place - t
                        
                        this_place[0] += v2[0] * dMin
                        this_place[1] += v2[1] * dMin

                        # Get the value of the distribution at the placement
                        i = np.argmin(np.abs(Y[:,0] - this_place[1]))
                        j = np.argmin(np.abs(X[0] - this_place[0]))

                        break
                        
                if valid:
                    break


            places.append(this_place)

        txts = []
        ls = []
        already_corners = []
        for t, p, l, c in zip(tops, places, labels, cs):

            ha = "right"
            va = "top"
            if t[0] < p[0]:
                ha = "left"
            if t[1] < p[1]:
                va = "bottom"

            txts.append(len(ax.texts))
            txt = ax.text(p[0]*conv_x[0]+conv_x[1], p[1]*conv_y[0]+conv_y[1], l.replace("/","/\n"), ha=ha, va=va, c=c, size=size)
            ls.append(len(ax.lines))
            ax.plot(np.array([t[0], p[0]])*conv_x[0]+conv_x[1], np.array([t[1], p[1]])*conv_y[0]+conv_y[1], color=c)

            # Check for clash with any distribution or with any other text box
            box = txt.get_tightbbox(r)
            box = box.transformed(ax.transData.inverted())
            corners = []
            corners.append([box.x0, box.y0])
            corners.append([box.x0, box.y1])
            corners.append([box.x1, box.y0])
            corners.append([box.x1, box.y1])

            for co in corners:
                for Z in Zs:
                    i = np.argmin(np.abs(Y[:,0] - (co[1]-conv_y[1])/conv_y[0]))
                    j = np.argmin(np.abs(X[0] - (co[0]-conv_x[1])/conv_x[0]))
                    if Z[i, j] > h:
                        failed = True
                        break
                if failed:
                    break
            
            for al in already_corners:
                if not (al[0] <= box.x1 or al[2] >= box.x0 or al[3] >= box.y0 or al[1] <= box.y1):
                    failed = True
                
            already_corners.append([box.x0, box.y0, box.x1, box.y1])
                    
            if failed:
                for t, l in zip(reversed(txts), reversed(ls)):
                    ax.lines[l].remove()
                    ax.texts[t].remove()
                break
        
        if not failed:
            break
        
    
    if failed:
        print("Labels could not be placed")
        hs = []
        for c in cs:
            hs.append(mpl.lines.Line2D([0], [0], color=c))
        
        ax.legend(hs, labels, bbox_to_anchor=(1.05, 1), prop={"size": size})
    
    return



def gauss(x, A, mu, sigma):
    """
    Plot a Gaussian on an array x.
    
    Inputs: - x         Array of x values
            - A         Amplitude (pre-factor) of the Gaussian
            - mu        Center of the Gaussian
            - sigma     Width of the Gaussian
            
    Output: - y         Value of the Gaussian at the x-values
    """
    
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))



def gauss_2d(xy, A, x0, y0, sig_x, sig_y, theta, offset):
    """
    Plot a 2D Gaussian on a grid xy.
    
    Inputs: - xy        Tuple of x and y values
            - A         Amplitude (pre-factor) of the Gaussian
            - x0        Center of the Gaussian in the x direction
            - y0        Center of the Gaussian in the y direction
            - sig_x     Width of the Gaussian in the x direction
            - sig_y     Width of the Gaussian in the y direction
            - theta     Angle of the Gaussian
            - offset    Offset of the Gaussian
            
    Output: - Z         Value of the Gaussian on the xy grid
    """
    
    (x, y) = xy
    
    a = (np.cos(theta)**2)/(2*sig_x**2) + (np.sin(theta)**2)/(2*sig_y**2)
    b = -(np.sin(2*theta))/(4*sig_x**2) + (np.sin(2*theta))/(4*sig_y**2)
    c = (np.sin(theta)**2)/(2*sig_x**2) + (np.cos(theta)**2)/(2*sig_y**2)
    
    g = offset + A*np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0)
                            + c*((y-y0)**2)))
    
    return g.ravel()
    


def draw_1D_distribution_and_hist(x, y, shifts, conv, w, fsize=(8,6), n_bins=50, al=1., ext=5, show_G=False, show_N=False, f=None, display=False):
    """
    Plot a 1D chemical shift distribution along with the corresponding histogram of shifts
    
    Inputs:     - x         Array of chemical shielding values to plot
                - y         Distribution to plot on the array x
                - shifts    Numpy array containing the shifts from which the distribution is constructed
                - conv      Conversion from shielding to shift
                - w         Graph depth
                - fsize     Figure size
                - n_bins    Number of bars to display for the histogram
                - al        Opacity of the histogram
                - ext       Extension for plotting range (factor before standard deviation of shifts)
                - show_G    Whether or not to show the Gaussian fit to the distribution function
                - show_N    Whether or not to show the Normal distribution of the shift histogram
                - f         File to save the figure to
                - display   Whether or not to display the figure
    """
    
    # Initialize figure handle
    fig = plt.figure(figsize=fsize)
    ax = fig.add_subplot(1,1,1)
    ax2 = ax.twinx()
    
    # Initialize legend handles
    hs = []
    ls = []
    
    # Get center and width of the shift distribution
    mu0 = np.mean(shifts)
    sig0 = np.std(shifts)
    
    # Get range for plotting
    sm = mu0 - ext * sig0
    sM = mu0 + ext * sig0
    
    # Plot the data histogram
    ns, bins = np.histogram(shifts*conv[0]+conv[1], bins=n_bins, range=[sM*conv[0]+conv[1], sm*conv[0]+conv[1]])
    
    old_n = 0
    for n, b0, b1 in zip(ns, bins[:-1], bins[1:]):
        if n > old_n:
            ax2.plot([b0, b0], [old_n, n], "k", alpha=al)
        ax2.plot([b0, b1], [n, n], "k", alpha=al)
        ax2.plot([b1, b1], [0, n], "k", alpha=al)
    h = ax2.plot([b1, b1], [old_n, n], "k", alpha=al)
    hs.append(h[0])
    ls.append("Hist")
    
    # Plot the distribution function
    h = ax.plot(x*conv[0]+conv[1], y, "C0")
    hs.append(h[0])
    ls.append("SoG")
    
    # Plot the Gaussian fit to the distribution function
    if show_G:
        coeffs, var_matrix = op.curve_fit(gauss, x, y, p0=(1., mu0, sig0))
        
        h = ax.plot(x*conv[0]+conv[1], gauss(x, coeffs[0], coeffs[1], coeffs[2]), "C2")
        hs.append(h[0])
        ls.append("G(SoG)\n(std={:.2f} ppm)".format(coeffs[2]))
        
    # Plot the Normal distribution of the shift histogram
    if show_N:
        h = ax.plot(x*conv[0]+conv[1], gauss(x, 1., mu0, sig0), "C1")
        hs.append(h[0])
        ls.append("N(hist)\n(std={:.2f} ppm)".format(sig0))
    
    # Set axes labels and limits
    ax.set_xlabel("Chemical shift [ppm]")
    ax.set_ylabel("Probability density")
    ax2.set_ylabel("Number of instances")
    
    ax.set_xlim(sm*conv[0]+conv[1], sM*conv[0]+conv[1])
    
    # Write the depth of the graph and the number of instances
    ax.text(0.01, 0.98, "w = {}, N = {}".format(w, len(shifts)), va="top", transform=ax.transAxes)
    
    ax.legend(hs, ls, loc=1)
    
    fig.tight_layout()
    
    # Save the plot
    if f is not None:
        fig.savefig(f)
    
    # Show the plot
    if display:
        plt.show()
    
    plt.close()
    
    return
    
    
    
def draw_1D_distributions(x, ys, centers, conv, labels, lims, exps=None, f=None, fsize=(10,6), dMin=0.001, c="C", display=False):
    """
    Plot the 1D chemical shift distributions for a molecule
    
    Inputs: - x         Array of chemical shielding values to plot
            - ys        Distributions to plot on the array x
            - centers   Centers of the distributions (value of the shielding)
            - conv      Shielding to shift conversion factors
            - labels    Labels of the chemical shift distributions
            - lims      Plotting range limits
            - exps      List of experimental shifts
            - f         File to save the figure to
            - fsize     Figure size
            - dMin      Minimum distance between labels (fraction of figure width)
            - c         Color scheme for labels
            - display   Whether or not to display the figure
    """
    
    # Initialize figure handle
    fig = plt.figure(figsize=fsize)
    ax = fig.add_subplot(1,1,1)
    
    # Plot the distributions
    for y in ys:
        ax.plot(x*conv[0]+conv[1], y)
    
    ax.set_xlabel("Chemical shift [ppm]")
    ax.set_ylabel("Probability density")
    
    ax.set_xlim(lims[0]*conv[0]+conv[1], lims[1]*conv[0]+conv[1])
    
    # Place annotations on the plot
    place_annot_1D(centers, lims, conv, labels, fig, ax, dMin=dMin, c=c)
    
    fig.tight_layout()
    
    # Save the plot
    if f is not None:
        fig.savefig(f)
        
    # Plot the experimental shifts
    if exps is not None:
        for e in exps:
            ax.plot([e, e], [-0.05, 0], "k")
        
        fig.tight_layout()
        
        # Save the plot
        if f is not None:
            tmp = f.split(".")
            f2 = ".".join(tmp[:-1])
            f2 += "_with_exp."
            f2 += tmp[-1]
            fig.savefig(f2)
    
    # Show the plot
    if display:
        plt.show()
    
    plt.close()
        
    return
    
    
    
def draw_2D_distribution_and_hist(X, Y, Z, c, shifts, conv_x, conv_y, w, xlabel="x", ylabel="y", fsize=(8, 6), n_bins=50, levels=[0.01, 0.1, 0.5, 0.9], lw=1., ext=5, show_G=False, show_N=False, f=None, display=False):
    """
    Plot a 2D distribution along with the histogram of the corresponding data. Optionally plot Gaussian fitting of the distribution function or of the histogram of the data
    
    Inputs: - X         Grid of x values
            - Y         Grid of y values
            - Z         Distribution on the XY mesh
            - c         Center of the distribution
            - shifts    Numpy array of shifts in the distribution
            - conv_x    Conversion of x-values from shielding to shift
            - conv_y    Conversion of y-values from shielding to shift
            - w         Depth of the graph
            - xlabel    Label of the x-axis
            - ylabel    Label of the y-axis
            - fsize     Size of the generated plot
            - n_bins    Number of bins to categorize the histogram data into
            - levels    Contour levels to display
            - lw        Linewidth of the contour plots
            - ext       Number of standard deviations to extend the plot range in every direction
            - show_G    Whether to display the Gaussian fit of the distribution function
            - show_N    Whether to display the normal distribution fitting of the histogram
            - f         Filename to save the plot to
            - display   Whether to display the plot or not
    """
    
    # Initialize legend handles
    hs = []
    ls = []
    
    # Initialize figure and axis handles
    fig = plt.figure(figsize=fsize)
    ax = fig.add_subplot(1,1,1)
    
    # Initialize colormap for the histogram
    cm = truncate_colormap(plt.get_cmap("Reds"), 0.3, 0.9)
    
    # Get centers and widths of the shift distribution
    mu0_x = np.mean(shifts[:,0])
    mu0_y = np.mean(shifts[:,1])
    
    sig0_x = np.std(shifts[:,0])
    sig0_y = np.std(shifts[:,1])
    
    # Get ranges for plotting
    sm_x = mu0_x - ext * sig0_x
    sM_x = mu0_x + ext * sig0_x
    sm_y = mu0_y - ext * sig0_y
    sM_y = mu0_y + ext * sig0_y
    
    # Get shift ranges
    rx = sM_x - sm_x
    ry = sM_y - sm_y
    
    # Plot data histogram
    cs = ax.hist2d(shifts[:,0]*conv_x[0]+conv_x[1], shifts[:,1]*conv_y[0]+conv_y[1], bins=n_bins,
                   range=[[sM_x*conv_x[0]+conv_x[1], sm_x*conv_x[0]+conv_x[1]], [sM_y*conv_y[0]+conv_y[1], sm_y*conv_y[0]+conv_y[1]]], cmin=0.5, cmap=cm)
    
    fig.colorbar(cs[3])
    
    # Plot distribution function
    cm = truncate_colormap(plt.get_cmap("Blues"), 0.3, 0.9)
    
    cs = ax.contour(X*conv_x[0]+conv_x[1], Y*conv_y[0]+conv_y[1], Z, levels, cmap=cm, linewidths=lw)
    
    ax.plot(c[0]*conv_x[0]+conv_x[1], c[1]*conv_y[0]+conv_y[1], ".", color=cm(1.))
    
    hs.append(mpl.lines.Line2D([0], [0], color=cm(0.9)))
    ls.append("SoG")
    
    # Gaussian fit of the distribution function
    if show_G:
        
        # Fit a Gaussian to the distribution function
        coeffs, var_matrix = op.curve_fit(gauss_2d, (X, Y), Z.ravel(), p0=(1., mu0_x, mu0_y, sig0_x, sig0_y, 0., 0.))
        
        Z2 = gauss_2d((X, Y), coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[5], coeffs[6]).reshape(X.shape[0], X.shape[1])
        
        cm = truncate_colormap(plt.get_cmap("Greens"), 0.3, 0.9)
        ax.contour(X*conv_x[0]+conv_x[1], Y*conv_y[0]+conv_y[1], Z2, levels, cmap=cm, linewidths=lw)
        hs.append(mpl.lines.Line2D([0], [0], color=cm(0.9)))
        ls.append("G(SoG) (std= {:.2f}/{:.2f} ppm)".format(coeffs[3], coeffs[4]))
        
    # Normal distribution fit of the histogram of data
    if show_N:
        
        # Fit a Gaussian to the 2D histogram of data
        Z2 = gauss_2d((X, Y), 1., mu0_x, mu0_y, sig0_x, sig0_y, 0., 0.).reshape(X.shape[0], X.shape[1])
        
        cm = truncate_colormap(plt.get_cmap("Reds"), 0.3, 0.9)
        ax.contour(X*conv_x[0]+conv_x[1], Y*conv_y[0]+conv_y[1], Z2, levels, cmap=cm, linewidths=lw)
        hs.append(mpl.lines.Line2D([0], [0], color=cm(0.9)))
        ls.append("N(shifts) (std= {:.2f}/{:.2f} ppm)".format(sig0_x, sig0_y))
        
    # Set labels and ax limits
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ax.set_xlim(sm_x*conv_x[0]+conv_x[1], sM_x*conv_x[0]+conv_x[1])
    ax.set_ylim(sm_y*conv_y[0]+conv_y[1], sM_y*conv_y[0]+conv_y[1])
    
    # Write the depth of the graph and the number of instances
    ax.text(0.01, 0.05, "w = {}, N = {}".format(w, len(shifts)), va="top", transform=ax.transAxes)
    
    ax.legend(hs, ls)
    
    fig.tight_layout()
    
    # Save the plot
    if f is not None:
       fig.savefig(f)
        
    # Display the plot
    if display:
        plt.show()
    
    plt.close()
    
    return



def draw_2D_distributions(X, Y, Zs, centers, labels, lims, conv_x, conv_y, cmaps, exps=None, xlabel="x", ylabel="y", levels=[0.1, 0.5, 0.9], fsize=(10, 6), f=None, display=False, f_size=16, l_size=14, h=0.05, seed=None, Nmax=1000):
    """
    Plot 2D distributions of chemical shifts. Optionally plot experimental shifts along with the distributions
    
    Inputs: - X         Grid of x values
            - Y         Grid of y values
            - Z         Distribution on the XY mesh
            - c         Center of the distribution
            - labels    Labels of the distributions
            - lims      Limits of the plot
            - conv_x    Conversion of x-values from shielding to shift
            - conv_y    Conversion of y-values from shielding to shift
            - cmaps     Color maps to cycle through
            - exps      Numpy array of experimental shifts
            - xlabel    Label of the x-axis
            - ylabel    Label of the y-axis
            - levels    Contour levels to display
            - fsize     Size of the generated plot
            - f         Filename to save the plot to
            - display   Whether to display the plot or not
            - l_size      Font size for the labels
            - h         Maximum value of the distributions where a label can be placed
            - seed      Seed to generate random numbers in a deterministic way
    """
    
    plt.rcParams.update({"font.size": f_size})
    
    cs = []
    
    # Initialize figure handle
    fig = plt.figure(figsize=fsize)
    ax = fig.add_subplot(1,1,1)
    
    # Loop over all distributions
    for i, (Z, c) in enumerate(zip(Zs, centers)):
    
        # Get the color of the distribution
        ind = i
        while ind >= len(cmaps):
            ind -= len(cmaps)

        # Plot the distribution
        cm = truncate_colormap(plt.get_cmap(cmaps[ind]), 0.3, 0.9)
        ax.contour(X*conv_x[0]+conv_x[1], Y*conv_y[0]+conv_y[1], Z, levels, cmap=cm)
        ax.plot(c[0]*conv_x[0]+conv_x[1], c[1]*conv_y[0]+conv_y[1], ".", color=cm(1.))
        cs.append(cm(0.5))
            
    # Set labels and ax limits
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ax.set_xlim(lims[0,0] * conv_x[0] + conv_x[1], lims[0,1] * conv_x[0] + conv_x[1])
    ax.set_ylim(lims[1,0] * conv_y[0] + conv_y[1], lims[1,1] * conv_y[0] + conv_y[1])
    
    # Place annotations
    place_annot_2D(X, Y, Zs, centers, lims, conv_x, conv_y, labels, fig, ax, cs, size=l_size, h=h, seed=seed, Nmax=Nmax)

    fig.tight_layout()
    
    # Save the plot
    if f is not None:
        fig.savefig(f)

    if exps is not None:
        ax.plot(exps[:,0], exps[:,1], "wo")
        ax.plot(exps[:,0], exps[:,1], "k.")
        
        fig.tight_layout()
        
        # Save the plot
        if f is not None:
            tmp = f.split(".")
            f2 = ".".join(tmp[:-1])
            f2 += "_with_exp."
            f2 += tmp[-1]
            fig.savefig(f2)
        
    # Display the plot
    if display:
        plt.show()
        
    plt.close()

    return



def load_prob_file(f):
    """
    Load a probabilities file and extract experimental shifts, nuclei labels and assignment probabilities
    
    Input:      - f         Path to file
    
    Outputs:    - exps      Dictionary of experimental shifts
                - labels    Dictionary of nuclei labels
                - probs     Dictionary of assignment probabilities
    """

    # Initialize dictionaries of experimental shifts, nuclei labels and assignment probabilities
    exps = {}
    labels = {}
    probs = {}

    ind = 0

    # Load the file
    with open(f, "r") as F:
        lines = F.read().split("\n")

    for i, l in enumerate(lines):
        if "Label" in l:
            
            # Initialize the arrays of labels and the matrix of probabilities
            labels[ind] = []
            probs[ind] = []
            
            # Load the experimental shifts line
            tmp = l.split()
            exps[ind] = tmp[1:]
            
            # Load the labels and probabilities
            j = 1
            while len(lines[i+j]) > 0:
                
                tmp = lines[i+j].replace("%","").split()
                
                labels[ind].append(tmp[0])
                
                probs[ind].append([float(k) for k in tmp[1:]])
                
                j += 1
            
            probs[ind] = np.array(probs[ind])
            
            ind += 1

    return exps, labels, probs



def print_probabilities(f, fontsize=14, display=True, cmap=None):
    """
    Print probabilistic assignment maps.

    Input:  - f     Input file
    """

    plt.rcParams.update({"font.size":14})

    # Load the file
    with open(f, "r") as F:
        lines = F.read().split("\n")

    # Get experimental shifts, nuclei labels and assignment probabilities
    exps, labels, probs = load_prob_file(f)

    # Split by number of equivalent nuclei/distributions
    for k in exps.keys():

        # Sort the experimental shifts by decreasing value
        num_exps = [float(l.split("/")[0].split("\\")[0]) for l in exps[k]]
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
        
        # Rearrange the probability matrix according to the changes made to the order
        #     of experimental shifts and labels
        sorted_probs = np.zeros_like(probs[k])
        
        for i, i2 in enumerate(sorted_lab_inds):
            for j, j2 in enumerate(sorted_exp_inds):
                
                sorted_probs[i, j] = probs[k][i2, j2]

        # Get the figure size (proportional to the number of experimental shifts and labels)
        le = 0
        ll = 0
        for e in exps[k]:
            le = max(le, len(e))
        for l in labels[k]:
            ll = max(ll, len(l))

        lx = (ll + 3) * fontsize / 100
        ly = (le + 2) * fontsize / 100
        
        lx += (len(exps[k]) + 1) * 0.5
        ly += len(labels[k]) * 0.5

        # Initialize figure handle
        fig = plt.figure(figsize=(lx, ly))
        ax = fig.add_subplot()

        # Plot the probability map
        if cmap is not None:
            c = ax.pcolormesh(sorted_probs, cmap=cmap, vmin=0., vmax=100., edgecolors=(0.9, 0.9, 0.9), linewidths=1)
        else:
            c = ax.pcolormesh(sorted_probs, cmap=WOrBr, vmin=0., vmax=100., edgecolors=(0.9, 0.9, 0.9), linewidths=1)
        fig.colorbar(c)

        x_ticks = np.array(range(sorted_probs.shape[1])) + 0.5
        y_ticks = np.array(range(sorted_probs.shape[0])) + 0.5

        # Set tick labels of x (experimental shifts) and y (nuclei labels) axes
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

        ax.set_xticklabels(sorted_exps, rotation=60, ha="right", va="center_baseline", rotation_mode="anchor")
        ax.set_yticklabels(sorted_labs)

        # Save the figure
        fig.tight_layout()
        fig.savefig(f.replace(".dat", "_{}.pdf".format(k)))

        if display:
            plt.show()
            
        plt.close()
        
    return


##############################################################################################################################
