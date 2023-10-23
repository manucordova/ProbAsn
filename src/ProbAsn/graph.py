###############################################################################
#                                                                             #
#                Functions for graph handling (creation, ...)                 #
#                        Author: Manuel Cordova (EPFL)                        #
#                          Last modified: 21.09.2023                          #
#                                                                             #
###############################################################################

# Import libraries
import numpy as np
import ase.data
import networkx as nx
from openbabel import pybel as pb
import matplotlib.pyplot as plt

# Check for MolDS package (faster distance matrix implementation)
has_molds = True
try:
    import MolDS.structure as st
except ImportError:
    has_molds = False

# Default values
max_valence_default = {
    "H": 1,
    "C": 4,
    "N": 4,
    "O": 2,
    "S": 6,
    "F": 1,
    "P": 5,
    "Cl": 1,
    "Na": 1,
    "Ca": 2,
    "Mg": 2,
    "K": 1
}

cutoffs_default = {}
for e, i in ase.data.atomic_numbers.items():
    cutoffs_default[e] = ase.data.covalent_radii[i]


def get_bonds_in_cryst(
    struct,
    cutoffs=None,
    cutoff_factor=1.1,
    max_valence=None
):
    """Get the bonds in a crystal structure.

    Parameters
    ----------
    struct : ase Atoms
        Input structure
    cutoffs : None or dict, default=None
        Covalent radius for each element,
        used to identify pairs of bonded atoms.
        If `None`, use default ASE covalent radii.
    cutoff_factor : float, default=1.1
        Tolerance factor to consider pairs of atoms as bonded.
        An atom pair a-b will be considered bonded if their distance
        is smaller than `(cutoffs[a] + cutoffs[b]) * cutoff_factor`
    max_valence : None or dict, default=None
        Maximum valence of each element. if `None`,
        use default values.

    Returns
    -------
    sym : list
        List of chemical symbol for each atom in the molecule.
    bonds : list
        List lists of atom indices bonded to the corresponding atom.
    """

    # Get default values if not set
    if cutoffs is None:
        cutoffs = cutoffs_default
    if max_valence is None:
        max_valence = max_valence_default

    # Get distance matrix
    sym = struct.get_chemical_symbols()
    if has_molds:
        _, D = st.get_distances(struct, mic=True)
    else:
        D = struct.get_all_distances(mic=True)

    # Get matrix of cutoff distances to consider bonds
    dmin = np.array([cutoffs[s] for s in sym])
    Dmin = np.add.outer(dmin, dmin) * cutoff_factor

    # Identify covalent contacts
    contacts = D < Dmin
    contacts[np.diag_indices_from(contacts)] = False

    # Extract bonds from the matrix of contacts
    bonds = [[] for _ in sym]
    for i, s in enumerate(sym):
        bonds[i] = [j for j in range(len(sym)) if contacts[i, j]]

    # Comply with maximum valence
    for i, (s, b) in enumerate(zip(sym, bonds)):
        if s in max_valence and len(b) > max_valence[s]:
            ds = [D[i, j] for j in b]
            to_remove = [b[j] for j in np.argsort(ds)[max_valence[s]:]]
            for j in to_remove:
                bonds[i].remove(j)
                bonds[j].remove(i)

    return sym, bonds


def make_mol(
    mol_in,
    in_type="smi",
    out_type="mol",
    from_file=False,
    name="",
    make_3d=False,
    save=None
):
    """Build a Pybel molecule object.

    Parameters
    ----------
    mol_in : str
        Input molecule or file path.
    in_type : str, default="smi"
        Type (format) of input.
    out_type : str, default="mol"
        Type (format) for saving output file.
    from_file : bool, default=False
        If `True`, then `mol_in` is the path to the file to read.
        Otherwise, `mol_in` contains the data to read.
    name : str, default=""
        Name of the molecule.
    make_3d : bool, default=False
        Build a 3D conformer of the molecule in the file to save.
        If `False`, a 2D conformer will be built instead.
    save : None or str, default=None
        File path to save the output. If `None`, no output will be saved.

    Returns
    -------
    mol : OBMol
        Molecule extracted from the input.
    """

    # Read the input
    if from_file:
        mol = next(pb.readfile(in_type, mol_in))
    else:
        mol = pb.readstring(in_type, mol_in)

    # Set molecule name
    mol.title = name

    # Add implicit hydrogens
    has_h = False
    for atom in mol.atoms:
        if atom.type == "H":
            has_h = True
            break
    if not has_h:
        mol.addh()

    # Make 2D/3D coordinates
    if make_3d:
        mol.make3D()
    else:
        mol.make2D()

    # Convert the molecule into the output format
    pp = mol.write(out_type)

    # Save molecule to file
    if save is not None:
        with open(save, "w") as F:
            F.write(pp)

    return mol


def get_bonds(mol):
    """Identify the bonds in a molecule.

    Parameters
    ----------
    mol : OBMol
        Input molecule.

    Returns
    -------
    atoms : list
        List of atoms in the molecule.
    bonds : list
        List of lists of bonded atomic indices for each atom in the molecule
    """

    # Get molfile of the molecule
    pp = mol.write("mol")
    lines = pp.split("\n")

    # Get number of atoms
    n_atoms = len(mol.atoms)
    # Initialize arrays of neighbours
    bonds = [[] for _ in mol.atoms]
    # Initialize array of atoms
    atoms = []

    # Old mol format
    if lines[3].split()[-1] == "V2000":

        # Parse the atom block
        for line in lines[4:n_atoms + 4]:
            atoms.append(line.split()[3])

        # Parse the bond block
        for line in lines[n_atoms + 4:]:

            # Detect end of file
            if (
                "END" in line or
                "CHG" in line or
                len(line.split()) not in [6, 7]
            ):
                break

            bond = [int(line[:3])-1, int(line[3:6])-1]

            # Update the list of bonds
            bonds[bond[0]].append(bond[1])
            bonds[bond[1]].append(bond[0])

    # New mol format
    elif lines[3].split()[-1] == "V3000":

        # Parse the atom block
        atom_block = False
        bond_block = False
        for line in lines:
            if "BEGIN ATOM" in line:
                atom_block = True
                continue
            elif "END ATOM" in line:
                atom_block = False
                continue
            if "BEGIN BOND" in line:
                bond_block = True
                continue
            elif "END BOND" in line:
                bond_block = False
                continue

            if atom_block:
                atoms.append(line.split()[3])
            elif bond_block:
                bond = [int(line.split()[-2])-1, int(line.split()[-1])-1]
                bonds[bond[0]].append(bond[1])
                bonds[bond[1]].append(bond[0])
    else:
        raise ValueError(f"Unknown molfile version: {lines[3].split()[-1]}")

    return atoms, bonds


def identify_env(G):
    """Identify the environment of the central node (index 0) of a graph.

    Parameters
    ----------
    G : Networkx Graph
        Input graph.

    Returns
    -------
    env : str
        Environment of the central node in G.
    """

    # Initialize array of neighbouring elements
    nei_elems = []
    # Identify all nodes bonded to the central node
    for e in G.edges:

        if 0 in e:
            # Get neighbour atom
            if e[0] == 0:
                i = e[1]
            else:
                i = e[0]
            # Update array of neighbouring elements
            nei_elems.append(G.nodes[i]["elem"])

    # Return the environment in string format,
    # with neighbours sorted alphabetically
    return "-".join(sorted(nei_elems))


def generate_graph(
    atoms,
    bonds,
    i0,
    max_w=6,
    elems=["H", "C", "N", "O", "S", "F", "P", "Cl", "Na", "Ca", "Mg", "K"],
    hetatm="error",
    hetatm_rep=None
):
    """Generate a graph centered on atom i0.

    Parameters
    ----------
    atoms : list
        List of atoms in the molecule.
    bonds : list
        List of lists of atomic indices bonded to the
        corresponding atom in the molecule.
    i0 : int
        Index of the atom to use as the central node in the graph.
    max_w : int, default=6
        Maximum graph depth.
    elems : list, default=["H", "C", "N", "O", "S", "F",
    "P", "Cl", "Na", "Ca", "Mg", "K"]
        List of allowed elements in the molecule.
    hetatm : str, default="error"
        Behaviour for handling unknown elements. `"error"` raises an error,
        `"ignore"`: ignores the unknown atoms, `"replace"` replaces the atom
        with another element (set in `hetatm_rep`), `"replace_and_terminate"`
        replaces the atom and cuts all bonds from this atom.
    hetatm_rep: None or dict, default=None
        Dictionary used to replace unknown elements.
        The key should be the element to replace,
        and the value should be the element to replace with.

    Returns
    -------
    G : Networkx Graph
        Generated graph.
    env : str
        Environment of the central node.
    """

    # The maximum depth should be at least one
    if max_w < 1:
        raise ValueError(f"max_w should be at least 1, not {max_w}")

    # Initialize graph object
    G = nx.Graph()
    # Add central node
    G.add_node(0, elem=atoms[i0], ind=i0)

    # Initialize number of nodes in the graph and atom index of each node
    num_nodes = 1
    node_inds = [i0]

    # Loop over all nodes
    i = 0
    while i < num_nodes:
        # Identify the atoms bonded to that node
        for j in bonds[node_inds[i]]:

            at = atoms[j]
            # Handle invalid elements
            if at not in elems:
                # Raise an error
                if hetatm == "error":
                    raise ValueError(f"Invalid element found: {at}")
                # Ignore the atom
                elif hetatm == "ignore":
                    continue
                # Replace the atom with another element
                elif hetatm == "replace":
                    at = hetatm_rep[at]
                # Replace the atom with another element and
                # cut all bonds from this atom
                elif hetatm == "replace_and_terminate":
                    at = hetatm_rep[at]
                    bonds[j] = []
                else:
                    raise ValueError(
                        f"Invalid behaviour for unknown elements: {hetatm}"
                    )

            # If a new node is found, add it to the graph
            if j not in node_inds:
                G.add_node(num_nodes, elem=at, ind=j)
                G.add_edge(i, num_nodes, w="1")

                # If the new node is too far away, remove it
                path_length = nx.shortest_path_length(
                    G,
                    source=0,
                    target=num_nodes
                )
                if path_length > max_w:
                    G.remove_node(num_nodes)
                # Otherwise, keep it and update the total number of nodes
                # and the atom index of each node
                else:
                    num_nodes += 1
                    node_inds.append(j)
            # If the bonde node is already in the graph, just add the edge
            else:
                G.add_edge(i, node_inds.index(j), w="1")

        # Proceed to the next node
        i += 1

    # Get the environment of the central node
    env = identify_env(G)

    return G, env


def generate_graphs(
    atoms,
    bonds,
    elem,
    max_w=6,
    elems=["H", "C", "N", "O", "S", "F", "P", "Cl", "Na", "Ca", "Mg", "K"],
    hetatm="error",
    hetatm_rep=None
):
    """Generate graphs for all atoms of a given element in the molecule.

    Parameters
    ----------
    atoms : list
        List of atoms in the molecule.
    bonds : list
        List of lists of atomic indices bonded to the
        corresponding atom in the molecule.
    elem : str
        Element to construct the graphs around.
    max_w : int, default=6
        Maximum graph depth.
    elems : list, default=["H", "C", "N", "O", "S", "F",
    "P", "Cl", "Na", "Ca", "Mg", "K"]
        List of allowed elements in the molecule.
    hetatm : str, default="error"
        Behaviour for handling unknown elements. `"error"` raises an error,
        `"ignore"`: ignores the unknown atoms, `"replace"` replaces the atom
        with another element (set in `hetatm_rep`), `"replace_and_terminate"`
        replaces the atom and cuts all bonds from this atom.
    hetatm_rep: None or dict, default=None
        Dictionary used to replace unknown elements.
        The key should be the element to replace,
        and the value should be the element to replace with.

    Returns
    -------
    Gs : list
        List of all graphs for the defined element.
    envs : list
        List of all environments for the defined element.
    """

    # Initialize arrays of graphs and environments
    Gs = []
    envs = []

    # Loop over all atoms
    for i, at in enumerate(atoms):
        # Identify the atoms for which a graph should be constructed
        if at == elem:
            # Construct the graph
            G, env = generate_graph(
                atoms,
                bonds,
                i,
                max_w=max_w,
                elems=elems,
                hetatm=hetatm,
                hetatm_rep=hetatm_rep
            )

            Gs.append(G)
            envs.append(env)

    return Gs, envs


def cut_graph(G, w):
    """Cut a graph down to a given depth.

    Parameters
    ----------
    G : Networkx Graph
        Input graph to cut.
    w : int
        Depth to cut the graph to.

    Returns
    -------
    cut_G : Networkx Graph
        Graph cut to the desired depth.
    """

    # Copy the initial graph and get the number of nodes
    cut_G = G.copy()
    N = len(G.nodes)

    # Check all nodes
    for i in range(N):
        # If the depth is greater than w, remove the node
        if nx.shortest_path_length(G, source=0, target=i) > w:
            cut_G.remove_node(i)

    return cut_G


def generate_hash(G):
    """Generate the Weisfeiler-Lehman hash corresponding to a graph.

    Parameters
    ----------
    G : Networkx Graph
        Input graph.

    Returns
    -------
    hash : str
        Hash corresponding to the graph `G`.
    """

    # Replace the central element by "Y" in order to make sure that
    # the hash correctly identifies the central node
    G2 = G.copy()
    G2.nodes[0]["elem"] = "Y"

    return nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(
        G2,
        edge_attr="w",
        node_attr="elem",
        iterations=5
    )


def print_graph(
    G,
    w,
    layout="kamada_kawai",
    base_color="C0",
    center_color="r",
    out_color="g",
    show=True,
    save=None
):
    """Plot a graph at a given depth.

    Parameters
    ----------
    G : Networkx graph
        Graph to plot.
    w : int
        Graph depth to plot.
    layout : str, default="kamada_kawai"
        Layout used to place the nodes.
        Allowed values: "kamada_kawai", "circular", "planar", "random",
        "shell", "spring", "spectral", "spiral".
    base_color : str, default="C0"
        Color of the nodes.
    center_color : str, default="r"
        Color of the central node.
    out_color : str, default="g"
        Color of the nodes at the maximum graph depth.
    show : bool, default=True
        Show the plot.
    save : None or str, default=None
        File path to save the plot to.
    """

    # Cut the graph to the maximum depth
    cut_G = cut_graph(G, w)

    # Get the label (element) of each node
    labs = nx.get_node_attributes(cut_G, "elem")

    # Set node layout
    if layout == "kamada_kawai":
        # Get the position of each node on the plot
        pos = nx.kamada_kawai_layout(cut_G)
    elif layout == "circular":
        pos = nx.circular_layout(cut_G)
    elif layout == "planar":
        pos = nx.planar_layout(cut_G)
    elif layout == "random":
        pos = nx.random_layout(cut_G)
    elif layout == "shell":
        pos = nx.shell_layout(cut_G)
    elif layout == "spring":
        pos = nx.spring_layout(cut_G)
    elif layout == "spectral":
        pos = nx.spectral_layout(cut_G)
    elif layout == "spiral":
        pos = nx.spiral_layout(cut_G)
    else:
        raise ValueError(f"Unknown layout: {layout}")

    # Initialize figure handle
    f = plt.figure(figsize=(6, 5))
    ax = f.add_subplot(1, 1, 1)

    # Draw the nodes
    nx.draw_networkx_nodes(
        cut_G,
        pos,
        ax=ax,
        node_color=base_color
    )

    # Draw the central node in red
    nx.draw_networkx_nodes(
        cut_G,
        pos,
        nodelist=[0],
        node_color=center_color,
        ax=ax
    )

    # Draw the edge nodes in green
    edge_nodes = []
    for i in range(cut_G.number_of_nodes()):
        if nx.shortest_path_length(cut_G, source=0, target=i) == w:
            edge_nodes.append(i)
    nx.draw_networkx_nodes(
        cut_G,
        pos,
        nodelist=edge_nodes,
        node_color=out_color,
        ax=ax
    )

    # Draw the edges
    nx.draw_networkx_edges(cut_G, pos, edgelist=cut_G.edges, ax=ax)

    # Draw the labels on the nodes
    nx.draw_networkx_labels(cut_G, pos, labs, ax=ax)

    f.tight_layout()

    # Show the plot
    if show:
        plt.show()

    # Save the plot
    if save:
        if save.endswith(".png"):
            f.savefig(save, dpi=150)
        else:
            f.savefig(save)
    plt.close()
    return
