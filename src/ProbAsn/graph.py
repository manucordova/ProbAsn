####################################################################################################
###                                                                                              ###
###                        Functions for graph handling (creation, ...)                          ###
###                               Author: Manuel Cordova (EPFL)                                  ###
###                                Last modified: 19.09.2022                                     ###
###                                                                                              ###
####################################################################################################

# Import libraries
import numpy as np
import ase.data
import networkx as nx
from openbabel import pybel as pb
import matplotlib.pyplot as plt
has_molds=True
try:
    import MolDS.structure as st
except ImportError:
    has_molds=False



max_valence_default = {
    "H": 1,
    "C": 4,
    "N": 4,
    "O": 2,
    "S": 6,
}

cutoffs_default = {}
for e, i in ase.data.atomic_numbers.items():
    cutoffs_default[e] = ase.data.covalent_radii[i]



def get_bonds_in_cryst(struct, cutoffs=None, cutoff_factor=1.1, max_valence=None):

    if cutoffs is None:
        cutoffs = cutoffs_default
    if max_valence is None:
        max_valence = max_valence_default

    sym = struct.get_chemical_symbols()
    if has_molds:
        _, D = st.get_distances(struct, mic=True)
    else:
        D = struct.get_all_distances(mic=True)

    dmin = np.array([cutoffs[s] for s in sym])
    Dmin = np.add.outer(dmin, dmin) * cutoff_factor

    contacts = D < Dmin
    contacts[np.diag_indices_from(contacts)] = False

    bonds = [[] for _ in sym]
    for i, s in enumerate(sym):
        bonds[i] = [j for j in range(len(sym)) if contacts[i, j]]
    
    for i, (s, b) in enumerate(zip(sym, bonds)):
        if s in max_valence and len(b) > max_valence[s]:
            ds = [D[i, j] for j in b]
            to_remove = [b[j] for j in np.argsort(ds)[max_valence[s]:]]
            for j in to_remove:
                bonds[i].remove(j)
                bonds[j].remove(i)

    return sym, bonds



def make_mol(mol_in, in_type="smi", out_type="mol", from_file=False, name="", make_3d=False, save=None):
    """
    Generate a molfile from the SMILES representation of the molecule

    Inputs: - mol_in        Input molecule
            - in_type       Type of input
            - out_type      Type of output
            - from_file     Whether mol_in is a string (False) or points to a file (True)
            - name          Name of the molecule
            - make_3d       Whether to make
            - save          If set, defines the file to which the molfile should be saved

    Output: - mol           OBMol object of the molecule
    """

    if from_file:
        # Read the input file
        mol = next(pb.readfile(in_type, mol_in))
    else:
        # Read the input string
        mol = pb.readstring(in_type, mol_in)

    # Set molecule name
    mol.title = name

    # Add implicit hydrogens
    has_h =False
    for atom in mol.atoms:
        if atom.type == "H":
            has_h = True
            break
    if not has_h:
        mol.addh()

    if make_3d:
        # Make 3D coordinates
        mol.make3D()
    else:
        # Make 2D coordinates for drawing
        mol.make2D()

    # Convert the molecule into the output format
    pp = mol.write(out_type)

    # Save molecule to file
    if save is not None:
        with open(save, "w") as F:
            F.write(pp)

    return mol



def get_bonds(mol):
    """
    Identify the bonds in a molecule

    Input:      - mol       OBMol object of the molecule

    Outputs:    - atoms     List of atoms in the molecule
                - bonds     List of bonded atoms for each atom in the molecule
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

    if lines[3].split()[-1] == "V2000":

        # Parse the atom block
        for l in lines[4:n_atoms + 4]:
            atoms.append(l.split()[3])

        # Parse the bond block
        for l in lines[n_atoms + 4:]:

            # Detect end of file
            if "END" in l or "CHG" in l or len(l.split()) not in [6, 7]:
                break

            bond = [int(l[:3])-1, int(l[3:6])-1]

            # Update the list of bonds
            bonds[bond[0]].append(bond[1])
            bonds[bond[1]].append(bond[0])
    
    elif lines[3].split()[-1] == "V3000":

        # Parse the atom block
        atom_block = False
        bond_block = False
        for l in lines:
            if "BEGIN ATOM" in l:
                atom_block = True
                continue
            elif "END ATOM" in l:
                atom_block = False
                continue
            if "BEGIN BOND" in l:
                bond_block = True
                continue
            elif "END BOND" in l:
                bond_block = False
                continue

            if atom_block:
                atoms.append(l.split()[3])
            elif bond_block:
                bond = [int(l.split()[-2])-1, int(l.split()[-1])-1]
                bonds[bond[0]].append(bond[1])
                bonds[bond[1]].append(bond[0])
    else:
        raise ValueError(f"Unknown molfile version: {lines[3].split()[-1]}")

    return atoms, bonds



def identify_env(G):
    """
    Identify the environment of the central node (index 0) of a graph

    Input:  - G     Input graph

    Output: - env   environment of the central node in G
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
    #   with neighbours sorted alphabetically
    return "-".join(sorted(nei_elems))



def generate_graph(atoms, bonds, i0, max_w, elems=["H", "C", "N", "O", "S"], hetatm="error", hetatm_rep=None):
    """
    Generate a graph from atom i0 using the list of atoms and bonds in the molecule

    Inputs:     - atoms             List of atoms in the molecule
                - bonds             Bonded atoms for each atom in the molecule (by index)
                - i0                Index of the central atom in the graph
                - max_w             Maximum graph depth
                - elems             Allowed elements in the molecule
                - hetatm            Behaviour for handling unknown elements:
                                        "error": raise an error
                                        "ignore": ignore the atom
                                        "replace": replace the atom with another element
                                        "replace_and_terminate": replace the atom with another element and
                                            cut all bonds from this atom
                - hetatm_rep        Dictionary of replacements for unknown elements
                                        (used only with hetatm set to "replace" or "replace_and_terminate")

    Outputs:    - G                 Graph generated
                - env               Environment of the central node
    """

    # The maximum depth should be at least one
    if max_w < 1:
        raise ValueError("max_weight should be at least 1, not {}".format(max_w))

    # Initialize graph object
    G = nx.Graph()
    # Add central node
    G.add_node(0, elem=atoms[i0], ind=i0)

    # Initialize number of nodes in the graph and atom index of each node
    N = G.number_of_nodes()
    node_inds = [G.nodes[i]["ind"] for i in range(N)]

    # Loop over all nodes
    i = 0
    while i < N:
        # Identify the atoms bonded to that node
        for j in bonds[node_inds[i]]:

            at = atoms[j]
            # Handle invalid elements
            if at not in elems:
                # Raise an error
                if hetatm == "error":
                    raise ValueError("Invalid element found: {}".format(at))
                # Ignore the atom
                elif hetatm == "ignore":
                    continue
                # Replace the atom with another element
                elif hetatm == "replace":
                    at = hetatm_rep[at]
                # Replace the atom with another element and cut all bonds from this atom
                elif hetatm == "replace_and_terminate":
                    at = hetatm_rep[at]
                    bonds[j] = []
                else:
                    raise ValueError("Invalid behaviour for unknown elements: {}".format(hetatm))

            # If a new node is found, add it to the graph
            if j not in node_inds:
                G.add_node(N, elem=at, ind=j)
                G.add_edge(i, N, w="1")

                # If the new node is too far away, remove it
                if nx.shortest_path_length(G, source=0, target=N) > max_w:
                    G.remove_node(N)
                # Otherwise, keep it and update the total number of nodes and the atom index of each node
                else:
                    N += 1
                    node_inds = [G.nodes[i]["ind"] for i in range(N)]
            # If the bonde node is already in the graph, just add the edge
            else:
                G.add_edge(i, node_inds.index(j), w="1")

        # Proceed to the next node
        i += 1

    # Get the environment of the central node
    env = identify_env(G)

    return G, env



def generate_graphs(atoms, bonds, elem, max_w, elems=["H", "C", "N", "O", "S"], hetatm="error", hetatm_rep=None):
    """
    Generate graphs for all atoms of a given element in the molecule

    Inputs:     - atoms             List of atoms in the molecule
                - bonds             Bonded atoms for each atom in the molecule (by index)
                - elem              Element for which to construct the graphs
                - max_w             Maximum graph depth
                - elems             Allowed elements in the molecule
                - hetatm            Behaviour for handling unknown elements:
                                        "error": raise an error
                                        "ignore": ignore the atom
                                        "replace": replace the atom with another element
                                        "replace_and_terminate": replace the atom with another element and
                                            cut all bonds from this atom
                - hetatm_rep        Dictionary of replacements for unknown elements
                                        (used only with hetatm set to "replace" or "replace_and_terminate")

    Outputs:    - Gs
                - envs
    """

    # Initialize arrays of graphs and environments
    Gs = []
    envs = []

    # Loop over all atoms
    for i, at in enumerate(atoms):
        # Identify the atoms for which a graph should be constructed
        if at == elem:
            # Construct the graph
            G, env = generate_graph(atoms, bonds, i, max_w, elems=elems, hetatm=hetatm, hetatm_rep=hetatm_rep)
            Gs.append(G)
            envs.append(env)

    return Gs, envs



def cut_graph(G, w):
    """
    Cut a graph down to a given depth

    Inputs: - G         Input graph
            - w         Depth to cut to

    Output: - cut_G     Cut graph
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
    """
    Generate the Weisfeiler-Lehman hash corresponding to a graph

    Input:  - G     Input graph

    Output: - H     Hash corresponding to graph G
    """

    # Replace the central element by "Y" in order to make sure that the hash correctly identifies the central node
    G2 = G.copy()
    G2.nodes[0]["elem"] = "Y"

    return nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(G2, edge_attr="w", node_attr="elem", iterations=5)



def print_graph(G, w, layout="kamada_kawai", base_color="C0", center_color="r", out_color="g", show=True, save=None):
    """
    Plot a graph at a given depth

    Inputs: - G                 Networkx graph object
            - w                 Maximum depth to display
            - layout            Node layout for plotting
            - base_color        Base color of nodes
            - center_color      Color of the central node
            - out_color         Color of nodes at the maximum graph depth
            - show              Whether the plot should be shown or not
            - save              If set, defines the file to save the plot to
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
        raise ValueError("Unknown layout: {}".format(layout))

    # Initialize figure handle
    f = plt.figure(figsize=(6,5))
    ax = f.add_subplot(1,1,1)

    # Draw the nodes
    nx.draw_networkx_nodes(cut_G, pos, ax=ax, node_color=base_color)

    # Draw the central node in red
    nx.draw_networkx_nodes(cut_G, pos, nodelist=[0], node_color=center_color, ax=ax)

    # Draw the edge nodes in green
    edge_nodes = []
    for i in range(cut_G.number_of_nodes()):
        if nx.shortest_path_length(cut_G, source=0, target=i) == w:
            edge_nodes.append(i)
    nx.draw_networkx_nodes(cut_G, pos, nodelist=edge_nodes, node_color=out_color, ax=ax)

    # Get the covalent bonds and H-bonds
    H_e = []
    es = []
    for i, e in enumerate(cut_G.edges):
        if cut_G.edges[e]["w"] == 0:
            H_e.append(e)
        else:
            es.append(e)

    # Draw the edges
    nx.draw_networkx_edges(cut_G, pos, edgelist=es, ax=ax)
    nx.draw_networkx_edges(cut_G, pos, edgelist=H_e, style="dashed", ax=ax)

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
