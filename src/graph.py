####################################################################################################
###                                                                                              ###
###                        Functions for graph handling (creation, ...)                          ###
###                               Author: Manuel Cordova (EPFL)                                  ###
###                                Last modified: 10.05.2021                                     ###
###                                                                                              ###
####################################################################################################

# Import libraries
import numpy as np
import os
import sys
import ase
import ase.io
import ase.neighborlist
import networkx as nx
from openbabel import pybel as pb
import itertools as it
import matplotlib.pyplot as plt



def generate_conformer(smi):
    """
    Generate a gas-phase conformer of a molecule from its SMILES representation
    
    Input:  - smi       SMILES representation of the molecule
    
    Output: - struct    ASE object containing the conformer generated
    """
    
    # Read the string and convert to mol format
    mol = pb.readstring("smi", smi)
    # Add implicit hydrogens
    mol.addh()
    # Generate 3D coordinates
    mol.make3D()
    # Optimize the conformer using force-field
    mol.localopt(forcefield="mmff94", steps=1000)
    
    # Get the molecule in xyz format
    pp = mol.write("xyz")
    lines = pp.split("\n")
    
    # Initialize arrays of atomic symbols and positions
    symbs = []
    pos = []
    
    # Get number of atoms in the molecule
    n_atoms = int(lines[0])
    
    # Get the coordinates and elements of each atom
    for i in range(n_atoms):
        tmp = lines[i+2].split()
        symbs.append(tmp[0])
        pos.append([float(tmp[1]), float(tmp[2]), float(tmp[3])])
    
    # Return ASE object
    return ase.Atoms(symbols=symbs, positions=pos)



def get_cutoffs(elems, factor):
    """
    Get cutoffs for identifying covalent bonds
    
    Inputs: - elems     Elements present in the molecule
            - factor    Tolerance factor for the identification of covalent bonds
    
    Output: - cutoffs   Cutoffs for every possible pair of elements in the structure
    """
    
    # Initialize dictionnary of cutoffs
    cutoffs = {}
    
    # Loop over all pairs of elements:
    for e1 in elems:
        for e2 in elems:
            # Get the cutoff as the (sum of covalent radii) * factor
            cutoffs[(e1, e2)] = round((ase.data.covalent_radii[ase.data.atomic_numbers[e1]]+ase.data.covalent_radii[ase.data.atomic_numbers[e2]])*factor, 6)
    
    return cutoffs



def get_neighbours(struct, cutoffs):
    """
    Obtain covalent bonds from a 3D structure (crystal or single molecule)
    
    Inputs: - struct    ASE object of the 3D structure
            - cutoffs   Cutoffs for each pair of atoms

    Output: - neis      Neighbours of each atom in the structure
    """
    
    # Get neighbour list
    a, b = ase.neighborlist.neighbor_list("ij", struct, cutoffs, self_interaction=False)
    
    # Initialize array of neighbours
    neis = []
    
    # Append neighbours
    for i in range(len(struct)):
        ia = np.where(a == i)
        neis.append(list(b[ia]))
    return neis



def get_envs_from_smiles(smi, elems=["H", "C", "N", "O", "S"], cutoff_factor=1.1):
    """
    Obtain topological chemical environments from a SMILES string
    
    Inputs:     - smi               SMILES representation of the molecule
                - elems             Possible elements in the structure
                - cutoff_factor     Factor for the tolerance of covalent bonds
                
    Outputs:    - struct            Structure of the conformer generated
                - neighbour_inds    Neighbouring indices for each atom
                - neighbour_atoms   Neighbouring atoms for each atom
                - envs              Environments (first neighbours) of each atom
    """
    
    # First, generate the 3D coordinates (in the gas phase) of the molecule
    struct = generate_conformer(smi)
    
    # Get cutoffs for the neighbour lists:
    cutoffs = get_cutoffs(elems, cutoff_factor)
    
    # Initialize neighbour lists for each element
    neighbour_inds = {}
    neighbour_atoms = {}
    envs = {}
    
    for e in elems:
        neighbour_inds[e] = []
        neighbour_atoms[e] = []
        envs[e] = []
    
    # Get the elements in the structure
    symbs = struct.get_chemical_symbols()
    
    # Construct neighbour list
    neis = get_neighbours(struct, cutoffs)
    
    # Loop over all atoms in the structure
    for i, s in enumerate(symbs):
        # Get the indices of the neighbours of atom i
        inds_nei = neis[i]
        
        # Get the elements of the neighbours
        elems_nei = [symbs[k] for k in inds_nei]
        
        # Get the indices of individual elements [k-th carbon, ...]
        sep_inds_nei = []
        for k, e in zip(inds_nei, elems_nei):
            sep_inds_nei.append(symbs[:k].count(e))
        
        # Initialize arrays of neighbours for atom i
        sorted_elems_nei = []
        sorted_inds_nei = []
        sorted_sep_inds_nei = []
        
        # Standardize the neighbours to be in alphabetical order of element
        for e, j in sorted(zip(elems_nei, range(len(elems_nei)))):
            sorted_elems_nei.append(e)
            sorted_inds_nei.append(inds_nei[j])
            sorted_sep_inds_nei.append(sep_inds_nei[j])
        
        # Get the environment name
        nei_str = "-".join(sorted_elems_nei)
        
        # Append the neighbours to the list
        neighbour_inds[s].append(sorted_sep_inds_nei)
        neighbour_atoms[s].append(sorted_elems_nei)
        envs[s].append(nei_str)
    
    return struct, neighbour_inds, neighbour_atoms, envs



def generate_neighbour_list(struct, elems=["H", "C", "N", "O", "S"], cutoff_factor=1.1):
    """
    Obtain topological chemical environments from a SMILES string
    
    Inputs:     - struct            ASE object of the molecule
                - elems             Possible elements in the structure
                - cutoff_factor     Factor for the tolerance of covalent bonds
                
    Outputs:    - struct            Structure of the conformer generated
                - neighbour_inds    Neighbouring indices for each atom
                - neighbour_atoms   Neighbouring atoms for each atom
                - envs              Environments (first neighbours) of each atom
    """
    
    # Get cutoffs for the neighbour lists:
    cutoffs = get_cutoffs(elems, cutoff_factor)
    
    # Initialize neighbour lists for each element
    neighbour_inds = {}
    neighbour_atoms = {}
    envs = {}
    
    for e in elems:
        neighbour_inds[e] = []
        neighbour_atoms[e] = []
        envs[e] = []
    
    # Get the elements in the structure
    symbs = struct.get_chemical_symbols()
    
    # Construct neighbour list
    neis = get_neighbours(struct, cutoffs)
    
    # Loop over all atoms in the structure
    for i, s in enumerate(symbs):
        # Get the indices of the neighbours of atom i
        inds_nei = neis[i]
        
        # Get the elements of the neighbours
        elems_nei = [symbs[k] for k in inds_nei]
        
        # Get the indices of individual elements [k-th carbon, ...]
        sep_inds_nei = []
        for k, e in zip(inds_nei, elems_nei):
            sep_inds_nei.append(symbs[:k].count(e))
        
        # Initialize arrays of neighbours for atom i
        sorted_elems_nei = []
        sorted_inds_nei = []
        sorted_sep_inds_nei = []
        
        # Standardize the neighbours to be in alphabetical order of element
        for e, j in sorted(zip(elems_nei, range(len(elems_nei)))):
            sorted_elems_nei.append(e)
            sorted_inds_nei.append(inds_nei[j])
            sorted_sep_inds_nei.append(sep_inds_nei[j])
        
        # Get the environment name
        nei_str = "-".join(sorted_elems_nei)
        
        # Append the neighbours to the list
        neighbour_inds[s].append(sorted_sep_inds_nei)
        neighbour_atoms[s].append(sorted_elems_nei)
        envs[s].append(nei_str)
    
    return neighbour_inds, neighbour_atoms, envs



def get_graph_neighbours(center, nei_inds, nei_atoms):
    """
    Obtain neighbours of node "center" in a graph
    
    Inputs: - center        Tuple (element, index) of the central node
            - nei_inds      List of neighbouring indices
            - nei_atoms     List of neighbouring chemical symbols
            
    Output: - neis          List of neighbours of the central node
    """
    
    # Initialize the array of neighbours
    neis = []
    
    # Retrieve element and index of the central node
    c_e = center[0]
    c_i = center[1]
    
    # Obtain all neighbours
    for n, e in zip(nei_inds[c_e][c_i], nei_atoms[c_e][c_i]):
        neis.append((e, n))
    
    return neis



def generate_graph(nodes, bonds):
    """
    Generate a graph given a list of nodes and bonds
    
    Inputs: - nodes     List of tuples containing the elements and indices of the atoms in the graph
            - bonds     Bonds to generate edges of the graph
            
    Output: - G         networkx graph object
    """
    
    # Initialize networkx graph object
    G = nx.Graph()
    
    # Append each node to the graph
    for i, n in enumerate(nodes):
        G.add_node(i, elem=n[0], ind=n[1])
            
    # Get list of indices of the nodes
    inds = nx.get_node_attributes(G, "ind")
    
    # Loop over all pairs of nodes
    for i, n in enumerate(nodes):
        for j, m in enumerate(nodes[i+1:]):
            pair = (n, m)
            for p in it.permutations(pair):
                # If there is a bond between the two nodes, add an edge to the graph
                if p in bonds:
                    G.add_edge(i, i+j+1, w="1")
                    break
    return G



def get_nodes(G):
    """
    Get the nodes in a graph
    
    Inputs: - G         Input graph
    
    Output: - nodes     List of nodes in the graph G
    """
    
    # Initialize the array of nodes
    nodes = []
    # Get the elements and indices of each node
    elems = nx.get_node_attributes(G, "elem")
    inds = nx.get_node_attributes(G, "ind")
    
    # Generate the array of nodes
    for i in range(G.number_of_nodes()):
        nodes.append((elems[i], inds[i]))
    return nodes



def extend_graph(G, nei_inds, nei_ats, max_w):
    """
    Extend graph towards max_w
    
    Inputs:     - G         Initial graph
                - nei_inds  Neighbour indices for each atom
                - nei_ats   Neighbour atoms for each atom
                - max_w     Maximum depth of the graph
    
    Outputs:    - G         New (extended) graph
                - change    Whether the graph changed or not
    """
    
    # Initialize change variable
    change = False
    # Get number of nodes in the initial graph
    N = G.number_of_nodes()
    
    # Loop over all nodes in the initial graph
    for i in range(N):
        # Update the list of nodes in the graph
        nodes = get_nodes(G)
        # Get neighbours of node i
        neis = get_graph_neighbours(nodes[i], nei_inds, nei_ats)
        for n in neis:
            # If the neighbouring atom is not in the initial graph, add it
            if n not in nodes:
                G.add_node(N, elem=n[0], ind=n[1])
                G.add_edge(i, N, w="1")
                
                # Check that the new node is within max_w, otherwise remove it
                if nx.shortest_path_length(G, source=0, target=N) > max_w:
                    G.remove_node(N)
                else:
                    # Update the number and list of nodes in the graph
                    N += 1
                    nodes = get_nodes(G)
                    change = True
            # If the neighbouring atom is in the initial graph, just add the edge
            else:
                G.add_edge(i, nodes.index(n), w="1")
    
    return G, change



def generate_graphs(elem, envs, nei_inds, nei_atoms, max_weight):
    """
    Generate all the graphs for one element given a list of neighbours
    
    Inputs: - elem          Element to generate the graphs for
            - envs          Environments around each atom (first coordination shell)
            - nei_inds      Neighbour indices for each atom
            - nei_atoms     Neighbour atoms for each atom
            - max_weight    Maximum depth of the graph
    
    Output: - Gs            List of graphs for each atom of the given element
    """
    Gs = []
    for i, e in enumerate(envs[elem]):
        # Initialize nodes and bonds for graph construction
        nodes = [(elem, i)]
        bonds = []

        # Get the neighbours of the central atom
        neighbours = get_graph_neighbours(nodes[0], nei_inds, nei_atoms)
        for nei in neighbours:
            nodes.append(nei)
            bonds.append((nodes[0], nei))

        # Generate initial graph (w = 1)
        G = generate_graph(nodes, bonds)

        # Extend the graph until the max depth
        change = True
        while change:
            G, change = extend_graph(G, nei_inds, nei_atoms, max_weight)

        # Append to the array of graphs
        Gs.append(G)
    
    return Gs



def generate_hash(G):
    """
    Generate the hash corresponding to a graph
    
    Input:  - G     Input graph

    Output: - H     Hash corresponding to graph G
    """
    
    # Replacing the central element by "Y" in order to make sure that isomorphism correctly identifies the central node
    G2 = G.copy()
    G2.nodes[0]["elem"] = "Y"
    
    return nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(G2, edge_attr="w", node_attr="elem", iterations=5)



def cut_graph(G, w):
    """
    Cut a graph down to a given depth
    
    Inputs: - G     Input graph
            - w     Depth to cut to

    Output: - G2    Cut graph
    """
    
    # Copy the initial graph and get the number of nodes
    cut_G = G.copy()
    N = len(G.nodes)
    
    for i in range(N):
        # If the depth is greater than w, remove the node
        if nx.shortest_path_length(G, source=0, target=i) > w:
            cut_G.remove_node(i)

    return cut_G



def check_isomorphism_hash(G, g, nm, em):
    """
    Check whether two graphs are isomorphic, using hashing of the two graphs
    
    Inputs: - G             First graph object
            - g             Second graph object
            - nm            Criterion for node matching
            - em            Criterion for edge matching

    Output: - True/False
    """
    
    # Generate the hash for each graph
    G_h = generate_hash(G)
    g_h = generate_hash(g)
    
    # Compare the hashes
    if G_h == g_h:
        return True
    return False



def get_corresponding_ind(G, G_ref, ind, nm, em, N, i_already=[]):
    """
    Get the index corresponding to node N[ind] in the reference graph
    
    Inputs: - G             Input graph
            - G_ref         Reference graph
            - ind           Index of G_ref to find a correspondence for
            - nm            Criterion for node matching
            - em            Criterion for edge matching
            - N             Number of first neighbours
            - i_already     Indexes already considered
    
    Output: - i             Index of G corresponding to index "ind" of G_ref
    """
    
    # Replace the node in the reference graph
    mod_G_ref = G_ref.copy()
    mod_G_ref.nodes[ind]["elem"] = "Z"
    
    # Loop over all first neighbours
    for i in range(1, N+1):
        if i not in i_already:
            # Replace the node in the current graph
            mod_G = G.copy()
            mod_G.nodes[i]["elem"] = "Z"
            # Check isomorphism
            if check_isomorphism_hash(mod_G, mod_G_ref, nm, em):
                return i
    return -1



def get_corresponding_inds(G, G_ref, nm, em, N, sel_elems=["H", "C", "N", "O"]):
    """
    Get the indices corresponding to the fist neighbours in the reference graph
    
    Inputs:     - G             Input graph
                - G_ref         Reference graph
                - nm            Criterion for node matching
                - em            Criterion for edge matching
                - N             Number of first neighbours
                - sel_elems     Elements to consider
    
    Outputs:    - ref_inds      Indices of the reference graph
                - corr_inds     Indices of the graph correspdonding to ref_inds
    """
    
    # Get the first neighbours in the reference graph
    ref_inds = []
    for i in range(1, N+1):
        if G_ref.nodes[i]["elem"] in sel_elems:
            ref_inds.append(i)
    
    # For each first neighbour, get the corresponding first neighbour in the graph G
    corr_inds = []
    for i in ref_inds:
        corr_inds.append(get_corresponding_ind(G, G_ref, i, nm, em, N, i_already=corr_inds))
        
    return ref_inds, corr_inds



def print_graph(G, max_weight, show=True, save=None):
    """
    Plot a graph
    
    Inputs: - G             networkx graph object
            - max_weight    Maximum depth of the graph
            - show          Whether the plot should be shown or not
            - save          Path to save the plot to
    """
    
    # Get the label of each node
    labs = nx.get_node_attributes(G, "elem")
    # Get the position of each node on the plot
    pos = nx.kamada_kawai_layout(G)
    
    f = plt.figure(figsize=(6,5))
    ax = f.add_subplot(1,1,1)
    
    # Get the nodes within max_weight
    inds = []
    for i in range(G.number_of_nodes()):
        if nx.shortest_path_length(G, source=0, target=i) < max_weight:
            inds.append(i)
    
    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, nodelist=inds, ax=ax)
    # Draw the central node in red
    nx.draw_networkx_nodes(G, pos, nodelist=[0], node_color="r", ax=ax)
    
    # Draw the edge nodes in green
    edge_nodes = []
    for i in range(G.number_of_nodes()):
        if nx.shortest_path_length(G, source=0, target=i) == max_weight:
            edge_nodes.append(i)
    nx.draw_networkx_nodes(G, pos, nodelist=edge_nodes, node_color="g", ax=ax)
    
    # Get the convalent bonds and H-bonds
    H_e = []
    es = []
    for i, e in enumerate(G.edges(inds)):
        if G.edges[e]["w"] == 0:
            H_e.append(e)
        else:
            es.append(e)
    
    # Draw the edges
    nx.draw_networkx_edges(G, pos, edgelist=es, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=H_e, style="dashed", ax=ax)
    
    # Draw the labels on the nodes
    nx.draw_networkx_labels(G, pos, labs, ax=ax)
    
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
