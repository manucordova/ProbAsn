#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import sys
import matplotlib.pyplot as plt

import networkx as nx
import ase
import ase.io
import time
import sqlite3 as sl

import ProbAsn.utils as ut
import ProbAsn.graph as gr


# In[2]:


in_dir = "../../../ProbAsn_ShiftML2/data/ShiftML2_probasn_dataset/"
ext = "xyz"

f = sys.argv[1]

elems = ["H", "C", "N", "O", "S"]

max_weight = 6
thresh = 0.01


# # Create DB file

# In[3]:


if not os.path.exists(f"../db/ProbAsn_{f.split('.')[0]}.db"):

    con = sl.connect(f"../db/ProbAsn_{f.split('.')[0]}.db")


    # In[4]:


    for e1 in elems:
        with con:
            con.execute(f"""
                CREATE TABLE {e1} (
                    env VARCHAR(16),
                    crystal VARCHAR(16),
                    ind UNSIGNED INTEGER,
                    shift FLOAT,
                    err FLOAT,
                    G2 VARCHAR(32),
                    G3 VARCHAR(32),
                    G4 VARCHAR(32),
                    G5 VARCHAR(32),
                    G6 VARCHAR(32)
                );
            """)
        for e2 in elems:
            with con:
                con.execute(f"""
                    CREATE TABLE {e1}_{e2} (
                        env VARCHAR(16),
                        crystal VARCHAR(16),
                        ind UNSIGNED INTEGER,
                        shift FLOAT,
                        err FLOAT,
                        nei_ind UNSIGNED INTEGER,
                        nei_shift FLOAT,
                        nei_err FLOAT,
                        G2 VARCHAR(32),
                        G3 VARCHAR(32),
                        G4 VARCHAR(32),
                        G5 VARCHAR(32),
                        G6 VARCHAR(32)
                    );
                """)


    # In[5]:


    def find_lowest_frequency(x):

        z = len(x)
        z2 = 1
        for i in range(1, len(x)):
            if x[i] == x[i-1]:
                z2 += 1
            else:
                z = min(z, z2)

        return z


    # In[6]:


    print(f"Generating graphs for file {f}...")

    # Load structures
    structs = ase.io.read(in_dir + f, index=":", format="extxyz")
    n_struct = len(structs)
    start = time.time()

    for istruct, struct in enumerate(structs):

        # Print time monitoring
        if (istruct + 1) % 10 == 0:
            stop = time.time()
            dt = stop - start
            eta = dt / (istruct + 1) * (n_struct - istruct - 1)
            print(f"  {istruct+1}/{n_struct}, time elapsed {dt:.2f} s, ETA {eta:.2f} s")

        # Get structure elements and identifier
        sym = struct.get_chemical_symbols()
        crystal = struct.info["ID"]

        # Get shifts
        cs = struct.get_array("cs")

        # Get atoms and bonds
        atoms, bonds = gr.get_bonds_in_cryst(struct)

        # Get zprime
        Gs = {}
        envs = {}
        inds = {}
        z = len(atoms)
        for e in elems:
            if e in sym:
                Gs[e], envs[e] = gr.generate_graphs(atoms, bonds, e, max_weight)
                inds[e] = [i for i, s in enumerate(sym) if s == e]
                hs = [gr.generate_hash(G) for G in Gs[e]]
                z = min(z, find_lowest_frequency(hs))

        for e in Gs:
            for env, G, i in zip(envs[e][::z], Gs[e][::z], inds[e][::z]):

                # 1D graphs
                hs = []
                for w in range(2, max_weight+1):
                    cut_G = gr.cut_graph(G, w)
                    hs.append(gr.generate_hash(cut_G))

                with con:
                    con.execute(f"""
                        INSERT INTO {e} (env, crystal, ind, shift, err,
                            G2, G3, G4, G5, G6)

                        VALUES ('{env}', '{crystal}', {i}, {cs[i, 0]}, {cs[i, 1]},
                            '{hs[0]}', '{hs[1]}', '{hs[2]}', '{hs[3]}', '{hs[4]}')

                    """)

                # 2D graphs
                n_nei = len(env.split("-"))
                if env == "":
                    n_nei = 0

                for j in range(1, n_nei+1):
                    hs = []
                    G2 = G.copy()
                    enei = G2.nodes[j]["elem"]
                    G2.nodes[j]["elem"] = "Z"
                    inei = G2.nodes[j]["ind"]

                    for w in range(2, max_weight+1):
                        cut_G = gr.cut_graph(G2, w)
                        hs.append(gr.generate_hash(cut_G))

                    with con:
                        con.execute(f"""
                            INSERT INTO {e}_{enei} (env, crystal, ind, shift, err,
                                nei_ind, nei_shift, nei_err, G2, G3, G4, G5, G6)

                            VALUES ('{env}', '{crystal}', {i}, {cs[i, 0]}, {cs[i, 1]},
                                {inei}, {cs[inei, 0]}, {cs[inei, 1]},
                                '{hs[0]}', '{hs[1]}', '{hs[2]}', '{hs[3]}', '{hs[4]}')

                        """)

