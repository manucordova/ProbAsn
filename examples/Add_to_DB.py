#!/usr/bin/env python
# coding: utf-8

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



if __name__ == "__main__":
    
    db_dir = "../db/"
    in_dir = "../../Data/Additional_dataset/"
    ext = "xyz"

    f = sys.argv[1]

    elems = ["H", "C", "N", "O"]
    all_elems = ["H", "C", "N", "O", "F", "S", "P", "Cl", "Na", "Ca", "K", "Mg"]

    max_weight = 6
    thresh = 0.01

    batch_size = 1000
    
    
    
    def find_lowest_frequency(x):

        z = len(x)
        z2 = 1
        for i in range(1, len(x)):
            if x[i] == x[i-1]:
                z2 += 1
            else:
                z = min(z, z2)

        return z
    
    
    
    print(f"Generating graphs for file {f}...")

    # Load structures
    structs = ase.io.read(in_dir + f, index=":", format="extxyz")
    n_struct = len(structs)
    start = time.time()

    data = {}

    for e1 in elems:
        data[e1] = []
        for e2 in elems:
            data[f"{e1}-{e2}"] = []

    for istruct, struct in enumerate(structs):

        # Print time monitoring
        if (istruct + 1) % 10 == 0:
            stop = time.time()
            dt = stop - start
            eta = dt / (istruct + 1) * (n_struct - istruct - 1)
            print(f"  Processing structure {istruct+1}/{n_struct}, time elapsed {dt:.2f} s, ETA {eta:.2f} s")

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
                Gs[e], envs[e] = gr.generate_graphs(atoms, bonds, e, max_weight, elems=all_elems)
                inds[e] = [i for i, s in enumerate(sym) if s == e]
                hs = [gr.generate_hash(G) for G in Gs[e]]
                z = min(z, find_lowest_frequency(hs))

        for e in Gs:
            for env, G, i in zip(envs[e][::z], Gs[e][::z], inds[e][::z]):

                hs = []
                for w in range(2, max_weight+1):
                    cut_G = gr.cut_graph(G, w)
                    hs.append(gr.generate_hash(cut_G))

                data[e].append((env, crystal, i, cs[i, 0], cs[i, 1], hs[0], hs[1], hs[2], hs[3], hs[4]))

                # 2D graphs
                n_nei = len(env.split("-"))
                if env == "":
                    n_nei = 0

                for j in range(1, n_nei+1):
                    G2 = G.copy()
                    enei = G2.nodes[j]["elem"]
                    G2.nodes[j]["elem"] = "Z"
                    inei = G2.nodes[j]["ind"]

                    hs = []
                    for w in range(2, max_weight+1):
                        cut_G = gr.cut_graph(G2, w)
                        hs.append(gr.generate_hash(cut_G))

                    if enei in elems:

                        data[f"{e}-{enei}"].append((env, crystal, i, cs[i, 0], cs[i, 1], inei, cs[inei, 0], cs[inei, 1], hs[0], hs[1], hs[2], hs[3], hs[4]))

                        
    
    # Update DB files

    for e1 in elems:

        print(f"  Updating DB file {e1}...")

        if len(data[e1]) > 0:

            con = sl.connect(f"{db_dir}ProbAsn_{e1}.db", timeout=10000)

            n_batch = len(data[e1]) // batch_size + int(len(data[e1]) / batch_size > len(data[e1]) // batch_size)

            for ibatch in range(n_batch):

                pp = f"INSERT INTO {e1} (env, crystal, ind, shift, err, G2, G3, G4, G5, G6)\nVALUES\n"

                for env, crystal, i, cs, err, h0, h1, h2, h3, h4 in data[e1][ibatch*batch_size:(ibatch+1)*batch_size]:

                    pp += f"('{env}', '{crystal}', {i}, {cs}, {err}, '{h0}', '{h1}', '{h2}', '{h3}', '{h4}'),\n"

                pp = pp[:-2] + ";"

                with con:
                    con.execute(pp)

            con.commit()
            con.close()

        for e2 in elems:
            if len(data[f"{e1}-{e2}"]) > 0:

                print(f"  Updating DB file {e1}-{e2}...")


                con = sl.connect(f"{db_dir}ProbAsn_{e1}-{e2}.db", timeout=10000)

                n_batch = len(data[f"{e1}-{e2}"]) // batch_size + int(len(data[f"{e1}-{e2}"]) / batch_size > len(data[f"{e1}-{e2}"]) // batch_size)

                for ibatch in range(n_batch):

                    pp = f"INSERT INTO {e1}_{e2} (env, crystal, ind, shift, err, nei_ind, nei_shift, nei_err, G2, G3, G4, G5, G6)\nVALUES\n"

                    for env, crystal, i, cs, err, inei, csnei, errnei, h0, h1, h2, h3, h4 in data[f"{e1}-{e2}"][ibatch*batch_size:(ibatch+1)*batch_size]:

                        pp += f"('{env}', '{crystal}', {i}, {cs}, {err}, {inei}, {csnei}, {errnei}, '{h0}', '{h1}', '{h2}', '{h3}', '{h4}'),\n"

                    pp = pp[:-2] + ";"

                    with con:
                        con.execute(pp)

                con.commit()
                con.close()