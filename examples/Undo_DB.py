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
    #f = "AABHTZ-AFOSOF.xyz"

    elems = ["H", "C", "N", "O"]
    all_elems = ["H", "C", "N", "O", "F", "S", "P", "Cl", "Na", "Ca", "K", "Mg"]

    max_weight = 6
    thresh = 0.01

    def find_lowest_frequency(x):

        z = len(x)
        z2 = 1
        for i in range(1, len(x)):
            if x[i] == x[i-1]:
                z2 += 1
            else:
                z = min(z, z2)

        return z

    print(f"Cleaning up entries from file {f}...")

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
        
        crystal = struct.info["ID"]
        
        for e1 in elems:
            
            con = sl.connect(f"{db_dir}ProbAsn_{e1}.db", timeout=10000)
            
            with con:
                con.execute(f"""
                    DELETE FROM {e1}
                    WHERE crystal="{crystal}"
                """)
            
            con.commit()
            con.close()
            
            for e2 in elems:
            
                con = sl.connect(f"{db_dir}ProbAsn_{e1}-{e2}.db", timeout=10000)

                with con:
                    con.execute(f"""
                        DELETE FROM {e1}_{e2}
                        WHERE crystal="{crystal}"
                    """)
                    
                con.commit()
                con.close()
