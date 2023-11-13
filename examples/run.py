#!/usr/bin/env python
# coding: utf-8

###############################################################################
#                                                                             #
#                Probabilistic assignment of organic crystals                 #
#                        Author: Manuel Cordova (EPFL)                        #
#                          Last modified: 13.11.2023                          #
#                             Example run script                              #
#                                                                             #
###############################################################################

# # Import libraries
import os
import sys

from ProbAsn import graph as gr
from ProbAsn import db
from ProbAsn import sim
from ProbAsn import assign as asn
from ProbAsn import draw as dr
from ProbAsn import utils as ut

if __name__ == "__main__":

    # Set input file
    input_file = sys.argv[1]

    sys_params, mol_params, nmr_params, asn_params = ut.parse_input(input_file)

    # Set input parameters
    if not os.path.exists(sys_params["out_root"]):
        os.mkdir(sys_params["out_root"])

    out_dir = sys_params["out_root"] + mol_params["name"] + "/"

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if mol_params["save_struct"]:
        mol_params["save"] = out_dir + "structure." + mol_params["out_type"]
    else:
        mol_params["save"] = None

    conv = sys_params["conv_" + nmr_params["elem"]]
    if nmr_params["nei_elem"] is not None:
        nei_conv = sys_params["conv_" + nmr_params["nei_elem"]]

    # Generate graphs
    print("Generating graphs...")

    # Construct molecule
    mol = gr.make_mol(
        mol_in=mol_params["input"],
        in_type=mol_params["in_type"],
        out_type=mol_params["out_type"],
        from_file=mol_params["from_file"],
        name=mol_params["name"],
        make_3d=mol_params["make_3d"],
        save=mol_params["save"]
    )

    # Get atoms and bonds in the molecule
    atoms, bonds = gr.get_bonds(mol)

    # Generate graphs for the molecule
    Gs, envs = gr.generate_graphs(
        atoms,
        bonds,
        nmr_params["elem"],
        sys_params["max_w"],
        elems=sys_params["elems"],
        hetatm=sys_params["hetatm"],
        hetatm_rep=sys_params["hetatm_rep"]
    )

    print("Done!")

    # Custom distributions
    custom = False
    if nmr_params["custom_distribs"] is not None:
        (
            all_shifts,
            all_errs,
            ws,
            labels,
            all_crysts,
            all_inds,
            hashes
        ) = ut.gen_custom_distribs(nmr_params["custom_distribs"])
        custom = True

    else:
        # Get entries in the database for each graph
        print("Fetching database...")
        (
            all_shifts,
            all_errs,
            ws,
            labels,
            all_crysts,
            all_inds,
            hashes
        ) = db.fetch_entries(
            sys_params["db_dir"],
            nmr_params["elem"],
            atoms, envs, Gs,
            sys_params["max_w"],
            n_min=sys_params["N_min"],
            nei_elem=nmr_params["nei_elem"],
            exclude=sys_params["exclude"],
            verbose=sys_params["verbose"]
        )
        print("Done!\n")

        # Select a custom set of nuclei to assign
        if nmr_params["custom_inds"] is not None:
            print("Custom selection of graphs...")
            [
                all_shifts,
                all_errs,
                ws,
                labels,
                all_crysts,
                all_inds,
                hashes
            ] = ut.custom_selection(
                [
                    all_shifts,
                    all_errs,
                    ws,
                    labels,
                    all_crysts,
                    all_inds,
                    hashes
                ],
                nmr_params["custom_inds"]
            )
            print("Done!\n")

        if not nmr_params["prevent_cleanup"]:

            # Remove duplicate distributions
            print("Cleaning up methyl groups...")

            # Remove methyl groups
            if nmr_params["elem"] == "H":
                (
                    Gs,
                    [
                        labels,
                        envs,
                        all_shifts,
                        all_errs,
                        ws,
                        all_crysts,
                        all_inds,
                        hashes
                    ]
                ) = sim.cleanup_methyl_protons(
                    Gs,
                    atoms,
                    bonds,
                    [
                        labels,
                        envs,
                        all_shifts,
                        all_errs,
                        ws,
                        all_crysts,
                        all_inds,
                        hashes
                    ]
                )
                print(labels)

            elif nmr_params["nei_elem"] == "H":
                (
                    labels,
                    [
                        all_shifts,
                        all_errs,
                        ws,
                        all_crysts,
                        all_inds,
                        hashes
                    ]
                ) = sim.cleanup_methyls(
                    labels,
                    atoms,
                    bonds,
                    [
                        all_shifts,
                        all_errs,
                        ws,
                        all_crysts,
                        all_inds,
                        hashes
                    ]
                )
            print("Done!\n")

        print("Saving graphs...")

        if nmr_params["nei_elem"] is not None:
            gr_dir = f"{out_dir}graphs_{nmr_params['elem']}"
            gr_dir += f"-{nmr_params['nei_elem']}/"
        else:
            gr_dir = f"{out_dir}graphs_{nmr_params['elem']}/"

        if not os.path.exists(gr_dir):
            os.mkdir(gr_dir)

        for k, (l, w) in enumerate(zip(labels, ws)):
            i = int(l.split("-")[0].split(nmr_params["elem"])[1]) - 1
            gr_file = gr_dir + l + ".pdf"
            if nmr_params["elem"] == "H":
                gr.print_graph(Gs[k], w, show=False, save=gr_file)
            else:
                gr.print_graph(Gs[i], w, show=False, save=gr_file)

        print("Done!\n")

        if not nmr_params["prevent_cleanup"]:
            print("Cleaning up equivalent distributions...")

            (
                labels,
                hashes,
                [
                    all_shifts,
                    all_errs,
                    ws,
                    all_crysts,
                    all_inds,
                ]
            ) = sim.cleanup_equivalent(
                labels,
                hashes,
                [
                    all_shifts,
                    all_errs,
                    ws,
                    all_crysts,
                    all_inds
                ]
            )
            print("Done!\n")
            print(labels)

    # Select multiplicity (if set)
    if asn_params["select_mult"] is not None:

        # Check that the length of the array of multiplicities matches
        # the length of the array of shifts
        if len(nmr_params["shifts"]) != len(nmr_params["multiplicities"]):
            msg = "The length of the array of multiplicities"
            msg += f" ({len(nmr_params['multiplicities'])})"
            msg += " does not match the length of the array"
            msg += f" of shifts ({len(nmr_params['shifts'])})"
            raise ValueError(msg)

        # Get multiplicities
        mults = {}
        for label in labels:
            i = int(
                label.split("/")[0]
                .split("-")[0]
                .split(nmr_params["elem"])[1]) - 1
            mults[l] = envs[i].split("-").count("H")

        sel_labels = []
        sel_shifts = []
        sel_errs = []
        sel_ws = []
        sel_crysts = []
        sel_inds = []
        sel_hashes = []

        if type(asn_params["select_mult"]) is list:
            for mult in asn_params["select_mult"]:
                for (
                    l,
                    sh,
                    er,
                    w,
                    cr,
                    inds,
                    h
                ) in zip(
                    labels,
                    all_shifts,
                    all_errs,
                    ws,
                    all_crysts,
                    all_inds,
                    hashes
                ):
                    if mults[l] == mult:
                        sel_labels.append(l)
                        sel_shifts.append(sh)
                        sel_errs.append(er)
                        sel_ws.append(w)
                        sel_crysts.append(cr)
                        sel_inds.append(inds)
                        sel_hashes.append(h)

            exp_shifts = [
                s for i, s in enumerate(nmr_params["shifts"])
                if nmr_params["multiplicities"][i] in asn_params["select_mult"]
            ]

        else:

            # Set desired multiplicity
            mult = asn_params["select_mult"]

            for (
                l,
                sh,
                er,
                w,
                cr,
                inds,
                h
            ) in zip(
                labels,
                all_shifts,
                all_errs,
                ws,
                all_crysts,
                all_inds,
                hashes
            ):
                if mults[l] == mult:
                    sel_labels.append(l)
                    sel_shifts.append(sh)
                    sel_errs.append(er)
                    sel_ws.append(w)
                    sel_crysts.append(cr)
                    sel_inds.append(inds)
                    sel_hashes.append(h)

            exp_shifts = [
                s for i, s in enumerate(nmr_params["shifts"])
                if nmr_params["multiplicities"][i] == mult
            ]

        labels = sel_labels
        all_shifts = sel_shifts
        all_errs = sel_errs
        ws = sel_ws
        all_crysts = sel_crysts
        all_inds = sel_inds
        hashes = sel_hashes

    else:
        exp_shifts = nmr_params["shifts"]

    # Check that there are not too many shifts to assign
    n_labels = len(labels)
    n_labels += sum([label.count("/") for label in labels])
    if len(exp_shifts) > n_labels:
        raise ValueError(f"Too many shifts to assign! (maximum: {n_labels})")

    # Evaluate distributions and individual assignment probabilities

    # 1D simulation
    if nmr_params["nei_elem"] is None:

        # Get plotting range
        lims = sim.get_lims_1D(all_shifts, all_errs, conv)

        # Generate the distributions
        x, ys = sim.make_1D_distributions(
            lims,
            nmr_params["n_points_distrib"],
            all_shifts,
            all_errs,
            conv,
            norm="max"
        )

        # Plot the distributions
        if sys_params["save_individual_distribs"]:

            pp = ""

            if asn_params["select_mult"] is not None:
                dist_dir = f"{out_dir}distribs_{nmr_params['elem']}"
                dist_dir += f"_mult_{mult}/"
            else:
                dist_dir = f"{out_dir}distribs_{nmr_params['elem']}/"
            if not os.path.exists(dist_dir):
                os.mkdir(dist_dir)

            for label, y, shifts, w in zip(labels, ys, all_shifts, ws):
                file = dist_dir + label.replace("/", "_") + ".pdf"
                t = dr.draw_1D_distribution_and_hist(
                    x,
                    y,
                    shifts,
                    conv,
                    w,
                    nmr_params["elem"],
                    f=file,
                    custom=custom
                )

                pp += f"{l:<4}\n{t}\n\n"

            with open(f"{dist_dir}distributions.dat", "w") as F:
                F.write(pp)

        if asn_params["select_mult"] is not None:
            file = f"{out_dir}distribs_{nmr_params['elem']}_mult_{mult}.pdf"
        else:
            file = f"{out_dir}distribs_{nmr_params['elem']}.pdf"
        dr.draw_1D_distributions(
            x,
            ys,
            labels,
            nmr_params["elem"],
            lims=lims,
            f=file
        )

        if nmr_params["assign"]:
            if asn_params["select_mult"] is not None:
                file = f"{out_dir}distribs_{nmr_params['elem']}"
                file += f"_mult_{mult}_with_exp.pdf"
            else:
                file = f"{out_dir}distribs_{nmr_params['elem']}_with_exp.pdf"
            dr.draw_1D_distributions(
                x,
                ys,
                labels,
                nmr_params["elem"],
                lims=lims,
                exps=exp_shifts,
                f=file
            )

        # Get individual assignment probabilities
        scores = sim.compute_scores_1D(exp_shifts, all_shifts, all_errs, conv)

    # 2D simulation
    else:

        # Get plotting range
        lims = sim.get_lims_2D(
            all_shifts,
            all_errs,
            conv,
            nei_conv,
            dqsq=nmr_params["dqsq"]
        )

        # Generate the distributions
        X, Y, Zs = sim.make_2D_distributions(
            lims,
            nmr_params["n_points_distrib"],
            all_shifts,
            all_errs,
            conv,
            nei_conv,
            dqsq=nmr_params["dqsq"],
            norm="max"
        )

        # Plot the distributions
        if sys_params["save_individual_distribs"]:

            if asn_params["select_mult"] is not None:
                dist_dir = f"{out_dir}distribs_{nmr_params['elem']}"
                dist_dir += f"-{nmr_params['nei_elem']}_mult_{mult}/"
            else:
                dist_dir = f"{out_dir}distribs_{nmr_params['elem']}"
                dist_dir += f"-{nmr_params['nei_elem']}/"
            if not os.path.exists(dist_dir):
                os.mkdir(dist_dir)

            for label, Z, shifts, w in zip(labels, Zs, all_shifts, ws):
                file = dist_dir + label.replace("/", "_") + ".pdf"
                dr.draw_2D_distribution_and_hist(
                    X,
                    Y,
                    Z,
                    shifts,
                    conv,
                    nei_conv,
                    w,
                    nmr_params["elem"],
                    nmr_params["nei_elem"],
                    dqsq=nmr_params["dqsq"],
                    f=file,
                    custom=custom
                )

        if asn_params["select_mult"] is not None:
            file = f"{out_dir}distribs_{nmr_params['elem']}"
            file += f"-{nmr_params['nei_elem']}_mult_{mult}.pdf"
        else:
            file = f"{out_dir}distribs_{nmr_params['elem']}"
            file += f"-{nmr_params['nei_elem']}.pdf"
        dr.draw_2D_distributions(
            X,
            Y,
            Zs,
            labels,
            nmr_params["elem"],
            nmr_params["nei_elem"],
            dqsq=nmr_params["dqsq"],
            lims=lims,
            f=file
        )

        if nmr_params["assign"]:
            if asn_params["select_mult"] is not None:
                file = f"{out_dir}distribs_{nmr_params['elem']}"
                file += f"-{nmr_params['nei_elem']}_mult_{mult}_with_exp.pdf"
            else:
                file = f"{out_dir}distribs_{nmr_params['elem']}"
                file += f"-{nmr_params['nei_elem']}_with_exp.pdf"
            dr.draw_2D_distributions(
                X,
                Y,
                Zs,
                labels,
                nmr_params["elem"],
                nmr_params["nei_elem"],
                dqsq=nmr_params["dqsq"],
                lims=lims,
                exps=exp_shifts,
                f=file
            )

        # Get individual assignment probabilities
        scores = sim.compute_scores_2D(
            exp_shifts,
            all_shifts,
            all_errs,
            conv,
            nei_conv,
            dqsq=nmr_params["dqsq"]
        )

    # Generate global assignments and obtain
    # marginal individual assignment probabilities
    if nmr_params["assign"]:

        if nmr_params["nei_elem"] is None:
            if asn_params["select_mult"] is not None:
                p_dir = f"{out_dir}probs_{nmr_params['elem']}_mult_{mult}/"
            else:
                p_dir = f"{out_dir}probs_{nmr_params['elem']}/"
            exp_str = [f"{e:.2f}" for e in exp_shifts]
        else:
            if asn_params["select_mult"] is not None:
                p_dir = f"{out_dir}probs_{nmr_params['elem']}"
                p_dir += f"-{nmr_params['nei_elem']}_mult_{mult}/"
            else:
                p_dir = f"{out_dir}probs_{nmr_params['elem']}"
                p_dir += f"-{nmr_params['nei_elem']}/"
            exp_str = [f"{e[0]:.2f}\\{e[1]:.2f}" for e in exp_shifts]
        if not os.path.exists(p_dir):
            os.mkdir(p_dir)

        # Write prior probabilities
        file = p_dir + "prior_probs.dat"
        asn.write_individual_probs(labels, exp_str, scores, file, decimals=2)
        dr.print_probabilities(file, display=False)

        if nmr_params["dqsq"]:
            raise NotImplementedError()

        # Get possible individual assignments
        possible_assignments, p_thresh = asn.get_possible_assignments(
            scores,
            labels,
            exp_str,
            thresh=asn_params["p_thresh"],
            thresh_increase=asn_params["thresh_increase"]
        )

        # Generate all possible assignments and get their probabilities
        (
            dist_pools,
            shift_pools,
            pool_asns,
            pool_scores,
            all_labels,
            equivs
        ) = asn.get_probabilistic_assignment(
            scores,
            possible_assignments,
            exp_shifts,
            labels,
            max_asn=asn_params["max_asn"],
            r_max_asn=asn_params["r_max_asn"],
            order=asn_params["asn_order"],
            max_excess=asn_params["max_excess"],
            disp_rank=asn_params["disp_r"],
            pool_inds=asn_params["pool_inds"],
            verbose=sys_params["verbose"]
        )

        # Write global assignments generated and their probabilities
        file = p_dir + "assignment_probs.dat"
        asn.write_global_probs(
            dist_pools,
            shift_pools,
            pool_asns,
            pool_scores,
            all_labels,
            exp_str,
            equivs,
            file,
            decimals=8
        )

        # Get marginal probabilities
        marginal = asn.update_split_scores(
            dist_pools,
            shift_pools,
            pool_asns,
            pool_scores,
            equivs,
            labels,
            all_labels
        )

        # Write marginal probabilities
        file = p_dir + "marginal_probs.dat"
        asn.write_split_individual_probs(labels, exp_str, marginal, file)
        dr.print_probabilities(file, display=False)

    print("All done!")
