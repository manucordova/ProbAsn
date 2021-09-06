# ProbAsn - Input file description

The code is freely available from https://github.com/manucordova/ProbAsn

## General remarks

- Input files are separated into blocks describing system, molecule, NMR and assignment parameters
- Comments can be added using a "#" sign
- Strings should be indicated by single ('abc') or double quotes ("abc")
- Floating point numbers should be indicated with a dot, even if decimals are zero. Otherwise, the number will be considered as an integer (1 is an integer, 1. is a floating point number)
- Arrays should be indicated with square brackets, and with a comma delimiter between elements of the array ([1, 2, 3])
- Dictionaries should be indicated with curvy brackets, and with a comma delimiter between elements of the dictionary. The key and value should be separated with two dots ({"H": 1, "C": 6})

## Input file parameters

### $SYS

- db_root: Root directory where the database is located
  - Default: "../db/"
- out_root: Root directory where the output will be stored
  - Default: "../output/"
- max_w: Maximum depth of graphs constructed
  - Default: 6
- N_min: Minimum number of matches in the database to construct a statistical distribution of chemical shifts
  - Default: 10
- conv_H: Parameters [slope, offset] for shielding to shift conversion for 1H
  - Default: [-1., 30.96]
- conv_C: Parameters [slope, offset] for shielding to shift conversion for 13C
  - Default: [-1., 168.64]
- conv_N: Parameters [slope, offset] for shielding to shift conversion for 15N
  - Default: [-1., 185.99]
- conv_O: Parameters [slope, offset] for shielding to shift conversion for 17O. WARNING - the default values are only indicative, and were obtain with only few points!
  - Default: [-1., 205.08]
- elems: List of elements allowed to construct graphs
  - Default: ["H", "C", "N", "O", "S"]
- hetatm: Behaviour for handling atoms not in the list above when constructing graphs
  - Default: "error"
  - Allowed values:
    - "error": raise an error
    - "ignore": ignore the atom
    - "replace": replace the atom with another element
    - "replace_and_terminate": replace the atom with another element and cut all bonds from that atom
- hetatm_rep: Replacements for unknown elements (used only with hetatm set to "replace" or "replace_and_terminate")
  - Default: None
  - Type: dictionary of replacements for each element present in the structure but not in the list above
- exclude: List of CSD Refcodes of the structures to ignore during the search
  - Default: None
- save_individual_distribs: Whether or not to save the individual distributions
  - Default: False
- verbose: Display additional information during the database search
  - Default: False

### $MOL

- name: Name of the molecule
- input: Input string or path to input file for molecular structure. This can be the SMILES string of the molecule, an MDL Molfile, a XYZ file, ... . All 3D informations will be discarded, only the connectivities will be retained
- in_type: Format of the input string/file
  - Default: "smi" (SMILES format)
- from_file: Whether or not the "input" parameter indicates a file
  - Default: False
- out_type: File type for saving the output structure used to identify the atomic labels
  - Default: "mol" (MDL Molfile)
- make_3d: Whether or not the output structure should be 3D
  - Default: False
- save_struct: Whether or not the output structure should be saved
  - Default: True

### $NMR

- elem: Element for which the statistical distributions of chemical shifts should be constructed
- nei_elem: Neighbouring element, used to construct simulated 2D spectra
  - Default: None
- n_points_distrib: Number of points (in each dimension) to draw the statistical distributions of chemical shifts on
  - Default: 1001
- assign: Whether or not to assign experimental shifts. Setting this to False only outputs the statistical distributions of shifts for the atoms in the molecule
  - Default: True
- shifts: List of experimental shifts to assign. The number of shifts should be lower or equal to the number of nuclei of the desired element in the molecule. For simulated 2D spectra, each shift should be entered as an array of the shifts in both dimensions
  - Default: []
- multiplicities: For carbon shifts, multiplicity (number of attached protons) of each experimental shift
  - Default: []
- custom_distribs: If set, generates custom distributions to assign. The distributions should be set as a dictionary where keys are labels of the distributions and values are arrays containing the center and width of each distribution. For 2D distributions, the values are arrays of four numbers, with the two first numbers being the center and width of the distribution for the central element and the two last numbrers being the center and width of the distribution for the neighbouring element. All custom distributions are considered as single Gaussian functions (one-dimensional or two-dimensional).
  - Default: None
  - Example (1D): {"a": [30.0, 1.0], "b": [40.5, 0.5]}
  - Example (2D): {"a": [30.0, 1.0, 100.0, 5.0], "b": [40.5, 0.5, 120.0, 10.0]}
- prevent_cleanup: Prevent cleaning up distributions. This include cleaning up methyl groups and gathering topologically equivalent nuclei.
  - Default: False
- custom_inds: If set, only the subset of graphs indicated (as an array) will be considered. This selection is performed before cleaning up distributions.
  - Default: None

### $ASN

- p_thresh: Relative probability threshold to discard an individual assignment
  - Default: 100.
- thresh_increase: Behaviour for increasing the threshold if no valid assignment can be generated with the currently set value of "p_thresh". This should be a string "+N" to increase the threshold by N, or "xN" to multiply the threshold by N
  - Default: "+1"
- select_mult: For carbon shifts, multiplicity to isolate for the assignment
  - Default: None (Assign all shifts regardless of multiplicity)
- max_asn: Maximum number of individual assignments to consider. A low value to reduce the number of global assignments to generate, but will make the assignment probabilities less reliable. See "r_max_asn" below also
  - Default: None (consider all possible individual assignments)
- r_max_asn: Number of individual assignments after which the "max_asn" value is considered. For example, if r_max_asn = 3 and max_asn = 2, then the three first nuclei will be assigned to any of their possible corresponding shifts, then only the two assignments yielding the highest intermediate probabilities will be considered for the following nuclei
  - Default: 0
- asn_order: Order in which the assignment is performed
  - Default: "default"
  - Allowed values:
    - "default": default order from the input
    - "increase": In order of increasing number of possible individual assignments
    - "decrease": In order of decreasing number of possible individual assignments
- disp_r: Number of individual assignments to display. This is used only to monitor the time remaining to generate all global assignments
  - Default: 2
- max_excess: Maximum excess allowed, defined as the number of nuclei assigned to a single shift
  - Default: 4
- pool_inds: Indices of the pools to assign. This is useful to use different assignment parameters on different pools
  - Default: None (assign all pools)

