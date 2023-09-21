###############################################################################
#                                                                             #
#                              Utility functions                              #
#                        Author: Manuel Cordova (EPFL)                        #
#                          Last modified: 15.09.2023                          #
#                                                                             #
###############################################################################

# Import libraries
import numpy as np
import zipfile
import requests
import io


default_db_url = "https://archive.materialscloud.org/"
default_db_url += "record/file?filename=db.zip&record_id=1824"


def download_database(
    url=default_db_url,
    output="../db/"
):
    """Download the database.

    Parameters
    ----------
    url : str
        Path to the ZIP file containing the database.
    output : str
        Path to the directory where the database should be stored.
    """

    print(f"Downloading database from {url}...")

    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(output)

    print("Database downloaded!")

    return


def get_default_values(block):
    """Get default parameters for a block.

    Parameters
    ----------
    block : str
        Name of the block.

    Returns
    -------
    params : dict
        Dictionary of default parameters for the desired block.
    """

    # System block
    if block == "SYS":
        params = {
            "db_dir": "../db/",
            "out_root": "../output/",
            "max_w": 6,
            "N_min": 10,
            "conv_H": [-1., 30.78],
            "conv_C": [-1., 170.04],
            "conv_N": [-1., 188.56],
            "conv_O": [-1., 254.62],
            "elems": [
                "H",
                "C",
                "N",
                "O",
                "S",
                "F",
                "P",
                "Cl",
                "Na",
                "Ca",
                "Mg",
                "K"
            ],
            "hetatm": "error",
            "hetatm_rep": None,
            "exclude": None,
            "save_individual_distribs": False,
            "verbose": False,
        }

    # Molecule block
    elif block == "MOL":
        params = {
            "in_type": "smi",
            "from_file": False,
            "out_type": "mol",
            "make_3d": False,
            "save_struct": True,
        }

    # NMR block
    elif block == "NMR":
        params = {
            "nei_elem": None,
            "dqsq": False,
            "n_points_distrib": 1001,
            "assign": True,
            "multiplicities": None,
            "custom_distribs": None,
            "prevent_cleanup": False,
            "custom_inds": None,
        }

    # Assignment block
    elif block == "ASN":
        params = {
            "p_thresh": 100.,
            "thresh_increase": "+1",
            "select_mult": None,
            "max_asn": None,
            "r_max_asn": 0,
            "asn_order": "default",
            "disp_r": 2,
            "max_excess": 4,
            "pool_inds": None,
        }

    else:
        raise ValueError("Unknown block: {}".format(block))

    return params


def clean_split(line, delimiter):
    """Split a line with the desired delimiter.

    Parameters
    ----------
    line : str
        Input line.
    delimiter : str
        Delimiter character(s).

    Returns
    -------
    ls : list
        List of delimiter-separated lines.
    """

    # Initialize sub-strings
    ls = []
    clean_line = ""

    # Loop over all line characters
    in_dq = False
    in_sq = False
    arr_depth = 0
    for li in line:
        # Identify strings with double quotes
        if li == "\"":
            if not in_dq:
                in_dq = True
            else:
                in_dq = False

        # Identify strings with single quotes
        if li == "\'":
            if not in_sq:
                in_sq = True
            else:
                in_sq = False

        # Identify arrays
        if li == "[":
            if not in_sq and not in_dq:
                arr_depth += 1
        if li == "]":
            if not in_sq and not in_dq:
                arr_depth -= 1

        # If the delimiter is not within quotes or in an array,
        # split the line at that character
        if li == delimiter and not in_dq and not in_sq and arr_depth == 0:
            ls.append(clean_line)
            clean_line = ""
        else:
            clean_line += li

    ls.append(clean_line)

    return ls


def get_dict(line):
    """Get the values in an dictionary contained in a line.

    Parameters
    ----------
    line : str
        Input line.

    Returns
    -------
    d : dict
        Dictionary of values in the line.
    """

    # Initialize array
    d = {}
    clean_line = ""

    # Loop over all line characters
    arr_depth = 0
    for li in line:

        # Identify end of array
        if li == "}":
            arr_depth -= 1

            # Check that there are not too many closing brackets
            # for the opening ones
            if arr_depth < 0:
                raise ValueError(
                    "Missing \"{\" for matching the number of \"}\""
                )

        # If we are within the array, extract the character
        if arr_depth > 0:
            clean_line += li

        # Identify start of array
        if li == "{":
            arr_depth += 1

    # Check that the array is properly closed at the end
    if arr_depth > 0:
        raise ValueError(
            "Missing \"}\" for matching the number of \"{\""
        )

    # Extract elements in the array
    ls = clean_split(clean_line, ",")

    # Get the value of each element in the array
    for li in ls:
        tmp = clean_split(li, ":")
        if len(tmp) != 2:
            raise ValueError("Erroneous dictionary part: {}".format(li))
        key = get_value(tmp[0].strip())
        val = get_value(tmp[1])

        d[key] = val

    return d


def get_array(line):
    """Get the values in an array contained in a line.

    Parameters
    ----------
    line : str
        Input line.

    Returns
    -------
    vals : list
        List of values in the line.
    """

    # Identify empty array
    if line.strip() == "[]":
        return []

    # Initialize array
    vals = []
    clean_line = ""

    # Loop over all line characters
    arr_depth = 0
    for li in line:

        # Identify end of array
        if li == "]":
            arr_depth -= 1

            # Check that there are not too many closing brackets
            # for the opening ones
            if arr_depth < 0:
                raise ValueError(
                    "Missing \"[\" for matching the number of \"]\""
                )

        # If we are within the array, extract the character
        if arr_depth > 0:
            clean_line += li

        # Identify start of array
        if li == "[":
            arr_depth += 1

    # Check that the array is properly closed at the end
    if arr_depth > 0:
        raise ValueError(
            "Missing \"]\" for matching the number of \"[\""
        )

    # Extract elements in the array
    ls = clean_split(clean_line, ",")

    # Get the value of each element in the array
    for li in ls:
        vals.append(get_value(li))

    return vals


def get_value(line):
    """Get the value from an input line.

    Parameters
    ----------
    line : str
        Input line.

    Returns
    -------
    val : any
        Extracted value
    """

    # Identify arrays
    if line.strip().startswith("["):
        return get_array(line)

    # Identify dicts
    if line.strip().startswith("{"):
        return get_dict(line)

    # Identify strings (double quotes)
    if line.strip().startswith("\""):
        l2 = ""
        rec = False
        for li in line:

            if li == "\"" and rec:
                break

            if rec:
                l2 += li

            if li == "\"" and not rec:
                rec = True

        return l2

    # Identify strings (single quotes)
    if line.strip().startswith("\'"):
        l2 = ""
        rec = False
        for li in line:

            if li == "\'" and rec:
                break

            l2 += li

            if li == "\'" and not rec:
                rec = True

        return l2

    # Identify bools and none
    if line.strip().lower() == "true":
        return True

    if line.strip().lower() == "false":
        return False

    if line.strip().lower() == "none":
        return None

    # Identify float
    if "." in line:
        return float(line)

    # Otherwise, assume this is an int
    return int(line)


def parse_line(line):
    """Parse a line from the input file.

    Parameters
    ----------
    line : str
        Input line.

    Returns
    -------
    key : str
        Parameter key.
    val : any
        Parameter value.
    """

    # Remove comment
    clean_l = clean_split(line, "#")[0]

    # Check that the line is valid
    ls = clean_split(clean_l, "=")
    if len(ls) != 2:
        raise ValueError(f"Unexpected input line: {line}")

    # Get key
    key = ls[0]

    # Get value
    val = get_value(ls[1])

    return key.strip(), val


def parse_block(lines, block):
    """Parse block from input file.

    Parameters
    ----------
    lines : list
        List of lines in the file.
    block : str
        Name of the block to parse.

    Returns
    -------
    params : dict
        Dictionary of parameters for the desired block.
    """

    # Get the default parameters
    params = get_default_values(block)

    # Loop over all lines in the file
    parse = False
    for line in lines:

        # If we are within the desired block, load the parameters in the line
        if parse:

            # If the line is a value assignment
            if "=" in line.split("#")[0]:
                key, val = parse_line(line)
                params[key] = val

            # Identify block termination
            if line == "$END":
                break

        # Identify block start
        if line == f"${block}":
            parse = True

    # If the block was not found, raise an error
    if not parse:
        raise ValueError(f"Cannot find block ${block}")

    return params


def parse_input(file):
    """Parse input file and return parameters.

    Parameters
    ----------
    file : str
        Input file.

    Returns
    -------
    sys_params : dict
        Dictionary of system parameters.
    mol_params : dict
        Dictionary of molecule parameters.
    nmr_params : dict
        Dictionary of NMR parameters.
    asn_params : dict
        Dictionary of assignment parameters.
    """

    # Load input file
    with open(file, "r") as F:
        lines = F.read().split("\n")

    # Load blocks
    sys_params = parse_block(lines, "SYS")
    mol_params = parse_block(lines, "MOL")
    nmr_params = parse_block(lines, "NMR")
    asn_params = parse_block(lines, "ASN")

    return sys_params, mol_params, nmr_params, asn_params


def custom_selection(lists, inds):
    """Perform a custom selection of graphs to assign.

    Parameters
    ----------
    lists : array_like
        Lists to extract the selection from.
    inds : array_like
        List of indices to select.

    Returns
    -------
    out_lists : list
        Output lists.
    """

    out_lists = []

    # Loop over each list
    for lst in lists:
        this_lst = []
        # Select the desired elements
        for i, li in enumerate(lst):
            if i in inds:
                this_lst.append(li)

        out_lists.append(this_lst)

    return out_lists


def gen_custom_distribs(distribs):
    """Generate custom distributions.

    Parameters
    ----------
    distribs : dict
        Dictionary of custom distributions.

    Returns
    -------
    shifts : list
        List of centers for each distribution.
    errs : list
        List of widths for each distribution.
    ws : list
        Dummy list of depth variables.
    labels : list
        List of labels (keys of the `distribs` dictionary).
    crysts : list
        Dummy list of crystals.
    inds : list
        Dummy list of indices.
    hashes : list
        Dummy list of hashes.
    """

    # Initialize output lists
    labels = []
    shifts = []
    errs = []
    ws = []
    crysts = []
    inds = []
    hashes = []

    for label in distribs:

        # Get label
        labels.append(label)

        # 1D distributions
        if len(distribs[label]) == 2:
            shifts.append(np.array([distribs[label][0]]))
            errs.append(np.array([distribs[label][1]]))

        # 2D distributions
        else:
            shifts.append(np.array([[distribs[label][0], distribs[label][2]]]))
            errs.append(np.array([[distribs[label][1], distribs[label][3]]]))

        # Set dummy values
        ws.append(0)
        crysts.append(["CUSTOM"])
        inds.append(["CUSTOM"])
        hashes.append("CUSTOM")

    return shifts, errs, ws, labels, crysts, inds, hashes
