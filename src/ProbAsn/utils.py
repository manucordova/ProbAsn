####################################################################################################
###                                                                                              ###
###                                     Utility functions                                        ###
###                               Author: Manuel Cordova (EPFL)                                  ###
###                                Last modified: 03.09.2021                                     ###
###                                                                                              ###
####################################################################################################

# Import libraries
import numpy as np
import gdown
import zipfile



def download_database(url=None, output="../db/ProbAsn.db.zip"):
    """
    Download the database
    """

    gdown.download(url, output, quiet=False)
    with zipfile.ZipFile(output,"r") as zip_ref:
        zip_ref.extractall(output.replace(".zip", ""))
    return



def get_default_values(block):
    """
    Get default parameters for a block
    
    Input:  - block     Block name
            - params    Default parameters for the desired block
    """

    if block == "SYS":
        params = {"db_file": "../db/ProbAsn.db",
                  "out_root": "../output/",
                  "max_w": 6,
                  "N_min": 10,
                  "conv_H": [-1., 30.78],
                  "conv_C": [-1., 170.04],
                  "conv_N": [-1., 188.56],
                  "conv_O": [-1., 254.62],
                  "elems": ["H", "C", "N", "O", "S"],
                  "hetatm": "error",
                  "hetatm_rep": None,
                  "exclude": None,
                  "save_individual_distribs": False,
                  "verbose": False,
                 }
    
    elif block == "MOL":
        params = {"in_type": "smi",
                  "from_file": False,
                  "out_type": "mol",
                  "make_3d": False,
                  "save_struct": True,
                 }
    
    elif block == "NMR":
        params = {"nei_elem": None,
                  "n_points_distrib": 1001,
                  "assign": True,
                  "multiplicities": None,
                  "custom_distribs": None,
                  "prevent_cleanup": False,
                  "custom_inds": None,
                 }
    
    elif block == "ASN":
        params = {"p_thresh": 100.,
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



def clean_split(l, delimiter):
    """
    Split a line with the desired delimiter, ignoring delimiters present in arrays or strings
    
    Inputs: - l     Input line
    
    Output: - ls    List of sub-strings making up the line
    """
    
    # Initialize sub-strings
    ls = []
    clean_l = ""
    
    # Loop over all line characters
    in_dq = False
    in_sq = False
    arr_depth = 0
    for li in l:
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
        
        # If the delimiter is not within quotes or in an array, split the line at that character
        if li == delimiter and not in_dq and not in_sq and arr_depth == 0:
            ls.append(clean_l)
            clean_l = ""
        else:
            clean_l += li
    
    ls.append(clean_l)
        
    return ls



def get_dict(l):
    """
    Get the values in an dictionary contained in a line

    Input:  - l         Input line

    Output: - d         dictionary of values
    """

    # Initialize array
    d ={}
    clean_l = ""

    # Loop over all line characters
    arr_depth = 0
    for li in l:

        # Identify end of array
        if li == "}":
            arr_depth -= 1
            
            # Check that there are not too many closing brackets for the opening ones
            if arr_depth < 0:
                raise ValueError("Missing \"{\" for matching the number of \"}\"")
        
        # If we are within the array, extract the character
        if arr_depth > 0:
            clean_l += li

        # Identify start of array
        if li == "{":
            arr_depth += 1

    # Check that the array is properly closed at the end
    if arr_depth > 0:
        raise ValueError("Missing \"}\" for matching the number of \"{\"")

    # Extract elements in the array
    ls = clean_split(clean_l, ",")

    # Get the value of each element in the array
    for li in ls:
        tmp = clean_split(li, ":")
        if len(tmp) != 2:
            raise ValueError("Erroneous dictionary part: {}".format(li))
        key = get_value(tmp[0].strip())
        val = get_value(tmp[1])
        
        d[key] = val

    return d
    
    
    
def get_array(l):
    """
    Get the values in an array contained in a line
    
    Input:  - l         Input line
    
    Output: - vals      Array of values
    """
    
    # Identify empty array
    if l.strip() == "[]":
        return []
    
    # Initialize array
    vals = []
    clean_l = ""
    
    # Loop over all line characters
    arr_depth = 0
    for li in l:
    
        # Identify end of array
        if li == "]":
            arr_depth -= 1
            
            # Check that there are not too many closing brackets for the opening ones
            if arr_depth < 0:
                raise ValueError("Missing \"[\" for matching the number of \"]\"")
        
        # If we are within the array, extract the character
        if arr_depth > 0:
            clean_l += li
    
        # Identify start of array
        if li == "[":
            arr_depth += 1
    
    # Check that the array is properly closed at the end
    if arr_depth > 0:
        raise ValueError("Missing \"]\" for matching the number of \"[\"")
    
    # Extract elements in the array
    ls = clean_split(clean_l, ",")
    
    # Get the value of each element in the array
    for li in ls:
        vals.append(get_value(li))

    return vals
    
    
    
def get_value(l):
    """
    Get the value from an input line
    
    Input:  - l         Input line
    
    Output: - val       Extracted value
    """
    
    # Identify arrays
    if l.strip().startswith("["):
        return get_array(l)
    
    # Identify dicts
    if l.strip().startswith("{"):
        return get_dict(l)
    
    # Identify strings (double quotes)
    if l.strip().startswith("\""):
        l2 = ""
        rec = False
        for li in l:
            
            if li == "\"" and rec:
                break
            
            if rec:
                l2 += li
            
            if li == "\"" and not rec:
                rec = True
        
        return l2
    
    # Identify strings (single quotes)
    if l.strip().startswith("\'"):
        l2 = ""
        rec = False
        for li in l:
            
            if li == "\'" and rec:
                break
            
            l2 += li
            
            if li == "\'" and not rec:
                rec = True
        
        return l2
    
    # Identify bools and none
    if l.strip().lower() == "true":
        return True
    
    if l.strip().lower() == "false":
        return False
    
    if l.strip().lower() == "none":
        return None
    
    # Identify float
    if "." in l:
        return float(l)
    
    # Otherwise, assume this is an int
    return int(l)
    
    

def parse_line(l):
    """
    Parse line from input file
    
    Input:      - l         Input line
    
    Outputs:    - key       Parameter key
                - val       Parameter value
    """
    
    # Remove comment
    clean_l = clean_split(l, "#")[0]
    
    # Check that the line is valid
    ls = clean_split(clean_l, "=")
    if len(ls) != 2:
        raise ValueError("Unexpected input line: {}".format(l))
    
    # Get key
    key = ls[0]
    
    # Get value
    val = get_value(ls[1])
    
    return key.strip(), val



def parse_block(lines, block):
    """
    Parse block from input file
    
    Inputs: - lines     List of lines in the file
            - block     Block name
    
    Output: - params    Dictionary of parameters
    """
    
    # Get the default parameters
    params = get_default_values(block)
    
    # Loop over all lines in the file
    parse = False
    for l in lines:
    
        # If we are within the desired block, load the parameters in the line
        if parse:
            
            # If the line is a value assignment
            if "=" in l.split("#")[0]:
                key, val = parse_line(l)
                params[key] = val
            
            # Identify block termination
            if l == "$END":
                break
        
        # Identify block start
        if l == "$" + block:
            parse = True
    
    # If the block was not found, raise an error
    if not parse:
        raise ValueError("Cannot find block $" + block)
    
    return params



def parse_input(file):
    """
    Parse input file and return parameters
    
    Input:      - file          Input file
    
    Outputs:    - sys_params    System parameters
                - mol_params    Molecule parameters
                - nmr_params    NMR parameters
                - asn_params    Assignment parameters
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
    """
    Perform a custom selection of graphs to assign
    
    Inputs: - lists         Lists to extract the selection from
            - inds          List of indices to select
            
    Output: - out_lists     Output lists
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
    """
    Generate custom distributions. This is an alternative to the db.fetch_db() function.
    
    Inputs:     - distribs      Dictionary of custom distributions
    
    Outputs:    - shifts        List of centers of each distribution
                - errs          List of widths of each distribution
                - ws            Dummy depth variable
                - labels        List of labels (keys of the distribs dictionary)
                - crysts        Dummy list of crystals
                - inds          Dummy list of indices
                - hashes        Dummy list of hashes
    """

    # Initialize output lists
    labels = []
    shifts = []
    errs = []
    ws = []
    crysts = []
    inds = []
    hashes = []
    
    for l in distribs.keys():
    
        # Get label
        labels.append(l)
        
        # 1D distributions
        if len(distribs[l]) == 2:
            shifts.append(np.array([distribs[l][0]]))
            errs.append(np.array([distribs[l][1]]))
        
        # 2D distributions
        else:
            shifts.append(np.array([[distribs[l][0], distribs[l][2]]]))
            errs.append(np.array([[distribs[l][1], distribs[l][3]]]))
        
        # Set dummy values
        ws.append(0)
        crysts.append(["CUSTOM"])
        inds.append(["CUSTOM"])
        hashes.append("CUSTOM")

    return shifts, errs, ws, labels, crysts, inds, hashes
