#Functions used in the S3_System_characteristics


import math
import scipy.io as sio
import os
import sys
import pandas as pd
import numpy as np

#Function to count the number of buses in the network at each voltage level
def count_buses(net):
    bus_counts = net.bus['vn_kv'].value_counts().sort_index()
    buses33 = bus_counts.get(33, 0)
    buses11 = bus_counts.get(11, 0)
    busesLV = bus_counts.get(0.4, 0)

    return buses33, buses11, busesLV

#Function to count the number of CBs in the network at each voltage level
def count_cbs_at_voltage(net, voltage_kv):
    # Get buses at the specified voltage level
    buses_at_voltage = net.bus[net.bus.vn_kv == voltage_kv].index
    if buses_at_voltage.empty:
        return 0
    # Get switches connected to these buses
    switches = net.switch[
        (net.switch.bus.isin(buses_at_voltage) | net.switch.element.isin(buses_at_voltage)) &
        (net.switch.et == 'b')  # 'b' for bus-bus switches
    ]
    if switches.empty:
        return 0
    # Count the number of circuit breakers
    cb_count = switches[switches.type == 'CB'].shape[0]
    return cb_count

#Function to count the number of fuses in the network at each voltage level
def count_fuses_at_voltage(net, voltage_kv):
    # Get buses at the specified voltage level
    buses_at_voltage = net.bus[net.bus.vn_kv == voltage_kv].index
    if buses_at_voltage.empty:
        return 0
    # Get switches connected to these buses
    switches = net.switch[
        (net.switch.bus.isin(buses_at_voltage) | net.switch.element.isin(buses_at_voltage)) &
        (net.switch.et == 'b')  # 'b' for bus-bus switches
    ]
    if switches.empty:
        return 0
    # Count the number of fuses
    fuse_count = switches[switches.type == 'LS'].shape[0]
    return fuse_count

#Function to count the number of lines in the network disagregared by stdtype and length
def count_lines_by_stdtype_length(net, line_combinations):
    line_count = net.line[['std_type', 'length_km']].value_counts().sort_index()
    lines = {}
    # Count for each combination
    lines = {
        f"{std_type}_{int(length * 1000)}m": line_count[(std_type, length)]
        for std_type, length in line_combinations
        if (std_type, length) in line_count.index
    }
    return lines

#Function to count the number of trafos in the network
def count_trafos(net):
    trafo_count = net.trafo[['vn_hv_kv', 'vn_lv_kv']].value_counts().sort_index()
    trafos3311 = trafo_count.get((33, 11), 0)
    trafos1104 = trafo_count.get((11, 0.4), 0)
    return trafos3311, trafos1104

#Function to count the number of LPs in the network
def count_LPs(net):
    LPs = len(net.load)
    load_mw_values = net.load.p_mw.tolist()
    load_mvar_values = net.load.q_mvar.tolist()
    return LPs, load_mw_values, load_mvar_values

#Function to count the number of MV buses in the MV part of the network
def count_MVbuses_MVnetwork(MVbuses):
    count = 0
    for value in MVbuses.values():
        if isinstance(value, dict):
            count += count_MVbuses_MVnetwork(value)
        elif isinstance(value, list):
            count += len(value)
    return count

#Function to count the number of MV CBs in the MV part of the network
def count_MVCBs_MVnetwork(MVCB_data, exclude_key='LV'):
    total = 0
    if isinstance(MVCB_data, dict):
        for key, value in MVCB_data.items():
            if key != exclude_key:
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if sub_key != 'connections' and isinstance(sub_value, (int, float)):
                            total += sub_value
    return total

#Function to count the number of MV fuses in the MV part of the network
def count_MVfuses_MVnetwork(MVfuses, exclude_key='LV'):
    total = 0
    if isinstance(MVfuses, dict):
        for key, value in MVfuses.items():
            if key != exclude_key:
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if sub_key != 'connections' and isinstance(sub_value, (int, float)):
                            total += sub_value
    return total

#Function to count the number of MV lines
def count_MVlines(MVline_data):
    total = 0
    for stdtype in MVline_data.values():
        for length in stdtype.values():
            for feeder in length.values():
                if isinstance(feeder, dict) and 'connections' in feeder:
                    total += len(feeder['connections'])
    return total

#Function to identify the indices of the MV components in the PC state matrix
def MV_Indices(MVbuses_MVnetwork, PCs_num, MVCBs_MVnetwork, MVfuses_MVnetwork, MVlines, lines):
    sum1 = MVbuses_MVnetwork #Count the MV buses in the MV part of the network
    sum2 = PCs_num['buses33']+PCs_num['buses11']+PCs_num['busesLV'] #Count all the buses to include those in the LV part of the network
    sum3 = sum2 + MVCBs_MVnetwork #Count the MV CBs in the MV part of the network
    sum4 =  sum2 + PCs_num['CBs33'] + PCs_num['CBs11'] + PCs_num['CBsLV'] #Count all the CBs to include those in the LV part of the network
    sum5 = sum4 + MVfuses_MVnetwork #Count the MV fuses in the MV part of the network
    sum6 = sum4 + PCs_num['fusesMV'] + PCs_num['fusesLV'] #Count all the fuses to include those in the LV part of the network
    sum7 = sum6 + MVlines #Count the MV lines
    sum8 = sum6 + sum(lines.values()) #Count all the lines to include the LV lines and LV trafosS
    sum9 = sum8 + PCs_num['trafos3311'] #Count the primary transformers

    MVindices = [
        range(0, sum1), #MV buses in the MV part of the network
        range(sum2, sum3), # MV CBs in the MV part of the network
        range(sum4, sum5), # MV fuses in the MV part of the network
        range(sum6, sum7), # MV lines the network
        range(sum8, sum9) # Primary trafos the MV network
    ]
    return MVindices

def MV_Indices_main_trunk(MVbuses_main_trunk_feeder, PCs_num, MVCBs_main_trunk_feeder, MVfuses_main_trunk_feeder, MVlines_main_trunk_feeder, lines):
    sum1 = MVbuses_main_trunk_feeder #Count the MV buses in the MV part of the network
    sum2 = PCs_num['buses33']+PCs_num['buses11']+PCs_num['busesLV'] #Count all the buses to include those in the LV part of the network
    sum3 = sum2 + MVCBs_main_trunk_feeder #Count the MV CBs in the MV part of the network
    sum4 =  sum2 + PCs_num['CBs33'] + PCs_num['CBs11'] + PCs_num['CBsLV'] #Count all the CBs to include those in the LV part of the network
    sum5 = sum4 + MVfuses_main_trunk_feeder #Count the MV fuses in the MV part of the network
    sum6 = sum4 + PCs_num['fusesMV'] + PCs_num['fusesLV'] #Count all the fuses to include those in the LV part of the network
    sum7 = sum6 + MVlines_main_trunk_feeder #Count the MV lines
    sum8 = sum6 + sum(lines.values()) #Count all the lines to include the LV lines and LV trafosS
    sum9 = sum8 + PCs_num['trafos3311'] #Count the primary transformers

    MVindices_main_trunk_feeder = [
        range(0, sum1), #MV buses in the MV part of the network
        range(sum2, sum3), # MV CBs in the MV part of the network
        range(sum4, sum5), # MV fuses in the MV part of the network
        range(sum6, sum7), # MV lines the network
        range(sum8, sum9) # Primary trafos the MV network
    ]
    return MVindices_main_trunk_feeder

#Function to calculate the effective failure rate for the lines
def calculate_effective_l_lines(net, failure_rates):
    effective_l_lines = {}
    for index, line in net.line.iterrows():
        std_type = line['std_type']
        length = line['length_km']
        if std_type in failure_rates:
            key = f"{std_type}_{int(length * 1000)}m"
            if key not in effective_l_lines:
                effective_l_lines[key] = 0
            effective_l_lines[key] = failure_rates[std_type] * length
    return effective_l_lines

#Function to initialize and populate the matrices with the PC failure rates
def create_and_combine_l_matrices(PCs_num, failure_rates, effective_l_lines):
    l_matrices = {}
    combined_l_matrix = []
    for pc_type, count in PCs_num.items():
        if pc_type == 'lines':
            for line_type, line_count in count.items():
                if line_type in effective_l_lines:
                    l_matrix = np.ones(line_count, dtype=float) * effective_l_lines[line_type]
                    l_matrices[f'l_{line_type}'] = l_matrix.tolist()
                    combined_l_matrix.extend(l_matrix)
        elif pc_type in failure_rates:
            l_matrix = np.ones(count, dtype=float) * failure_rates[pc_type]
            l_matrices[f'l_{pc_type}'] = l_matrix.tolist()
            combined_l_matrix.extend(l_matrix)
            lambda_per_year = np.array(combined_l_matrix)
    return l_matrices, lambda_per_year

#Function to Calculate the effective repair times for the lines
def calculate_effective_MTTR_lines(net, repair_times):
    effective_MTTR_lines = {}
    for index, line in net.line.iterrows():
        std_type = line['std_type']
        length = line['length_km']
        if std_type in repair_times:
            key = f"{std_type}_{int(length * 1000)}m"
            if key not in effective_MTTR_lines:
                effective_MTTR_lines[key] = 0
            effective_MTTR_lines[key] = repair_times[std_type]
    return effective_MTTR_lines

#Function to initialize and populate the matrices with the PC mean repair times
def create_and_combine_MTTR_matrices(PCs_num, repair_times, effective_MTTR_lines):
    MTTR_matrices = {}
    combined_MTTR_matrix = []
    for pc_type, count in PCs_num.items():
        if pc_type == 'lines':
            for line_type, line_count in count.items():
                if line_type in effective_MTTR_lines:
                    MTTR_matrix = np.ones(line_count, dtype=float) * effective_MTTR_lines[line_type]
                    MTTR_matrices[f'l_{line_type}'] = MTTR_matrix.tolist()
                    combined_MTTR_matrix.extend(MTTR_matrix)
        elif pc_type in repair_times:
            MTTR_matrix = np.ones(count, dtype=float) * repair_times[pc_type]
            MTTR_matrices[f'l_{pc_type}'] = MTTR_matrix.tolist()
            combined_MTTR_matrix.extend(MTTR_matrix)
            MTTR_hours = combined_MTTR_matrix.extend(MTTR_matrix)
    return MTTR_matrices, MTTR_hours

#Function to check whether all the PCs have reliability data
def check_all_PCs_have_reliability_data(PCs, lambda_per_year, MTTR_hours):
    if len(lambda_per_year) == PCs and len(MTTR_hours) == PCs:
        print("All PCs in the network have been assigned reliability data")
    else:
        print("Some PCs in the network are missing reliability data")
        sys.exit(1)  # Cancel the script due to missing reliability data

#Function to split the matrices into smaller matrices for processing
def split_matrix(matrix, max_cols):
    matrix = np.array(matrix) # Convert to numpy array if it's not already
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1) # If the matrix is 1D, reshape it to 2D
    rows, cols = matrix.shape # Get the number of rows and columns in the original matrix
    num_splits = math.ceil(cols / max_cols) # Calculate the number of splits needed
    split_matrices = [] # Create a list to store the split matrices
    
    # Split the matrix
    for i in range(num_splits):
        start_col = i * max_cols
        end_col = min((i + 1) * max_cols, cols)
        split_matrices.append(matrix[:, start_col:end_col])
    return split_matrices

#Function to save lambda and mu dictionaries to a matfile
def save_lambda_mu_matrices(lambda_split, mu_split, mcs_parameters, target_folder) -> None:
    lambda_dict = {}
    for i, var in enumerate(lambda_split):
        lambda_dict[f'lambda_{i}'] = var
    mu_dict = {}
    for i, var in enumerate(mu_split):
        mu_dict[f'mu_{i}'] = var
    # Save the dictionary to a .mat file
    System_xtics = {
        'mcs_parameters': mcs_parameters,
        'lambda_dict': lambda_dict,
        'mu_dict': mu_dict
    }
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)

    file_path = os.path.join(target_folder, 'System_xtics.mat')
    sio.savemat(file_path, System_xtics)
