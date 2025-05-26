#Functions used in S5_panda_sim

import pandapower as pp
import numpy as np
import h5py
import mat73
import os
import re
import pandas as pd
import scipy.io

#Function to extract D matrices and combine them into one PC state matrix
def load_D_files(target_folder):
    D_files = [f for f in os.listdir(target_folder) if re.match(r'D_\d+\.mat$', f)]
    D_files_sorted = sorted(D_files, key=lambda x: int(re.search(r'D_(\d+)\.mat$', x).group(1)))
    D_matrices = {}  # Initialize an empty dictionary to hold combined data

    for mat_file in D_files_sorted:
        full_path = os.path.join(target_folder, mat_file)
        data_dict = mat73.loadmat(full_path)  # Load the .mat file into a dictionary
        for key, value in data_dict.items():
            if key not in D_matrices:
                D_matrices[key] = value  # Initialize with first file's data
            else:
                D_matrices[key] = np.concatenate((D_matrices[key], value), axis=1)  # Concatenate the D matrices

    D = D_matrices['D']  # Load the concatenated D matrix
    return D

#Function to extract D matrices and combine them into one PC state matrix
def load_D_files_mat(target_folder):
    D_files = [f for f in os.listdir(target_folder) if re.match(r'D_\d+\.mat$', f)]
    D_files_sorted = sorted(D_files, key=lambda x: int(re.search(r'D_(\d+)\.mat$', x).group(1)))
    D_list = [] # Initialize an empty array to hold combined data

    for mat_file in D_files_sorted:
        full_path = os.path.join(target_folder, mat_file)
        data_dict = scipy.io.loadmat(full_path)
        D = data_dict['D'] # Extract D matrix
        if D.ndim == 0:  # Handle scalar values
            D = np.array([[D.item()]])
        elif D.ndim == 1:  # Convert 1D arrays to 2D column vectors
            D = D.reshape(-1, 1)
        D_list.append(D)

    D_combined = np.concatenate(D_list, axis=1) if D_list else np.array([]) # Concatenate along columns (axis=1)
    return D_combined

#Function to extract B matrices and combine them into one PC state matrix
def load_B_files(target_folder):
    B_files = [f for f in os.listdir(target_folder) if re.match(r'B_\d+\.mat$', f)] # List all .mat files in the directory that match the pattern B_i.mat
    B_matrices = {}  # Initialize an empty dictionary to hold combined data

    for mat_file in B_files:
        full_path = os.path.join(target_folder, mat_file)
        data_dict = mat73.loadmat(full_path)  # Load the .mat file into a dictionary
        for key, value in data_dict.items():
            if key not in B_matrices:
                B_matrices[key] = value  # Initialize with first file's data
            else:
                B_matrices[key] = np.concatenate((B_matrices[key], value), axis=1)  # Concatenate the B matrices

    B = B_matrices['B']  # Load the concatenated B matrix
    return B

#Function to extract D matrices and combine them into one PC state matrix considering only the [startIndex:startIndex + tsperyear] indices in the simulation
def load_D_sim(target_folder, startIndex, ts):
    D_files = [f for f in os.listdir(target_folder) if re.match(r'D_\d+\.mat$', f)] # List all .mat files in the directory that match the pattern D_i.mat
    D_matrices = {}  # Initialize an empty dictionary to hold combined data

    for mat_file in D_files:
        full_path = os.path.join(target_folder, mat_file)
        data_dict = mat73.loadmat(full_path)  # Load the .mat file into a dictionary
        for key, value in data_dict.items():
            if key not in D_matrices:
                D_matrices[key] = value[startIndex:startIndex + ts]  # Initialize with first file's data
            else:
                D_matrices[key] = np.concatenate((D_matrices[key], value[startIndex:startIndex + ts]), axis=1)  # Concatenate the D matrices
    D_year = D_matrices['D']  # Load the concatenated D matrix
    return D_year


#Function to compare the number of random failures in the simulation and the expected failures at the constant failure rate in an excel sheet
def failure_calc(D, lambda_per_year, Years, target_folder):
    zero_count = []     # Initialize an array to store the counts

    for col in range(D.shape[1]): # Loop through each column
        column = D[:, col] # Load the column for the PC states (ON/OFF)
        count = 1 if column[0] == 0 else 0 # Always count the first row if it's zero

        # Find zeros that are not preceded by another zero in the same column
        valid_zeros = (column[1:] == 0) & (column[:-1] != 0)  # Skip the first row for this check
        count += valid_zeros.sum() # Add the count of valid zeros
        zero_count.append(count)

    actual_failures = np.array(zero_count)
    lambda_failures = lambda_per_year * Years
    diff = actual_failures - lambda_failures

    # Create a dictionary with your arrays
    data = {
        'lambda_failures': lambda_failures,
        'actual_failures': actual_failures,
        'diff': diff
    }
    df = pd.DataFrame(data)
    excel_file_path = os.path.join(target_folder, 'Failures.xlsx')
    df.to_excel(excel_file_path, index=False)

#Function to identify the rows with zeros in the PC state matrix D
def identify_zeros_D(D):
    rows_with_zeros = np.any(D == 0, axis=1)
    indices_with_zeros = np.where(rows_with_zeros)[0]
    sims = len(indices_with_zeros)
    return indices_with_zeros, sims

#Function to allocate indices in the PC column of the D matrix to each of the PCs in the network
#Matrix with the number of PCs in each PC type
#PCs are switched off according to the order in the PCs_num matrix
#PCs_num = {'buses33': buses33, 'buses11': buses11, 'busesMV': busesLV, 'CBs33': CBs33, 'CBs11': CBs11, 'CBsLV': CBsLV,
 #          'fusesMV': fusesMV, 'fusesLV': fusesLV, 'lines': lines, 'trafos3311': trafos3311, 'trafos1104': trafos1104}
 #Lines 'S_100_1500m','S_100_1000m','S_100_500m','T_50_1000m','T_50_500m','J_50_35m','K_35_30m','M_25_30m'

def allocate_indices_to_PCtemp(PCtemp, PCs_num):
    PCtemp_slices = {} # Initialize a dictionary to hold the slices
    start_index = 0 # Define the starting index for slicing

    # Allocate indices for each component type in PCs_num
    # Buses
    PCtemp_slices['buses33'] = PCtemp[start_index:start_index + PCs_num['buses33']]
    start_index += PCs_num['buses33']

    PCtemp_slices['buses11'] = PCtemp[start_index:start_index + PCs_num['buses11']]
    start_index += PCs_num['buses11']

    PCtemp_slices['busesLV'] = PCtemp[start_index:start_index + PCs_num['busesLV']]
    start_index += PCs_num['busesLV']

    # Circuit Breakers
    PCtemp_slices['CBs33'] = PCtemp[start_index:start_index + PCs_num['CBs33']]
    start_index += PCs_num['CBs33']

    PCtemp_slices['CBs11'] = PCtemp[start_index:start_index + PCs_num['CBs11']]
    start_index += PCs_num['CBs11']

    PCtemp_slices['CBsLV'] = PCtemp[start_index:start_index + PCs_num['CBsLV']]
    start_index += PCs_num['CBsLV']

    # Fuses
    PCtemp_slices['fusesMV'] = PCtemp[start_index:start_index + PCs_num['fusesMV']]
    start_index += PCs_num['fusesMV']

    PCtemp_slices['fusesLV'] = PCtemp[start_index:start_index + PCs_num['fusesLV']]
    start_index += PCs_num['fusesLV']

    # Lines (nested dictionary)
    for line_type, count in PCs_num['lines'].items():
        PCtemp_slices[line_type] = PCtemp[start_index:start_index + count]
        start_index += count

    # Transformers
    PCtemp_slices['trafos3311'] = PCtemp[start_index:start_index + PCs_num['trafos3311']]
    start_index += PCs_num['trafos3311']

    PCtemp_slices['trafos1104'] = PCtemp[start_index:start_index + PCs_num['trafos1104']]

    return PCtemp_slices

#Function to update the status of PCs in the network according to the failure state in the PC matrix
def update_component_status(net, PCtemp_slices, bus_indices, lines) -> None:
    # Update buses using bus_indices
    bus_types = ['buses33', 'buses11', 'busesLV']
    start_index = 0
    for bus_type in bus_types:
        end_index = start_index + len(PCtemp_slices[bus_type])
        bus_slice = bus_indices[start_index:end_index]
        net.bus.loc[bus_slice, 'in_service'] = PCtemp_slices[bus_type].astype(bool)
        start_index = end_index

    # Update switches (CBs and fuses)
    switch_types = ['CBs33', 'CBs11', 'CBsLV', 'fusesMV', 'fusesLV']
    start_index = 0
    for switch_type in switch_types:
        end_index = start_index + len(PCtemp_slices[switch_type])
        net.switch.iloc[start_index:end_index, net.switch.columns.get_loc('closed')] = PCtemp_slices[switch_type].astype(bool)
        start_index = end_index

    # Update lines using the provided line types
    start_index = 0
    for line_type, count in lines.items():
        if line_type in PCtemp_slices:
            end_index = start_index + count
            net.line.iloc[start_index:end_index, net.line.columns.get_loc('in_service')] = PCtemp_slices[line_type].astype(bool)
            start_index = end_index

    # Update transformers
    start_index = 0
    for trafo_type in ['trafos3311', 'trafos1104']:
        end_index = start_index + len(PCtemp_slices[trafo_type])
        net.trafo.iloc[start_index:end_index, net.trafo.columns.get_loc('in_service')] = PCtemp_slices[trafo_type].astype(bool)
        start_index = end_index

#Function for saving load matrix LF in HDF5 matrix
def save_LF(target_folder, LF) -> None :
    file_path = os.path.join(target_folder, 'LF.h5')
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('LF', data=LF, compression="gzip", compression_opts=9)  # Save data to HDF5 format


