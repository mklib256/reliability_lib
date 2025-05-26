## Functions used in the S1_network_model_inputs
import numpy as np
import os

# Function to create the MVbus_dictionary with the number of buses in each section at each voltage level
def create_MVbus_data(MVNetwork_data, bus_data):
    MVbus_data = {}  # Initialize dictionary for MV bus data
    voltage_levels = list(MVNetwork_data['voltage_levels'].keys())
    sections = list(MVNetwork_data['sections'].keys())

    for voltage in voltage_levels:
        MVbus_data[voltage] = {}
        for section in sections:
            section_data = bus_data[f"{voltage}_{section}"]
            MVbus_data[voltage][section] = {}
            for i, bus_count in enumerate(section_data):
                MVbus_data[voltage][section][f'{section}{i + 1}'] = bus_count
    return MVbus_data

# Function to add MVCB data from the lists to the dictionary
def create_MVCB_data(sections, CB_counts, connections):
    MVCB_data = {}
    for section, CB_count, section_connections in zip(sections, CB_counts, connections):
        MVCB_data[section] = {
            f'CBsMV_{section.lower()}': CB_count,
            'connections': [
                {'from_bus': from_bus, 'to_bus': to_bus}
                for from_bus, to_bus in section_connections
            ]
        }
    return MVCB_data

# Function to create dictionary with the bus lists for the MVfuses
def create_MVfuses(sections, fuse_counts, connections):
    MVfuses = {}
    for section, fuse_count, section_connections in zip(sections, fuse_counts, connections):
        MVfuses[section] = {
            f'fusesMV_{section.lower()}': fuse_count,
            'connections': [
                {'from_bus': from_bus, 'to_bus': to_bus}
                for from_bus, to_bus in section_connections
            ]
        }
    return MVfuses

# Function to add the MVline data from the lists to the dictionary
# Run for each std type
def add_MVline_data(MVline_data, stdtype, length_data):
    if stdtype not in MVline_data:
        MVline_data[stdtype] = {}

    for length, sections_data in length_data.items():
        if length not in MVline_data[stdtype]:
            MVline_data[stdtype][length] = {}

        for section, data in sections_data.items():
            MVline_data[stdtype][length][section] = {
                'connections': [
                    {
                        'from_bus': conn[0],
                        'to_bus': conn[1]
                    }
                    for conn in data['connections']
                ]
            }
    return MVline_data

# Function to create LV_bus_data
def update_LVbus_data(LVbus_data, Sec_substations_data, Sec_substations_reduced_data, feeder_buses_LVnetworks):
    LVbus_data['Sec_substations'] = Sec_substations_data
    LVbus_data['LV_networks'] = len(Sec_substations_data)
    LVbus_data['Sec_substations_reduced'] = Sec_substations_reduced_data
    LVbus_data['LVnetworks_LP'] = len(Sec_substations_reduced_data)
    LVbus_data['feeder_buses_LVnetworks'] = feeder_buses_LVnetworks

    return LVbus_data

# Function to add LV line data
def add_LVline_data(LVline_data, stdtype, length_data):
    if stdtype not in LVline_data:
        LVline_data[stdtype] = {}

    for length, data in length_data.items():
        LVline_data[stdtype][length] = {
            'connections': [
                {
                    'from_bus': conn[0],
                    'to_bus': conn[1]
                }
                for conn in data['connections']
            ]
        }
    return LVline_data

# Function for creating LPs at the LV load points
def calculate_lp_power(LP_data):
    # Calculate real power for each LP
    for load in LP_data['loads']:
        load['mw'] = load['customers'] * LP_data['LP_per_customer']

    # Calculate reactive power for each LP
    for load in LP_data['loads']:
        load['mvar'] = load['customers'] * LP_data['LP_per_customer'] * np.tan(np.arccos(LP_data['power_factor']))

    # Generate LP_bus, LPs_mw, and LPs_mvar lists
    LP_buses = [load['bus'] for load in LP_data['loads']]
    LPs_mw = [load['mw'] for load in LP_data['loads']]
    LPs_mvar = [load['mvar'] for load in LP_data['loads']]

    return LP_buses, LPs_mw, LPs_mvar

# Function to create results folder and results text file and add title
def create_Results_file(text) -> None:
    results_folder = 'Results'
    if not os.path.exists(results_folder): # Check if the Results folder exists
        os.mkdir(results_folder) #, Create the folder if it doesn't exist

    file_path = os.path.join(results_folder, 'Results.txt')
    if os.path.exists(file_path):
        os.remove(file_path)     # Delete the Results.txt file if it exists

    with open(file_path, 'a') as file: # Create new Results.txt file
        file.write(f"{text}\n") # Write text new Results.txt file

    return results_folder

# Function to add to results text file
def add_to_Results_file(text, results_folder) -> None:
    file_path = os.path.join(results_folder, 'Results.txt')
    with open(file_path, 'a') as file:
        file.write(text)

# Function to save network data dictionaries in the results folder under the Network_data sub-folder
def save_dictionary_to_file(dictionary, dict_name):
    dict_folder = 'Network_data'
    if not os.path.exists(dict_folder):
        os.mkdir(dict_folder) # Create the folder if it doesn't exist
    file_path = os.path.join(dict_folder, f"{dict_name}.txt")
    def write_dict_to_file(d, file, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):  # If the value is a dictionary, recurse
                file.write(f"{' ' * indent}{key}:\n")
                write_dict_to_file(value, file, indent + 2)
            else:  # Otherwise, write the key-value pair
                file.write(f"{' ' * indent}{key}: {value}\n")
    # Save the dictionary to the file
    with open(file_path, 'w') as file:
        write_dict_to_file(dictionary, file)
