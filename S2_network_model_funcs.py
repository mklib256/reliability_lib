## Functions used in the S2_network_model

import pandapower as pp
import math
import sys
import contextlib
import io
import os
import pandas as pd

#Function to add MV buses to the network from the MVbus_data dictionary
def addbuses_MV(net, MVNetwork_data, MVbus_data):
    bus_index = 1
    MVbuses = {}

    for voltage_level, voltage in MVNetwork_data['voltage_levels'].items():
        MVbuses[voltage_level] = {}
        for section in MVNetwork_data['sections']:
            MVbuses[voltage_level][section] = {}
            for subsection, num_buses in MVbus_data[voltage_level][section].items():
                MVbuses[voltage_level][section][subsection] = []
                for i in range(num_buses):
                    if section == 'feeder':
                        feeder_id = int(subsection.replace('feeder', '')) - 1
                        bus_name = f"{voltage_level}_{section}_{MVNetwork_data['feeder_buses'][feeder_id]}{i + 1}"
                        index = int(f"{MVNetwork_data['feeder_buses'][feeder_id]}{i + 1}")
                    else:
                        bus_name = f"{voltage_level}_{section}_{subsection}_{i + 1}"
                        index = bus_index
                        bus_index += 1

                    bus = pp.create_bus(net, vn_kv=voltage, name=bus_name, index=index)
                    MVbuses[voltage_level][section][subsection].append(index)

    # Add External grid and set reference voltage at slack bus
    pp.create_ext_grid(net, bus=MVNetwork_data['slack_bus'], vm_pu=MVNetwork_data['V_ref'])

    return MVbuses

#Function to add MV buses to the network from the MVbus_data dictionary
def addbuses_MV_feeder_eq(net, MVNetwork_data, MVbus_data, feeder_buses):
    bus_index = feeder_buses[0]
    MVbuses = {}

    for voltage_level, voltage in MVNetwork_data['voltage_levels'].items():
        MVbuses[voltage_level] = {}
        for section in MVNetwork_data['sections']:
            MVbuses[voltage_level][section] = {}
            for subsection, num_buses in MVbus_data[voltage_level][section].items():
                MVbuses[voltage_level][section][subsection] = []
                for i in range(num_buses):
                    if section == 'feeder':
                        feeder_id = int(subsection.replace('feeder', '')) - 1
                        bus_name = f"{voltage_level}_{section}_{MVNetwork_data['feeder_buses'][feeder_id]}{i + 1}"
                        index = int(f"{MVNetwork_data['feeder_buses'][feeder_id]}{i + 1}")
                    else:
                        bus_name = f"{voltage_level}_{section}_{subsection}_{i + 1}"
                        index = bus_index
                        bus_index += 1

                    bus = pp.create_bus(net, vn_kv=voltage, name=bus_name, index=index)
                    MVbuses[voltage_level][section][subsection].append(index)

    # Add External grid and set reference voltage at slack bus
    pp.create_ext_grid(net, bus=MVNetwork_data['slack_bus'], vm_pu=MVNetwork_data['V_ref'])

    return MVbuses


# Function to add LV buses to the network from the LVbus_data dictionary
'''
def addbuses_LV (net, LVbus_data) -> None:
    for i in range(LVbus_data['LV_networks']):
        for j in range(LVbus_data['buses11_LV']):
            bus_index = int(f"{LVbus_data['Sec_substations'][i]}{j}")
            bus = pp.create_bus(net, index=bus_index, vn_kv=LVbus_data['voltage_levels']['MV_11kV'], name=bus_index, type="b")

        for j in range(LVbus_data['busesLV']):
            bus_index = int(f"{LVbus_data['Sec_substations'][i]}{j+LVbus_data['buses11_LV']}")
            bus = pp.create_bus(net, index=bus_index, vn_kv=LVbus_data['voltage_levels']['LV'], name=bus_index, type="b")
'''

def addbuses_LV(net, LVbus_data) -> None:
    used_indices = set()  # Track used bus indices
    for i, substation in enumerate(LVbus_data['Sec_substations']):
        # Add MV buses
        for j in range(LVbus_data['buses11_LV']):
            suffix = LVbus_data['busesLV']  # Initialize suffix
            base_index = int(f"{substation}{j}")
            bus_index = base_index
            while bus_index in used_indices:  # Ensure unique index
                suffix += 1
                bus_index = int(f"{substation}{suffix}{j}")
            pp.create_bus(net, index=bus_index, vn_kv=LVbus_data['voltage_levels']['MV_11kV'],name=f"{bus_index}", type="b")
            used_indices.add(bus_index)  # Store used index

        # Add LV buses
        for j in range(LVbus_data['busesLV']):
            suffix = LVbus_data['busesLV']  # Reset suffix for LV buses
            base_index = int(f"{substation}{j + LVbus_data['buses11_LV']}")
            bus_index = base_index
            while bus_index in used_indices:  # Ensure unique index
                suffix += 1
                bus_index = int(f"{substation}{suffix}{j + LVbus_data['buses11_LV']}")

            pp.create_bus(net, index=bus_index, vn_kv=LVbus_data['voltage_levels']['LV'],name=f"{bus_index}", type="b")
            used_indices.add(bus_index)  # Store used index


# Function to add MV CBs to the network from the MVCB_data dictionary
def add_MVCBs(net, MVCB_data, LVbus_data) -> None:
    for section, data in MVCB_data.items():
        for connection in data['connections']:
            if section != 'LV':
                from_bus = connection['from_bus']
                to_bus = connection['to_bus']
                pp.create_switch(net, bus=from_bus, element=to_bus, et='b', type="CB", closed=True)
            else:
                used_indices = set()
                for mv_bus in LVbus_data['Sec_substations']:
                    suffix = LVbus_data['busesLV']
                    from_bus = mv_bus if connection['from_bus'] is None else int(f"{mv_bus}{connection['from_bus']}")
                    to_bus = int(f"{mv_bus}{connection['to_bus']}")
                    while to_bus in used_indices:
                        suffix += 1
                        to_bus = int(f"{mv_bus}{suffix}{connection['to_bus']}")
                    used_indices.add(to_bus)
                    pp.create_switch(net, bus=from_bus, element=to_bus, et='b', type="CB", closed=True)
            
# Function to add LV CBs to the network from the MVCB_data dictionary
def addCBs_LV(net, LVbus_data, LVCBs) -> None:
    used_buses = set()
    for mv_bus in LVbus_data['Sec_substations']:
        suffix = LVbus_data['busesLV']
        if mv_bus in used_buses:
            suffix += 1
            for connection in LVCBs['connections']:
                from_bus = int(f"{mv_bus}{suffix}{connection['from_bus']}")
                to_bus = int(f"{mv_bus}{suffix}{connection['to_bus']}")
                pp.create_switch(net, bus=from_bus, element=to_bus, et='b', type="CB", closed=True)
        else:
            for connection in LVCBs['connections']:
                from_bus = int(f"{mv_bus}{connection['from_bus']}")
                to_bus = int(f"{mv_bus}{connection['to_bus']}")
                pp.create_switch(net, bus=from_bus, element=to_bus, et='b', type="CB", closed=True)
        used_buses.add(mv_bus)


#Function to add MV fuses from the MVfuses dictionary
def add_MVfuses(net, MVfuses) -> None:
    for section, data in MVfuses.items():
        for connection in data['connections']:
            from_bus = connection['from_bus']
            to_bus = connection['to_bus']
            pp.create_switch(net, bus=from_bus, element=to_bus, et='b', type="LS", closed=True)


#Function to add LV fuses in each Lv network from the LVfuses dictionary
def addfuses_LV(net, LVbus_data, LVfuses) -> None:
    used_indices = set()
    for mv_bus in LVbus_data['Sec_substations']:
        for connection in LVfuses['connections']:
            suffix = LVbus_data['busesLV']
            from_bus = int(f"{mv_bus}{connection['from_bus']}")
            while from_bus in used_indices:
                suffix += 1
                from_bus = int(f"{mv_bus}{suffix}{connection['from_bus']}")
            used_indices.add(from_bus)
            
            suffix = LVbus_data['busesLV']
            to_bus = int(f"{mv_bus}{connection['to_bus']}")
            while to_bus in used_indices:
                suffix += 1
                to_bus = int(f"{mv_bus}{suffix}{connection['to_bus']}")
            used_indices.add(to_bus)
            pp.create_switch(net, bus=from_bus, element=to_bus, et='b', type="LS", closed=True)

#Function to add MV lines from the MVline_data dictionary
def add_MVlines(net, MVline_data) -> None:
    for std_type, lengths in MVline_data.items():
        for length, sections in lengths.items():
            for section, data in sections.items():
                for connection in data['connections']:
                    pp.create_line(net,
                                   from_bus=connection['from_bus'],
                                   to_bus=connection['to_bus'],
                                   length_km=length,
                                   std_type=std_type)

#Function to add LV lines in LV each network from the LVline_data dictionary
def add_LVlines(net, LVline_data, LVbus_data) -> None:
    used_indices_from = set()
    used_indices_to = set()
    for mv_bus in LVbus_data['Sec_substations']:
        for std_type, lengths in LVline_data.items():
            for length, data in lengths.items():
                for connection in data['connections']:
                    suffix = LVbus_data['busesLV']
                    from_bus = int(f"{mv_bus}{connection['from_bus']}")
                    to_bus = int(f"{mv_bus}{connection['to_bus']}")
                    while from_bus in used_indices_from and to_bus in used_indices_to:
                        suffix += 1
                        from_bus = int(f"{mv_bus}{suffix}{connection['from_bus']}")
                        to_bus = int(f"{mv_bus}{suffix}{connection['to_bus']}")
                    used_indices_from.add(from_bus)
                    used_indices_to.add(to_bus)
                    pp.create_line(net, from_bus=from_bus, to_bus=to_bus, length_km=float(length), std_type=std_type)

#Function to add MV trafos from the MVtrafos dictionary
def addMV_trafos(net, MV_trafos) -> None:
    for std_type, data in MV_trafos.items():
        for connection in data['connections']:
            from_bus = connection[0]
            to_bus = connection[1]
            pp.create_transformer(net, hv_bus=from_bus, lv_bus=to_bus, std_type=std_type)

#Function to add LV trafos to each LV network from the LVtrafos dictionary
def addLV_trafos(net, LV_trafos, LVbus_data) -> None:
    used_indices = set()
    for mv_bus in LVbus_data['Sec_substations']:
        for std_type, data in LV_trafos.items():
            for connection in data['connections']:
                suffix = LVbus_data['busesLV']
                from_bus = int(f"{mv_bus}{int(connection[0])}")
                while from_bus in used_indices:
                    suffix += 1
                    from_bus = int(f"{mv_bus}{suffix}{connection[0]}")
                used_indices.add(from_bus)
                suffix = LVbus_data['busesLV']
                to_bus = int(f"{mv_bus}{int(connection[1])}")
                while to_bus in used_indices:
                    suffix += 1
                    to_bus = int(f"{mv_bus}{suffix}{connection[1]}")
                used_indices.add(to_bus)
                pp.create_transformer(net, hv_bus=from_bus, lv_bus=to_bus, std_type=std_type)

#Funtion to add LPs in each LV network from LPs_data
def addLPs(net, LP_data, LVbus_data) -> None:
    used_indices = set()
    for j, mv_bus in enumerate(LVbus_data['Sec_substations']):
        for i, load in enumerate(LP_data['loads']):
            suffix = LVbus_data['busesLV']
            LP_bus = int(f"{mv_bus}{load['bus']}")
            while LP_bus in used_indices:
                suffix += 1
                LP_bus = int(f"{mv_bus}{suffix}{load['bus']}")
            used_indices.add(LP_bus)
            LP_mw = load['customers'] * LP_data['LP_per_customer']
            LP_mvar = LP_mw * math.tan(math.acos(LP_data['power_factor']))
            pp.create_load(net, bus=LP_bus, p_mw=LP_mw, q_mvar=LP_mvar, name=f"LP_{i+1}")


#Funtion to add LPs to the buses connected to the MV buses connected to the equivalent networks
def addLPs_MV(net, LP_data, LVbus_data):
    for j, substation in enumerate(LVbus_data['Sec_substations_reduced']):
        for i, load in enumerate(LP_data['loads']):
            LP_bus = substation
            LP_mw = load['customers'] * LP_data['LP_per_customer']
            LP_mvar = LP_mw * math.tan(math.acos(LP_data['power_factor']))
            pp.create_load(net, bus=LP_bus, p_mw=LP_mw, q_mvar=LP_mvar, name=f"LP_{i+1}")

#Funtion to add LPs to the buses connected to the feeders with the equivalent networks
def addLPs_MV_feeder_eq(net, LP_data, LVbus_data):
    for substation, num_LVnetworks in zip(LVbus_data['Sec_substations_reduced'], LVbus_data['feeder_buses_LVnetworks']):
        for load_index, load in enumerate(LP_data['loads']):
            for network_number in range(num_LVnetworks):
                LP_mw = load['customers'] * LP_data['LP_per_customer']
                LP_mvar = LP_mw * math.tan(math.acos(LP_data['power_factor']))
                pp.create_load(net, bus=substation, p_mw=LP_mw, q_mvar=LP_mvar,
                               name=f"LP_Sub{substation}_Net{network_number + 1}_Load{load_index + 1}")


#Function to run a diagnostic on the network. If there are issues, exit and print the exception
def network_diagnostic(net):
    try:
        # Suppress output from diagnostic
        with contextlib.redirect_stdout(io.StringIO()):
            diagnostic_result = pp.diagnostic(net, report_style='compact', warnings_only=True, return_result_dict=True)

        # Check if there are any issues in the diagnostic result
        if any(diagnostic_result.values()):
            raise Exception("Diagnostic found issues in the network")
        # If no issues, the script will continue. If any issue is picked up, the script will be cancelled
    except Exception as e:
        print(f"Script canceled: {str(e)}")
        sys.exit(1)  # Cancel the script due to error in the network

#Function to run a power flow, print network results to an excel file and save a json with the network data in the target folder
def print_network_results(net, target_folder):
    pp.runpp(net)

    file_path = os.path.join(target_folder, 'network_results.xlsx')
    if os.path.exists(file_path):
        os.remove(file_path)

    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        # Save bus results
        net.res_bus.to_excel(writer, sheet_name='Bus Results')

        # Save line results
        net.res_line.to_excel(writer, sheet_name='Line Results')

        # Save transformer results
        net.res_trafo.to_excel(writer, sheet_name='Transformer Results')

        # Save load results
        net.res_load.to_excel(writer, sheet_name='Load Results')

        # Save switch results
        net.switch.to_excel(writer, sheet_name='Switch Results')

    # Save the network to a JSON file
    file_path = os.path.join(target_folder, 'network.json')
    pp.to_json(net, file_path)





