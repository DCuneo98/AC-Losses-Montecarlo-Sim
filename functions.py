#################
### LIBRARIES ###
#################
import numpy as np
import matplotlib.pyplot as plt
import os

##################################
### FUNCTIONS INCLUDED IN MAIN ###
##################################

def time_delay(time_delay_DCCT, time_delay_cRIO, time_delay_ISOBLOCK, time_delay_correction): 
    time_delay_current = np.random.uniform(low = time_delay_DCCT[0], high = time_delay_DCCT[1]) + np.random.uniform(low = time_delay_cRIO-0.01*time_delay_cRIO, high = time_delay_cRIO+0.01*time_delay_cRIO)
    time_delay_voltage = np.random.uniform(low = time_delay_cRIO-0.01*time_delay_cRIO,  high = time_delay_cRIO+0.01*time_delay_cRIO) + np.random.uniform(low = time_delay_ISOBLOCK[0], high = time_delay_ISOBLOCK[1])
    time_delay_corr = time_delay_current + np.random.uniform(low = time_delay_correction-0.1*time_delay_correction, high = time_delay_correction+0.1*time_delay_correction)

    return time_delay_current, time_delay_voltage, time_delay_corr

def sinusoidal_transition(x, x0, x1, y0, y1):
    t = (x - x0) / (x1 - x0)
    t = np.clip(t, 0, 1)
    return y0 + (y1 - y0) * 0.5 * (1 - np.cos(np.pi * t))

def generate_current_ramp(I_min, I_max, time_plateau, t, time_offset, time_ramp, period):

    time_offset = np.random.uniform(time_offset - 0.01 * time_offset, time_offset + 0.01 * time_offset)
    t_mod = np.mod(t - time_offset, period)

    I = np.zeros_like(t)

    for i, t_val in enumerate(t):
        if t_val < time_offset:
            # Ensure zero before the time offset
            I[i] = 0
        else:
            t_mod_val = t_mod[i]
            if t_mod_val < time_ramp:
                # Ramp up
                I[i] = I_min + (I_max - I_min) * (t_mod_val / time_ramp)
            elif t_mod_val < time_ramp + time_plateau:
                # Plateau at max
                I[i] = sinusoidal_transition(t_mod_val, time_ramp, time_ramp + time_plateau, I_max, I_max)
            elif t_mod_val < 2 * time_ramp + time_plateau:
                # Ramp down
                I[i] = I_max - (I_max - I_min) * ((t_mod_val - time_ramp - time_plateau) / time_ramp)
            else:
                # Plateau at min
                I[i] = sinusoidal_transition(t_mod_val, 2 * time_ramp + time_plateau, period, I_min, I_min)

    return I

def current_reading(current, k_DCCT, gain_error_DCCT, offset_DCCT, gain_error_cRIO, offset_error_cRIO, range_cRIO, ADC_resolution): 
    total_samples = len(current)
    I_measured = ((1+ gain_error_DCCT* np.random.randn(total_samples))*current)* k_DCCT + np.random.uniform(low = -offset_DCCT, high=offset_DCCT, size= total_samples) #Adding the uncertainty of DCCT
    I_measured = ((1+ gain_error_cRIO* np.random.randn(total_samples))* I_measured) + range_cRIO*np.random.uniform(low=-offset_error_cRIO, high=offset_error_cRIO, size=total_samples)#Adding the uncertainty of cRIO
    I_measured = np.round(I_measured / ADC_resolution) * ADC_resolution #Adding the quantization error
    
    del current
    
    return I_measured
    
def voltage_magnet_and_voltage_derivative_sensor(current, inductance, resistance, proportionality_factor, gain_ds, offset_ds, time_interval):
    # Derivative of current
    dI_dt = np.concatenate((np.diff(current), [0])) / time_interval

    # Magnet voltage in the time domain
    voltage_magnet = (resistance * current + inductance * dI_dt)

    k_ds = ((1 + gain_ds * np.random.randn(len(current)))*proportionality_factor) + np.random.uniform(low = -offset_ds, high=offset_ds, size= len(current))
    voltage_ds_measured = k_ds * dI_dt
    
    del dI_dt, k_ds
    
    return voltage_magnet, voltage_ds_measured

def voltage_correction(current, proportionality_factor, offset_correction, time_interval):
    dI_dt = np.concatenate((np.diff(current), [0])) / time_interval
    
    k_ds_correction = proportionality_factor + np.random.uniform(low = -offset_correction, high=offset_correction, size= len(current))
    voltage_correction = k_ds_correction* dI_dt
    
    del dI_dt, k_ds_correction
    
    return voltage_correction

def voltages_post_ISOBLOCK(voltage_magnet, voltage_derivative_sensor, attenuation_factor_ISOBLOCK, gain_error_ISOBLOCK, offset_error_ISOBLOCK): 
    total_samples = len(voltage_magnet)

    voltage_magnet_ISO = (((1 + gain_error_ISOBLOCK * np.random.randn(total_samples)) * voltage_magnet) / attenuation_factor_ISOBLOCK) + offset_error_ISOBLOCK * np.random.uniform(low=-1, high=1, size=total_samples)
    voltage_ds_measured_ISO = (((1 + gain_error_ISOBLOCK * np.random.randn(total_samples)) * voltage_derivative_sensor) / attenuation_factor_ISOBLOCK) + offset_error_ISOBLOCK * np.random.uniform(low=-1, high=1, size=total_samples)
    
    del voltage_magnet, voltage_derivative_sensor
    
    return voltage_magnet_ISO, voltage_ds_measured_ISO

def voltages_in_cRIO(voltage_magnet, voltage_derivative_sensor, gain_error_cRIO, offset_error_cRIO, range_cRIO, ADC_resolution): 
    total_samples = len(voltage_magnet)
    
    voltage_magnet_cRIO = ((1 + gain_error_cRIO * np.random.randn(total_samples)) * voltage_magnet) + (offset_error_cRIO * np.random.uniform(low=-1, high=1, size=total_samples) * range_cRIO)
    voltage_ds_cRIO = ((1 + gain_error_cRIO * np.random.randn(total_samples)) * voltage_derivative_sensor) + (offset_error_cRIO * np.random.uniform(low=-1, high=1, size=total_samples) * range_cRIO)
    
    del voltage_magnet, voltage_derivative_sensor
        
    voltage_magnet_cRIO = np.round(voltage_magnet_cRIO / ADC_resolution) * ADC_resolution
    voltage_ds_cRIO = np.round(voltage_ds_cRIO / ADC_resolution) * ADC_resolution
    
    return voltage_magnet_cRIO, voltage_ds_cRIO

def power_estimation(voltage_magnet, voltage_derivative_sensor, voltage_correction, current_measured, current_correction, k_DCCT, attenuation_factor_ISOBLOCK ):
    
    magnet_power_no_comp = (voltage_magnet * attenuation_factor_ISOBLOCK) * (current_measured / k_DCCT)

    magnet_power_comp = ((voltage_magnet - voltage_derivative_sensor) * attenuation_factor_ISOBLOCK) * (current_measured / k_DCCT)
    
    magnet_power_corr = (((voltage_magnet) * attenuation_factor_ISOBLOCK) -  voltage_correction) * (current_correction / k_DCCT)
    
    del voltage_magnet, voltage_derivative_sensor, voltage_correction, current_measured, current_correction
    
    return magnet_power_no_comp, magnet_power_comp, magnet_power_corr

def simulate(I_min, I_max,  time_delay_DCCT, time_delay_cRIO, time_delay_ISOBLOCK, time_delay_correction, time_plateau, time_ramp, t,  L_magnet, R_magnet, kds, gain_ds, offset_ds, offset_correction, 
             dt, period, attenuation_factor_ISOBLOCK, gain_error_ISOBLOCK, offset_error_ISOBLOCK, 
             gain_error_cRIO, offset_error_cRIO, range_cRIO, ADC_resolution, k_DCCT, gain_error_DCCT, offset_DCCT, cycles):
    
    time_delay_current, time_delay_voltage, time_delay_corr = time_delay(time_delay_DCCT, time_delay_cRIO, time_delay_ISOBLOCK, time_delay_correction)
        
    I_voltage = generate_current_ramp(I_min, I_max, time_plateau, t, time_delay_voltage, time_ramp, period)
    I_current = generate_current_ramp(I_min, I_max, time_plateau, t, time_delay_current, time_ramp, period)
    I_correction = generate_current_ramp(I_min, I_max, time_plateau, t, time_delay_corr, time_ramp, period)
    
    voltage_magnet, voltage_ds_measured = voltage_magnet_and_voltage_derivative_sensor(I_voltage, L_magnet, R_magnet, kds, gain_ds, offset_ds, dt)
    voltage_corr = voltage_correction(I_voltage, kds, offset_correction, dt)
    
    voltage_magnet, voltage_ds_measured = voltages_post_ISOBLOCK(voltage_magnet, voltage_ds_measured, attenuation_factor_ISOBLOCK, gain_error_ISOBLOCK, offset_error_ISOBLOCK) 
    voltage_magnet, voltage_ds_measured= voltages_in_cRIO(voltage_magnet, voltage_ds_measured, gain_error_cRIO, offset_error_cRIO, range_cRIO, ADC_resolution)
    
    I_measured = current_reading(I_current, k_DCCT, gain_error_DCCT, offset_DCCT, gain_error_cRIO, offset_error_cRIO, range_cRIO, ADC_resolution)
    I_measured_correction = current_reading(I_correction, k_DCCT, gain_error_DCCT, offset_DCCT, gain_error_cRIO, offset_error_cRIO, range_cRIO, ADC_resolution)
    
    magnet_power_no_comp, magnet_power_comp, magnet_power_corr = power_estimation(voltage_magnet, voltage_ds_measured, voltage_corr, I_measured, I_measured_correction, k_DCCT, attenuation_factor_ISOBLOCK)

    del voltage_magnet, voltage_ds_measured, voltage_corr, I_measured, I_measured_correction
    
    # Calcolo delle medie delle potenze 
    end_indices = (cycles * period / dt).astype(int)
    
    magnet_power_no_comp_avg = np.array([np.mean(magnet_power_no_comp[:end_idx]) for end_idx in end_indices])
    magnet_power_comp_avg = np.array([np.mean(magnet_power_comp[:end_idx]) for end_idx in end_indices])
    magnet_power_corr_avg = np.array([np.mean(magnet_power_corr[:end_idx]) for end_idx in end_indices])
    
    del magnet_power_no_comp, magnet_power_comp, magnet_power_corr
    
    return magnet_power_no_comp_avg, magnet_power_comp_avg, magnet_power_corr_avg

def statistics_calculation(magnet_power_no_comp_mean, magnet_power_comp_mean, magnet_power_corr_mean):
    mean_power_no_comp = np.mean(magnet_power_no_comp_mean, axis=0)
    mean_power_comp = np.mean(magnet_power_comp_mean, axis=0)
    mean_power_corr = np.mean(magnet_power_corr_mean, axis=0)
    
    std_power_no_comp = np.std(magnet_power_no_comp_mean, axis=0)
    std_power_comp = np.std(magnet_power_comp_mean, axis=0)
    std_power_corr = np.std(magnet_power_corr_mean, axis=0)
    
    del magnet_power_no_comp_mean, magnet_power_comp_mean, magnet_power_corr_mean
    
    return  mean_power_no_comp, mean_power_comp, mean_power_corr, std_power_no_comp, std_power_comp, std_power_corr

def write_results_to_file(file_path, magnet_power_no_comp_mean, magnet_power_comp_mean, magnet_power_corr_avg, mean_power_no_comp, std_power_no_comp, mean_power_comp, std_power_comp, mean_power_corr, std_power_corr):
    if os.path.exists(file_path):
        os.remove(file_path)
    
    with open(file_path, 'w') as file:
        # Write the non-compensated mean power
        file.write("Mean power (not compensated)(acquisition):\n")
        np.savetxt(file, magnet_power_no_comp_mean, delimiter=",")
        
        # Write the compensated mean power
        file.write("\nMean power (compensated)(acquisition):\n")
        np.savetxt(file, magnet_power_comp_mean, delimiter=",")
        
        # Write the corrected mean power
        file.write("\nMean power (corrected):\n")
        np.savetxt(file, magnet_power_corr_avg, delimiter=",")
        
        # Write the final results for non-compensated power
        file.write("\nFinal results for non-compensated power:\n")
        file.write(f"Mean power for non-compensated: {mean_power_no_comp}\n")
        file.write(f"Std of power for non-compensated: {std_power_no_comp}\n")
        
        # Write the final results for compensated power
        file.write("\nFinal results for compensated power:\n")
        file.write(f"Mean power for compensated: {mean_power_comp}\n")
        file.write(f"Std of power for compensated: {std_power_comp}\n")
        
        # Write the final results for non-compensated power
        file.write("\nFinal results for corrected power:\n")
        file.write(f"Mean power for corrected: {mean_power_corr}\n")
        file.write(f"Std of power for corrected: {std_power_corr}\n")
    
def salva_dati_iterazione(nome_file, vettore):
    # Converte il vettore in tipo float32 per ridurre l'uso della memoria
    vettore = vettore.astype(np.float32)
    
    # Apre il file in modalità 'append' per aggiungere una nuova riga
    with open(nome_file, 'a') as f:
        # Scrive il vettore nel file come una nuova riga
        np.savetxt(f, vettore.reshape(1, -1), delimiter=',', fmt='%.8e')  # Formato scientifico per float32
        
######################
### PLOT FUNCTIONS ###
######################

def plot_power_cycles(cycle_number_array, mean_power_no_comp, mean_power_comp, mean_power_corr, std_power_no_comp, std_power_comp, std_power_corr): 
    P_ac = 37.6  # Power losses in W
    plt.figure(figsize=(12, 6))
    
    plt.errorbar(cycle_number_array, mean_power_no_comp, yerr=std_power_no_comp, fmt='o', capsize=5, label='Power losses: NC')
    plt.errorbar(cycle_number_array, mean_power_comp, yerr=std_power_comp, fmt='o', capsize=5, label='Power losses: compensation')
    plt.errorbar(cycle_number_array, mean_power_corr, yerr=std_power_corr, fmt='o', capsize=5, label='Power losses: correction')
    
    plt.axhline(y=P_ac, xmin=0, xmax=1, linestyle='dashed', color='k', label='Power losses DISCORAP')
    plt.xlabel('Number of acquisition cycles')
    plt.ylabel('Power losses (W)')
    plt.title('Power Losses vs Number of acquisition cycles')
    plt.legend()
    plt.grid(True)
    plt.xticks(cycle_number_array)    
    plt.savefig('power_losses_statistics.svg', format='svg')
    plt.show()

def plot_current_cycle(I_min, I_max, I_ramp_rate, time_plateau, dt, N_max):
    # Calculate the cycle parameters
    time_transient = (I_max - I_min) / I_ramp_rate
    period = 2 * time_plateau + 2 * time_transient  # Period of the cycle in s
    total_time = N_max * period  # Total time for N_max cycles
    t = np.arange(0, total_time, dt, dtype=np.float32)  # Time array

    # Generate the current cycle
    I = generate_current_ramp(I_min, I_max, I_ramp_rate, 1 / period, t)

    # Plot the current cycle
    plt.figure(figsize=(10, 6))
    plt.plot(t, I, label=f'Current Cycle for N={N_max} acquisition cycles')
    plt.xlabel('Time (s)')
    plt.ylabel('Current (A)')
    plt.title(f'Current Cycle for N={N_max}')
    plt.grid(True)
    plt.savefig('current_cycle.svg', format='svg')
    plt.show()
    
def plot_power_distribution(magnet_power_mean, column_index, filename):
    # Extract the first column of the vector
    power_to_distribute = magnet_power_mean[:, column_index]
    # Create the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(power_to_distribute, edgecolor='black')
    plt.xlabel('Mean Power (Not Compensated)')
    plt.ylabel('Frequency')
    plt.title('Histogram of the Mean Power (Not Compensated) for N = 4')
    plt.grid(True)
    plt.savefig(filename, format='svg')
    plt.show()
  
