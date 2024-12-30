import numpy as np
import matplotlib.pyplot as plt
import os

### FUNCTIONS INCLUDED IN THE MAIN PARALLELIZED SCRIPT ###

def sinusoidal_transition(x, x0, x1, y0, y1):
    """Generates a sinusoidal transition between two points (x0, y0) and (x1, y1)."""
    t = (x - x0) / (x1 - x0)
    t = np.clip(t, 0, 1)
    return y0 + (y1 - y0) * 0.5 * (1 - np.cos(np.pi * t))

def generate_current_ramp(I_min, I_max, I_ramp_rate, frequency, t, uncertainty_percent=1):
    I_average = (I_max + I_min) / 2
    
    time_plateau = 3.5
    time_transient = (I_max - I_min) / I_ramp_rate
    period = 2 * time_plateau + 2 * time_transient  # Period of a ramp cycle in s
    
    # Time points for the ramp
    t_mod = np.mod(t, period)
    
    I = np.zeros_like(t)
    
    for i, t_val in enumerate(t_mod):
        if t_val < time_transient:
            # Ramp up
            I[i] = sinusoidal_transition(t_val, 0, time_transient, I_min, I_max)
        elif t_val < time_transient + time_plateau:
            # Plateau at max
            I[i] = I_max
        elif t_val < 2 * time_transient + time_plateau:
            # Ramp down
            I[i] = sinusoidal_transition(t_val, time_transient + time_plateau, 2 * time_transient + time_plateau, I_max, I_min)
        else:
            # Plateau at min
            I[i] = I_min

    return I

def voltage_magnet_and_voltage_derivative_sensor(current, inductance, resistance, proportionality_factor, proportionality_factor_HW, time_interval):
    # Derivative of current
    dI_dt = np.concatenate((np.diff(current), [0])) / time_interval

    # Magnet voltage in the time domain
    voltage_magnet = (resistance * current + inductance * dI_dt)

    k_ds = np.random.uniform(proportionality_factor - proportionality_factor_HW, proportionality_factor + proportionality_factor_HW)

    # Calculate the voltage measured by the derivative sensor
    voltage_ds_measured = (k_ds * dI_dt)
    
    # Calculate power losses
    #voltage_ds_measured = (proportionality_factor * (1 + proportionality_factor_stdv * np.random.randn(len(current)))) * dI_dt
   
    del dI_dt
    
    return voltage_magnet, voltage_ds_measured

def voltages_post_ISOBLOCK(voltage_magnet, voltage_derivative_sensor, total_samples, attenuation_factor_ISOBLOCK, gain_error_ISOBLOCK, offset_error_ISOBLOCK): 
   
    voltage_magnet_ISO = (((1 + gain_error_ISOBLOCK * np.random.uniform(low=-1, high=1, size=total_samples)) * voltage_magnet) / attenuation_factor_ISOBLOCK) + offset_error_ISOBLOCK * np.random.uniform(low=-1, high=1, size=total_samples)
    voltage_ds_measured_ISO = (((1 + gain_error_ISOBLOCK * np.random.uniform(low=-1, high=1, size=total_samples)) * voltage_derivative_sensor) / attenuation_factor_ISOBLOCK) + offset_error_ISOBLOCK * np.random.uniform(low=-1, high=1, size=total_samples)

    del voltage_magnet, voltage_derivative_sensor
    
    return voltage_magnet_ISO, voltage_ds_measured_ISO

def voltages_in_cRIO(voltage_magnet, voltage_derivative_sensor, total_samples, gain_error_cRIO, offset_error_cRIO, range_cRIO, ADC_resolution): 

    voltage_magnet_cRIO = ((1 + gain_error_cRIO * np.random.uniform(low=-1, high=1, size=total_samples)) * voltage_magnet) + (offset_error_cRIO * np.random.uniform(low=-1, high=1, size=total_samples) * range_cRIO)
    voltage_ds_cRIO = ((1 + gain_error_cRIO * np.random.uniform(low=-1, high=1, size=total_samples)) * voltage_derivative_sensor) + (offset_error_cRIO * np.random.uniform(low=-1, high=1, size=total_samples) * range_cRIO)
    
    del voltage_magnet, voltage_derivative_sensor
    
    voltage_magnet_cRIO = np.round(voltage_magnet_cRIO / ADC_resolution) * ADC_resolution
    voltage_ds_cRIO = np.round(voltage_ds_cRIO / ADC_resolution) * ADC_resolution
    
    return voltage_magnet_cRIO, voltage_ds_cRIO


def current_reading(current, total_samples, k_DCCT, gain_error_cRIO, offset_error_cRIO, range_cRIO, ADC_resolution): 
   
    I_measured = current * k_DCCT
    I_measured = ((1 + gain_error_cRIO * np.random.uniform(low=-1, high=1, size=total_samples)) * I_measured) + (offset_error_cRIO * np.random.uniform(low=-1, high=1, size=total_samples) * range_cRIO) 
    I_measured = np.round(I_measured / ADC_resolution) * ADC_resolution
    
    del current
    
    return I_measured

def power_estimation(voltage_magnet, voltage_derivative_sensor, current, current_measured, number_samples, k_DCCT, attenuation_factor_ISOBLOCK ):

    magnet_power_no_comp = (voltage_magnet * attenuation_factor_ISOBLOCK) * (current_measured / k_DCCT)
    magnet_power_no_comp_mean = np.sum(magnet_power_no_comp) / number_samples

    magnet_power_comp = ((voltage_magnet - voltage_derivative_sensor) * attenuation_factor_ISOBLOCK) * (current_measured / k_DCCT)
    magnet_power_comp_mean = np.sum(magnet_power_comp) / number_samples

    del magnet_power_no_comp, magnet_power_comp
        
    return magnet_power_comp_mean, magnet_power_no_comp_mean

def simulate(I_min, I_max, I_ramp_rate, freq, L_magnet, kds, kds_tol, dt, P_ac, N, period, monte_carlo_iterations, attenuation_factor_ISOBLOCK, gain_error_ISOBLOCK, offset_error_ISOBLOCK, gain_error_cRIO, offset_error_cRIO, range_cRIO, ADC_resolution, k_DCCT):
    total_time = N * period
    t = np.arange(0, total_time, dt, dtype=np.float32)
    total_samples = len(t)
    
    I = generate_current_ramp(I_min, I_max, I_ramp_rate, freq, t)
    R_magnet = P_ac / np.mean(I[0:int(period / dt)]**2)
    number_integration_samples = round(N * period / dt)
    del t 
    voltage_magnet, voltage_ds_measured = voltage_magnet_and_voltage_derivative_sensor(I, L_magnet, R_magnet, kds, kds_tol, dt)
    voltage_magnet, voltage_ds_measured = voltages_post_ISOBLOCK(voltage_magnet, voltage_ds_measured, total_samples, attenuation_factor_ISOBLOCK, gain_error_ISOBLOCK, offset_error_ISOBLOCK) 
    voltage_magnet, voltage_ds_measured = voltages_in_cRIO(voltage_magnet, voltage_ds_measured, total_samples, gain_error_cRIO, offset_error_cRIO, range_cRIO, ADC_resolution)
    I_measured = current_reading(I, total_samples, k_DCCT, gain_error_cRIO, offset_error_cRIO, range_cRIO, ADC_resolution)
    
    magnet_power_comp, magnet_power_no_comp = power_estimation(voltage_magnet, voltage_ds_measured, I, I_measured, total_samples, k_DCCT, attenuation_factor_ISOBLOCK)
    
    return magnet_power_comp, magnet_power_no_comp

def statistics_calculation(magnet_power_comp_mean, magnet_power_no_comp_mean):
    mean_power_comp = np.mean(magnet_power_comp_mean, axis=0)
    mean_power_no_comp = np.mean(magnet_power_no_comp_mean, axis=0)

    std_power_comp = np.std(magnet_power_comp_mean, axis=0)
    std_power_no_comp = np.std(magnet_power_no_comp_mean, axis=0)
    
    del magnet_power_no_comp_mean, magnet_power_comp_mean
    
    return mean_power_comp, mean_power_no_comp, std_power_comp, std_power_no_comp


def plot_power_cycles(cycle_number_array, mean_power_comp, mean_power_no_comp, std_power_comp, std_power_no_comp): 
    P_ac = 37.6  # Power losses in W
    plt.figure(figsize=(12, 6))
    plt.errorbar(cycle_number_array, mean_power_comp, yerr=std_power_comp, fmt='o', capsize=5, label='Compensated Power losses')
    plt.errorbar(cycle_number_array, mean_power_no_comp, yerr=std_power_no_comp, fmt='o', capsize=5, label='Uncompensated Power losses')
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

    
def write_results_to_file(file_path, magnet_power_no_comp_mean, magnet_power_comp_mean, mean_power_comp, std_power_comp, mean_power_no_comp, std_power_no_comp):
    if os.path.exists(file_path):
        os.remove(file_path)
    
    with open(file_path, 'w') as file:
        # Write the non-compensated mean power
        file.write("Mean power (not compensated)(acquisition):\n")
        np.savetxt(file, magnet_power_no_comp_mean, delimiter=",")
        
        # Write the compensated mean power
        file.write("\nMean power (compensated)(acquisition):\n")
        np.savetxt(file, magnet_power_comp_mean, delimiter=",")
        
        # Write the final results for compensated power
        file.write("\nFinal results for compensated power:\n")
        file.write(f"Mean power for compensated: {mean_power_comp}\n")
        file.write(f"Std of power for compensated: {std_power_comp}\n")
        
        # Write the final results for non-compensated power
        file.write("\nFinal results for non-compensated power:\n")
        file.write(f"Mean power for non-compensated: {mean_power_no_comp}\n")
        file.write(f"Std of power for non-compensated: {std_power_no_comp}\n")
