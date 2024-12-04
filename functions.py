import numpy as np
import matplotlib.pyplot as plt
import os
###PARAMETERS###
# Current cycle parameters
I_min = np.float32(0)  # valore minimo in A
I_max = np.float32(3e3)  # valore massimo in A
I_ramp_rate = np.float32(2e3)  # velocità del ramp in A/s
time_plateau = np.float32(3.5)  # tempo di plateau in s
I_average = (I_max + I_min) / 2  # corrente media in A
time_transient = (I_max - I_min) / I_ramp_rate  # durata del ramp in s
period = np.float32(2 * time_plateau + 2 * time_transient)  # periodo del ciclo in s
freq = np.float32(1 / period)  # frequenza del ciclo in Hz

#DISCORAP parameters
P_ac = np.float32(37.6)  # perdite di potenza in W
L_magnet = np.float32(22.5e-3)  # induttanza magnete in H

# Derivative sensor parameters
mu0 = np.float32(np.pi * 4e-7)  # permeabilità del vuoto in H/m
mu_r = np.float32(50000)  # permeabilità relativa Vitroperm500
mu = mu0 * mu_r  # permeabilità del materiale in H/m
Ac = np.float32(18.0e-4)  # sezione trasversale in m^2
n2 = np.float32(20000)  # spire della bobina
la = np.float32(1.0e-3)  # lunghezza gap aria in m
lc = np.float32(0.3)  # percorso magnetico in m
kds = mu * Ac * n2 / (lc + 2 * mu_r * la)
kds_stdv = np.float32(0.0063)  # deviazione standard relativa

# Sampling parameters
dt = np.float32(1 / 1e3)  # passo di campionamento in s


### FUNCTIONS INCLUDED IN THE MAIN PARALLELIZED SCRIPT ###


def sinusoidal_transition(x, x0, x1, y0, y1):
    """Genera una transizione sinusoidale tra due punti (x0, y0) e (x1, y1)."""
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

def voltage_magnet_and_voltage_derivative_sensor(current, inductance, resistance, proportionality_factor, proportionality_factor_stdv, time_interval):
    # Derivata della corrente
    dI_dt = np.concatenate((np.diff(current), [0])).astype(np.float32) / time_interval

    # Tensione del magnete nel dominio del tempo
    voltage_magnet = (resistance * current + inductance * dI_dt).astype(np.float32)

    # Calcolo delle perdite di potenza
    voltage_ds_measured = (proportionality_factor * (1 + proportionality_factor_stdv * np.random.randn(len(current)).astype(np.float32))) * dI_dt
   
    del dI_dt
    
    return voltage_magnet, voltage_ds_measured

def voltages_post_ISOBLOCK(voltage_magnet,voltage_derivative_sensor,total_samples): 
    attenuation_factor_ISOBLOCK = np.float32(10)
    gain_error_ISOBLOCK = np.float32(0.2 / 100 + 0.005)
    offset_error_ISOBLOCK = np.float32(500e-6)
    
    voltage_magnet_ISO = (((1 + gain_error_ISOBLOCK * np.random.randn(total_samples).astype(np.float32)) * voltage_magnet) / attenuation_factor_ISOBLOCK) + offset_error_ISOBLOCK * np.random.randn(total_samples).astype(np.float32)
    voltage_ds_measured_ISO = (((1 + gain_error_ISOBLOCK * np.random.randn(total_samples).astype(np.float32)) * voltage_derivative_sensor) / attenuation_factor_ISOBLOCK) + offset_error_ISOBLOCK * np.random.randn(total_samples).astype(np.float32)

    del voltage_magnet, voltage_derivative_sensor
    
    return voltage_magnet_ISO, voltage_ds_measured_ISO

def voltages_in_cRIO(voltage_magnet, voltage_derivative_sensor, total_samples): 
    gain_error_cRIO = np.float32(0.03 / 100)
    offset_error_cRIO = np.float32(0.008 / 100)
    range_cRIO = np.float32(10.52)
    ADC_resolution = np.float32(1.2e-6)
    
    voltage_magnet_measured_cRIO = ((1 + gain_error_cRIO * np.random.randn(total_samples).astype(np.float32)) * voltage_magnet) + (offset_error_cRIO * np.random.randn(total_samples).astype(np.float32) * range_cRIO)
    voltage_ds_measured_cRIO = ((1 + gain_error_cRIO * np.random.randn(total_samples).astype(np.float32)) * voltage_derivative_sensor) + (offset_error_cRIO * np.random.randn(total_samples).astype(np.float32) * range_cRIO)
    
    del voltage_magnet, voltage_derivative_sensor
    
    voltage_magnet_measured_cRIO = np.round(voltage_magnet_measured_cRIO / ADC_resolution) * ADC_resolution
    voltage_ds_measured_cRIO = np.round(voltage_ds_measured_cRIO / ADC_resolution) * ADC_resolution
    
    return voltage_magnet_measured_cRIO, voltage_ds_measured_cRIO


def current_reading(current, total_samples): 
    k_DCCT = np.float32(10/np.max(current))
    gain_error_cRIO = np.float32(0.03 / 100)
    offset_error_cRIO = np.float32(0.008 / 100)
    range_cRIO = np.float32(10.52)
    ADC_resolution = np.float32(1.2e-6)
    I_measured = (1 + gain_error_cRIO * np.random.randn(total_samples).astype(np.float32)) * current + (offset_error_cRIO * np.random.randn(total_samples).astype(np.float32) * range_cRIO / k_DCCT)
    I_measured = np.round(I_measured / ADC_resolution) * ADC_resolution
    
    del current
    
    return I_measured

def power_estimation(voltage_magnet, voltage_derivative_sensor, current, number_samples):
    attenuation_factor_ISOBLOCK = np.float32(10)
    magnet_power_no_comp = voltage_magnet * current* attenuation_factor_ISOBLOCK
    magnet_power_no_comp_mean = np.sum(magnet_power_no_comp) / number_samples

    magnet_power_comp = (voltage_magnet - voltage_derivative_sensor) * current* attenuation_factor_ISOBLOCK
    magnet_power_comp_mean = np.sum(magnet_power_comp) / number_samples

    del magnet_power_no_comp, magnet_power_comp
        
    return magnet_power_comp_mean, magnet_power_no_comp_mean

def simulate(N, monte_carlo_iterations):
    total_time = N * period
    t = np.arange(0, total_time, dt, dtype=np.float32)
    total_samples = len(t)
    
    I = generate_current_ramp(I_min, I_max, I_ramp_rate, freq, t).astype(np.float32)
    R_magnet = P_ac / np.mean(I[0:int(period/dt)]**2)
    number_integration_samples = round(N * period / dt)
    del t 
    voltage_magnet, voltage_ds_measured = voltage_magnet_and_voltage_derivative_sensor(I, L_magnet, R_magnet, kds, kds_stdv, dt)
    voltage_magnet_ISO, voltage_ds_measured_ISO = voltages_post_ISOBLOCK(voltage_magnet, voltage_ds_measured, total_samples) 
    voltage_magnet_measured_cRIO, voltage_ds_measured_cRIO = voltages_in_cRIO (voltage_magnet_ISO,voltage_ds_measured_ISO, total_samples)
    I_measured = current_reading(I, total_samples)
    
    magnet_power_comp, magnet_power_no_comp = power_estimation(voltage_magnet_measured_cRIO, voltage_ds_measured_cRIO, I_measured, total_samples)
    
    return magnet_power_comp, magnet_power_no_comp

def statistics_calculation(magnet_power_comp_mean, magnet_power_no_comp_mean):
    mean_power_comp = np.mean(magnet_power_comp_mean, axis=0)
    mean_power_no_comp = np.mean(magnet_power_no_comp_mean, axis=0)

    std_power_comp = np.std(magnet_power_comp_mean,axis=0)
    std_power_no_comp = np.std(magnet_power_no_comp_mean, axis=0)
    
    del magnet_power_no_comp_mean, magnet_power_comp_mean
    
    return mean_power_comp, mean_power_no_comp, std_power_comp, std_power_no_comp


def plot_power_cycles(cycle_number_array, mean_power_comp, mean_power_no_comp, std_power_comp, std_power_no_comp): 
    P_ac = np.float32(37.6)  # perdite di potenza in W
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
    
    
def write_results_to_file(file_path, magnet_power_no_comp_mean, magnet_power_comp_mean,mean_power_comp,std_power_comp,mean_power_no_comp,std_power_no_comp):
    if os.path.exists(file_path):
        os.remove(file_path)
    
    with open(file_path, 'w') as file:
        # Scrivi la potenza media non compensata
        file.write("Mean power (not compensated)(acquisition cycle number in the columns and montecarlo iteration rows):\n")
        np.savetxt(file, magnet_power_no_comp_mean, fmt='%.6f')
        # Lascia una riga vuota
        file.write("\n")
        file.write("Mean power (compensated) (number of cycles of acquisition in the columns and montecarlo iteration rows):\n")
        np.savetxt(file, magnet_power_comp_mean, fmt='%.6f')
        file.write("\n")
        file.write("########## STATISTICS ##########:\n \n")
        file.write("Compensated Mean (in rows the statistic of each acquisition cycle number is given):\n")
        np.savetxt(file, mean_power_comp, fmt='%.6f')
        file.write("\n \n")
        file.write("Not Compensated Mean:\n")
        np.savetxt(file, mean_power_no_comp, fmt='%.6f')
        file.write("\n \n")
        file.write("Compensated StdDv:\n")
        np.savetxt(file, std_power_comp, fmt='%.6f')
        file.write("\n \n")
        file.write("Not Compensated StdDv:\n")
        np.savetxt(file, std_power_no_comp, fmt='%.6f')
        
def plot_current_cycle(I_min, I_max, I_ramp_rate, time_plateau, dt, N_max):
    # Calcola i parametri del ciclo
    time_transient = (I_max - I_min) / I_ramp_rate
    period = 2 * time_plateau + 2 * time_transient  # Periodo del ciclo in s
    total_time = N_max * period  # Tempo totale per N_max cicli
    t = np.arange(0, total_time, dt, dtype=np.float32)  # Array dei tempi

    # Genera il ciclo di corrente
    I = generate_current_ramp(I_min, I_max, I_ramp_rate, 1/period, t).astype(np.float32)

    # Plot del ciclo di corrente
    plt.figure(figsize=(10, 6))
    plt.plot(t, I, label=f'Current Cycle for N={N_max} acquisition cycles')
    plt.xlabel('Time (s)')
    plt.ylabel('Current (A)')
    plt.title(f'Current Cycle for N={N_max}')
    plt.grid(True)
    plt.savefig('current_cycle.svg', format='svg')
    plt.show()
    
def plot_power_distribution(magnet_power_mean, column_index, filename):
        # Estrai la prima colonna del vettore
    power_to_distribute = magnet_power_mean[:, column_index]
    # Crea l'istogramma
    plt.figure(figsize=(10, 6))
    plt.hist(power_to_distribute, edgecolor='black')
    plt.xlabel('Mean Power (Not Compensated)')
    plt.ylabel('Frequency')
    plt.title('Histogram of the Mean Power (Not Compensated) for N = 4')
    plt.grid(True)
    plt.savefig(filename, format='svg')
    plt.show()