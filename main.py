import numpy as np
import matplotlib.pyplot as plt
import time
import gc
from functions import generate_current_ramp, voltage_magnet_and_voltage_derivative_sensor, voltages_post_ISOBLOCK, voltages_in_cRIO, current_reading, power_estimation, simulate, statistics_calculation, plot_power_cycles, write_results_to_file, plot_current_cycle, plot_power_distribution
from joblib import Parallel, delayed

start_time = time.time()

### PARAMETERS ###
file_name = 'results_simulation.txt'  # Output file name
monte_carlo_iterations_array = np.arange(0, 2)  # Array for Monte Carlo iterations (here it runs 2 iterations)

################################
### Current cycle parameters ###
################################
I_min = 0  # Minimum current in A
I_max = 3e3  # Maximum current in A
I_ramp_rate = 2e3  # Ramp rate in A/s
time_plateau = 3.5  # Plateau time in s
I_average = (I_max + I_min) / 2  # Average current in A
time_transient = (I_max - I_min) / I_ramp_rate  # Ramp duration in s
period = 2 * time_plateau + 2 * time_transient  # Cycle period in s
freq = 1 / period  # Frequency in Hz

###############################
### Magnet model parameters ###
###############################
P_ac = 37.6  # Power losses in W
L_magnet = 22.5e-3  # Magnet inductance in H

####################################
### Derivative sensor parameters ###
####################################

mu0 = np.pi * 4e-7  # Permeability of free space in H/m
mu_r = 50000  # Relative permeability of Vitroperm500
mu = mu0 * mu_r  # Material permeability in H/m
Ac = 18.0e-4  # Cross-sectional area in m^2
n2 = 20000  # Coil turns
la = 1.0e-3  # Air gap length in m
lc = 0.3  # Magnetic path length in m
kds = mu * Ac * n2 / (lc + 2 * mu_r * la)  # Derivative sensor, constant
kds_tol = 0.019  # Tolerance on kds
gain_kds = 0.63/100

#time_delay_ds = 250e-3

##################################
### Instrumentation parameters ###
##################################

dt = 1 / 1e3  # Sampling step in s
attenuation_factor_ISOBLOCK = 10  # Attenuation factor for ISOBLOCK
gain_error_ISOBLOCK = 0.2 / 100  # Gain error for ISOBLOCK in percentage
offset_error_ISOBLOCK = 500e-6  # Offset error for ISOBLOCK in V
time_delay_ISOBLOCK = np.array([2e-6,6e-6])

gain_error_cRIO = 0.03 / 100  # Gain error for cRIO in percentage
offset_error_cRIO = 0.008 / 100  # Offset error for cRIO in V
range_cRIO = 10.52  # cRIO range
time_delay_cRIO = 200e-6 #cRIO time delay

ADC_resolution = 1.2e-6  # ADC resolution in V
k_DCCT = 1/750  # DCCT constant
gain_error_DCCT = 37/1e6 # DCCT gain error
offset_DCCT = 70e-6
time_delay_DCCT = 3e-6 #DCCT time delay

time_delay_current = (time_delay_DCCT*np.random.randn()) + np.random.uniform(low = time_delay_cRIO-0.01*time_delay_cRIO, high = time_delay_cRIO+0.01*time_delay_cRIO)
time_delay_voltage = np.random.uniform(low = time_delay_cRIO-0.01*time_delay_cRIO,  high = time_delay_cRIO+0.01*time_delay_cRIO) + np.random.uniform(low = time_delay_ISOBLOCK[0], high = time_delay_ISOBLOCK[1])

# N_values array for different cycles in the simulation
#N_values = np.array([1, 2,])#3, 4, 5, 10, 15, 20, 25, 30, 40, 50])  
N = 3

def simulate(I_min, I_max, I_ramp_rate, time_offset_current, time_offset_voltages, time_plateau, L_magnet, kds, gain_ds, offset_ds, dt, P_ac, N, 
             period, attenuation_factor_ISOBLOCK, gain_error_ISOBLOCK, offset_error_ISOBLOCK, gain_error_cRIO, offset_error_cRIO, range_cRIO, ADC_resolution, k_DCCT, gain_error_DCCT, offset_DCCT):
    
    total_time = N * period
    t = np.arange(0, total_time, dt)
    
    I_voltage = generate_current_ramp(I_min, I_max, I_ramp_rate, time_plateau, t, time_offset_voltages)
    I_current = generate_current_ramp(I_min, I_max, I_ramp_rate, time_plateau, t, time_offset_current)
    R_magnet = P_ac / np.mean(I_voltage[0:int(period / dt)]**2)
    print(R_magnet)
 
    voltage_magnet, voltage_ds_measured = voltage_magnet_and_voltage_derivative_sensor(I_voltage, L_magnet, R_magnet, kds, gain_ds, offset_ds, dt)
    voltage_magnet, voltage_ds_measured = voltages_post_ISOBLOCK(voltage_magnet, voltage_ds_measured, attenuation_factor_ISOBLOCK, gain_error_ISOBLOCK, offset_error_ISOBLOCK) 
    voltage_magnet, voltage_ds_measured = voltages_in_cRIO(voltage_magnet, voltage_ds_measured, gain_error_cRIO, offset_error_cRIO, range_cRIO, ADC_resolution)
    I_measured = current_reading(I_current, k_DCCT, gain_error_DCCT, offset_DCCT, gain_error_cRIO, offset_error_cRIO, range_cRIO, ADC_resolution)
    
    magnet_power_comp, magnet_power_no_comp = power_estimation(voltage_magnet, voltage_ds_measured, I_measured, k_DCCT, attenuation_factor_ISOBLOCK)
    
    return t, magnet_power_comp, magnet_power_no_comp




t, magnet_power_comp, magnet_power_no_comp = simulate(I_min, I_max, I_ramp_rate, time_delay_current, time_delay_voltage, time_plateau, L_magnet, kds, gain_kds, kds_tol, dt, P_ac, N, 
             period, attenuation_factor_ISOBLOCK, gain_error_ISOBLOCK, offset_error_ISOBLOCK, gain_error_cRIO, offset_error_cRIO, range_cRIO, ADC_resolution, k_DCCT, gain_error_DCCT, offset_DCCT)

print(len(t))
print(len(magnet_power_comp))
#print(len(magnet_power_comp))

plt.plot(t, magnet_power_comp)
plt.show()

# Arrays to store the results of the simulations
#magnet_power_comp_mean = np.zeros((len(monte_carlo_iterations_array), len(N_values)))
#magnet_power_no_comp_mean = np.zeros((len(monte_carlo_iterations_array), len(N_values)))

# Main simulation loop
# for j, monte_carlo_iterations in enumerate(monte_carlo_iterations_array):
#     # Parallel execution of the simulation for different values of N
#     results = Parallel(n_jobs=-1)(delayed(simulate)(
#         I_min, I_max, I_ramp_rate, time_delay_current, time_delay_voltage, time_plateau, L_magnet, kds, gain_kds, kds_tol, dt, P_ac, N, period, 
#         monte_carlo_iterations, attenuation_factor_ISOBLOCK, gain_error_ISOBLOCK, offset_error_ISOBLOCK, 
#         gain_error_cRIO, offset_error_cRIO, range_cRIO, ADC_resolution, k_DCCT, gain_error_DCCT, offset_DCCT) for N in N_values)
    
#     print("Monte Carlo iteration: ", j + 1)
#     magnet_power_comp_mean[j, :], magnet_power_no_comp_mean[j, :] = zip(*results)
    
    
    
    
    # Perform garbage collection to free memory
gc.collect()

# Calculate statistics (mean and standard deviation)
# mean_power_comp, mean_power_no_comp, std_power_comp, std_power_no_comp = statistics_calculation(
#     magnet_power_comp_mean, magnet_power_no_comp_mean)

# Perform garbage collection again to free memory
gc.collect()
print("Simulation completed in {:.2f} seconds.".format(time.time() - start_time))

# Write the results to the output file
#write_results_to_file(file_name, magnet_power_no_comp_mean, magnet_power_comp_mean)#, 
#                      mean_power_comp, std_power_comp, mean_power_no_comp, std_power_no_comp)

# Plotting the results
#plot_current_cycle(I_min, I_max, I_ramp_rate, time_plateau, dt, N_max=2)
#plot_power_cycles(N_values, mean_power_comp, mean_power_no_comp, std_power_comp, std_power_no_comp)
#plot_power_distribution(magnet_power_no_comp_mean, column_index=4, filename='distribution_power_mean_no_comp.svg')
#plot_power_distribution(magnet_power_comp_mean, column_index=4, filename='distribution_power_mean_comp.svg')
