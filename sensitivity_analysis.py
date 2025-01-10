#################
### LIBRARIES ###
#################
import numpy as np
import matplotlib.pyplot as plt
import time
import gc
from functions import generate_current_ramp, voltage_magnet_and_voltage_derivative_sensor, voltages_post_ISOBLOCK, voltages_in_cRIO, current_reading, power_estimation, simulate, statistics_calculation, plot_power_cycles, write_results_to_file, plot_current_cycle, plot_power_distribution, write_parameters_file, plot_stdv_sensitivity_analysis
start_time = time.time()

#############################
### SIMULATION PARAMETERS ###
#############################
file_name = 'sensitivity_analysis_time_delay_cRIO.txt'  # Output file name
file_parameters = 'parameters_file_time_delay_cRIO'  # Output file name
N_tot_cycles = 100 # N_values array for different cycles in the simulation
monte_carlo_iterations = 1000 # Number of Monte Carlo iterations to simulate
cycles = np.array([1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]) # Number of cycles on which calculate statistics
num_cycles = len(cycles)

################################
### Current cycle parameters ###  
################################
I_min = 0  # Minimum current in A
I_max = 3e3  # Maximum current in A
I_ramp_rate = 2e3  # Ramp rate in A/s
time_plateau = 3.5  # Plateau time in s
I_average = (I_max + I_min) / 2  # Average current in A
time_ramp = (I_max - I_min) / I_ramp_rate  # Ramp duration in s
period = 2 * time_plateau + 2 * time_ramp  # Cycle period in s
freq = 1 / period  # Frequency in Hz
total_time = N_tot_cycles * period

###############################
### Magnet model parameters ###
###############################
P_ac = 37.6  # Power losses in W
L_magnet = 22.5e-3  # Magnet inductance in H
R_magnet = 9.284e-6
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
kds_comp_off = 0.0019  # Tolerance on kds
gain_kds = 0.63/100

##################################
### Instrumentation parameters ###
##################################
dt = 1 / 1e3  # Sampling step in s
attenuation_factor_ISOBLOCK = 10  # Attenuation factor for ISOBLOCK
range_cRIO = 10.52  # cRIO range
ADC_resolution = 1.2e-6  # ADC resolution in V
k_DCCT = 1/750  # DCCT constant

#######################################
### Sensitivity analysis parameters ###
#######################################

#time_delay_ISOBLOCK = np.array([0,5.7e-6]) #ISOBLOCK time delay, uniform BUT WITH MEAN DIFFERENT THAN 0 
time_delay_ISOBLOCK =2.85e-6
#time_delay_DCCT = np.array([1e-6,5e-6]) #DCCT time delay, UNIFORM BUT WITH MEAN DIFFERENT THAN 0 
time_delay_DCCT = 3e-6
time_delay_cRIO = 200e-6              # cRIO time delay, UNIFORM, ON INVESTIGATION
time_delay_corr = 250e-4    *0          # correction time delay, UNIFORM
offset_kds_corr = 0.1*kds   *0          # correction offset, UNIFORM
gain_error_DCCT = 37/1e6                # DCCT gain error, GAUSSIAN
offset_DCCT = 70e-6         *0          # Offset DCCT, UNIFORM
gain_error_ISOBLOCK = 0.2 / 100         # Gain error for ISOBLOCK in percentage, Gaussian
offset_error_ISOBLOCK = 500e-6    *0    # Offset error for ISOBLOCK in V, UNIFORM
gain_error_cRIO = 0.03 / 100            # Gain error for cRIO in percentage, Gaussian
offset_error_cRIO = 0.008 / 100   *0    # Offset error for cRIO in V, UNIFORM


sensitivity_parameters = 11

#Initialize vectors to save variables
all_magnet_power_no_comp_avg = np.zeros((monte_carlo_iterations, num_cycles))
all_magnet_power_comp_avg = np.zeros((monte_carlo_iterations, num_cycles))
all_magnet_power_corr_avg = np.zeros((monte_carlo_iterations, num_cycles))

sensitivity_parameters_tot  = np.zeros((monte_carlo_iterations, sensitivity_parameters))

t = np.arange(0, total_time, dt)

for i in range(monte_carlo_iterations):
    # Main simulation loop
    ISOBLOCKdelay, DCCTdelay, time_delay_cRIO, time_delay_correction, offset_correction, gain_error_DCCT, offset_DCCT, gain_error_ISOBLOCK, offset_error_ISOBLOCK, gain_error_cRIO, offset_error_cRIO, magnet_power_no_comp_avg, magnet_power_comp_avg, magnet_power_corr_avg = simulate(
        I_min, I_max, time_delay_DCCT, time_delay_cRIO, time_delay_ISOBLOCK, time_delay_corr, time_plateau, time_ramp, t,  L_magnet, R_magnet, kds, gain_kds, kds_comp_off, offset_kds_corr,
        dt,  period, attenuation_factor_ISOBLOCK, gain_error_ISOBLOCK, offset_error_ISOBLOCK,
        gain_error_cRIO, offset_error_cRIO, range_cRIO, ADC_resolution, k_DCCT, gain_error_DCCT, offset_DCCT, cycles
    )

    all_magnet_power_no_comp_avg[i, :] = magnet_power_no_comp_avg
    all_magnet_power_comp_avg[i, :] = magnet_power_comp_avg
    all_magnet_power_corr_avg[i, :] = magnet_power_corr_avg
    sensitivity_parameters_tot[i, :] = [ISOBLOCKdelay,  DCCTdelay, time_delay_cRIO, time_delay_correction, offset_correction, gain_error_DCCT, offset_DCCT, gain_error_ISOBLOCK, offset_error_ISOBLOCK, gain_error_cRIO, offset_error_cRIO]
        
    gc.collect()
    print("Monte Carlo iteration n. ", i+1)

mean_power_no_comp, mean_power_comp, mean_power_corr, std_power_no_comp, std_power_comp, std_power_corr = statistics_calculation(all_magnet_power_no_comp_avg, all_magnet_power_comp_avg, all_magnet_power_corr_avg)

# Writing the results on the file

write_results_to_file(file_name, all_magnet_power_no_comp_avg, all_magnet_power_comp_avg, all_magnet_power_corr_avg, mean_power_comp, std_power_comp, mean_power_no_comp, std_power_no_comp, mean_power_corr, std_power_corr)
write_parameters_file(file_parameters, sensitivity_parameters_tot)
gc.collect()

print("Simulation completed in {:.2f} seconds.".format(time.time() - start_time))

#############
### PLOTS ###
#############

plot_power_cycles(cycles, mean_power_no_comp, mean_power_comp, mean_power_corr, std_power_no_comp, std_power_comp,  std_power_corr)
plot_stdv_sensitivity_analysis(cycles,std_power_corr)
plot_power_distribution(all_magnet_power_no_comp_avg, column_index=4, filename="distr_no_comp")
plot_power_distribution(all_magnet_power_comp_avg, column_index=4, filename="distr_comp")
plot_power_distribution(all_magnet_power_corr_avg, column_index=4, filename="distr_corr")