import numpy as np
import matplotlib.pyplot as plt
from functions import generate_current_ramp, voltage_magnet_and_voltage_derivative_sensor, voltages_post_ISOBLOCK, voltages_in_cRIO, current_reading, power_estimation, simulate, statistics_calculation, plot_power_cycles, write_results_to_file, plot_current_cycle, plot_power_distribution

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
time_delay_ds = 250e-3

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
k_DCCT = 1/750 # DCCT constant
gain_error_DCCT = 37/1e6 # DCCT gain error
offset_DCCT = 70e-6
time_delay_DCCT = 3e-6 #DCCT time delay

time_delay_current = (time_delay_DCCT*np.random.randn()) + np.random.uniform(low = time_delay_cRIO-0.01*time_delay_cRIO, high = time_delay_cRIO+0.01*time_delay_cRIO)
time_delay_voltage = np.random.uniform(low = time_delay_cRIO-0.01*time_delay_cRIO,  high = time_delay_cRIO+0.01*time_delay_cRIO) + np.random.uniform(low = time_delay_ISOBLOCK[0], high = time_delay_ISOBLOCK[1])
# N_values array for different cycles in the simulation
N = 1
#N_values = np.array([1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50])  

total_time = N*period

t = np.arange(0, total_time, dt, dtype=np.float32)
total_samples = len(t)
time_delay_ISOBLOCK = np.array([2e-6,6e-6])

time_delay_voltage = np.random.uniform(low = time_delay_cRIO-0.01*time_delay_cRIO,  high = time_delay_cRIO+0.01*time_delay_cRIO) + np.random.uniform(low = time_delay_ISOBLOCK[0], high = time_delay_ISOBLOCK[1])
time_delay_current = (time_delay_DCCT*np.random.randn()) + np.random.uniform(low = time_delay_cRIO-0.01*time_delay_cRIO, high = time_delay_cRIO+0.01*time_delay_cRIO)

########################
#######VARIABLES########
########################

I_read = generate_current_ramp(I_min, I_max, I_ramp_rate, time_plateau, t, time_delay_current)
I_measured = current_reading(I_read, k_DCCT, gain_error_DCCT, offset_DCCT, gain_error_cRIO, offset_error_cRIO, range_cRIO, ADC_resolution)
I_voltage = generate_current_ramp(I_min, I_max, I_ramp_rate, time_plateau, t, time_delay_voltage)

R_magnet = 9.28e-6
voltage_magnet, voltage_ds_measured = voltage_magnet_and_voltage_derivative_sensor(I_voltage, L_magnet, R_magnet, kds, gain_kds, kds_tol, dt)
dI_dt = np.concatenate((np.diff(I_voltage), [0])) / dt

# Generazione della rampa

# Plot
#plt.plot(t, I_read)
plt.plot(t, I_measured, "-o")
#plt.plot(t,I_voltage,"r")
#plt.plot(t, voltage_magnet, "b")
#plt.plot(t, voltage_ds_measured, "r-o")
#plt.plot(t,dI_dt)
plt.xlabel('Tempo (s)')
plt.ylabel('Corrente (A)')
plt.title('Corrente a rampa sinusoidale')
plt.grid()
plt.show()