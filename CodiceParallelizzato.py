import numpy as np
import matplotlib.pyplot as plt
import time
import gc
from functions import generate_current_ramp, voltage_magnet_and_voltage_derivative_sensor, voltages_post_ISOBLOCK, voltages_in_cRIO, current_reading, power_estimation, simulate, statistics_calculation, plot_power_cycles, write_results_to_file,plot_current_cycle,plot_power_distribution
from joblib import Parallel, delayed

start_time = time.time()

### PARAMETERS ###
file_name = 'results_simulation.txt'
monte_carlo_iterations_array = np.arange(0,10)
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

N_values = np.array([1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50], dtype=np.float32)

magnet_power_comp_mean = np.zeros((len(monte_carlo_iterations_array), len(N_values)))
magnet_power_no_comp_mean = np.zeros((len(monte_carlo_iterations_array), len(N_values)))

for j, monte_carlo_iterations in enumerate(monte_carlo_iterations_array):
    results = Parallel(n_jobs=-1)(delayed(simulate)(N, monte_carlo_iterations) for N in N_values)
    print("Montecarlo index: ", j+1)
    magnet_power_comp_mean[j, :], magnet_power_no_comp_mean[j, :] = zip(*results)
    gc.collect()
mean_power_comp, mean_power_no_comp, std_power_comp, std_power_no_comp = statistics_calculation(magnet_power_comp_mean, magnet_power_no_comp_mean)

gc.collect()
print("Simulation completed in {:.2f} seconds.".format(time.time() - start_time))

write_results_to_file(file_name,magnet_power_no_comp_mean, magnet_power_comp_mean,mean_power_comp,std_power_comp,mean_power_no_comp,std_power_no_comp)

plot_current_cycle(I_min, I_max, I_ramp_rate, time_plateau, dt, N_max=4)

plot_power_cycles(N_values, mean_power_comp, mean_power_no_comp, std_power_comp, std_power_no_comp)
plot_power_distribution(magnet_power_no_comp_mean, column_index=4, filename='distribution_power_mean_no_comp.svg')
plot_power_distribution(magnet_power_comp_mean, column_index=4, filename='distribution_power_mean_comp.svg')
