import numpy as np
import time
import gc
from functions import generate_current_ramp, calculate_instant_power, save_results
from functions import cust_plot_current, cust_plot_power, cust_hist_power


## NOTES ON FUTURE ENHANCEMENTS
# 1) in composing the time delays, I'm not sure we have to take different ones on different streams if it is the same contribution, hence the currently implemented one is a worst case scenario
# 2) we could define a variable with the inverse of the current k_DCCT, this would reduce the number of division, but it is kept as it is for compatibility with previous versions of the code
# 3) the names "gain_*" and "offset_*" could be changed into "u_*" and "D_*" respectively (just to shorten variable names)
# 4) make the "custom plot functions" less hardcoded...


## WARNING
# THE CURRENT DELAY FOR THE CORRECTION CASE, COULD BE UNREALISTICALLY TOO LOW



#%% INPUTS FOR THE SIMULATION (DISCORAP case study)

## Monte Carlo simulation 
# parameters
MC_iterations = 10                                                           # number of Monte Carlo iterations
cycles = np.array([1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100])     # number of cycles of interest

Nmax_cycles = np.max(cycles)                                                    # N_values array for different cycles in the simulation
num_cycles = len(cycles)

# name of the file with results
file_name = "results_MC" + str(int(MC_iterations/1000)) + "k_cycles" + str(Nmax_cycles) + ".txt"


## supply current 
# parameters
I_min = 0                                                                       # minimum current in A
I_max = 3e3                                                                     # maximum current in A
I_ramp_rate = 2e3                                                               # ramp rate in A/s
time_plateau = 3.5                                                              # plateau time in s

time_ramp = (I_max - I_min) / I_ramp_rate                                       # ramp duration in s
period = 2 * time_plateau + 2 * time_ramp                                       # period in s
freq = 1 / period                                                               # frequency in Hz

# single current cycle to calculate Irms (squared)
f_samp = 50e3                                                                   # (local) sample rate in Sa/s
t = np.arange(0, period, 1/f_samp, dtype=np.float32)                            # time instants
I = generate_current_ramp(t, I_min, I_max, time_ramp, time_plateau, 0, period, f_samp)
Irms2 = np.mean(I**2)                                                           # mean square (Irms^2) of the current in A^2


## magnet model
P_ac = 37.6                                                                     # power losses in W
L_magnet = 22.5e-3                                                              # magnet inductance in H
R_magnet = P_ac/Irms2                                                           # magnet resistance, should be ~9.284e-6 ohm


## compensation/correction mechanism (derivative sensor or numerical derivative)
# material and geometrical parameters
mu0 = np.pi * 4e-7                                                              # permeability of free space in H/m
mu_r = 50000                                                                    # relative permeability of Vitroperm500
n2 = 20000                                                                      # coil turns
la = 1.0e-3                                                                     # air gap length in m
lc = 0.3                                                                        # magnetic path length in m
Ac = 18.0e-4                                                                    # cross-sectional area in m^2

kds = mu0 * mu_r  * Ac * n2 / (lc + 2 * mu_r * la)                              # derivative sensor constant



#%% UNCERTAINTY CONTRIBUTIONS AND FURTHER PARAMETERS FROM THE MEASUREMENT CHAIN

## DCCT DR500UX-10V/7500A
k_DCCT = 1/750                                                                  # conversion factor in V/A
gain_DCCT = 37e-6                                                               # relative uncertainty
offset_DCCT = 70e-6                                                             # absolute tolerance in V
delay_DCCT = np.array([1e-6, 5e-6])                                             # range of time delays in s

## isolation module ISOBLOCKv4
att_isolation = 10                                                              # attenuation factor of the isolation module in V/V
gain_isolation = 2e-3                                                           # relative uncertainty
offset_isolation = 500e-6                                                       # absolute tolerance in V
delay_isolation = np.array([0.0e-6, 5.7e-6])                                    # range of time delays in s

## acquisition module NI9239 in compactRIO 9049
range_acq = 10.52                                                               # acquisition module range in V
gain_acq = 3e-4                                                                 # relative uncertainty
offset_acq = 8e-5                                                               # relative tolerance
delay_acq = np.array([199e-6, 201e-6])                                          # range of time delays in s

## Compensation and correction part 
# sensor case
gain_comp = 63e-4                                                               # relative uncertainty on kds (obtained with a further MC simulation)
offset_comp = 19e-4                                                             # tolerance on kds (obtained with a further MC simulation)
delay_comp = np.array([0, 0])                                                   # time shift due to sensor (hypothesis: negligible)

# derivative case
gain_corr = 0e-4                                                                # relative uncertainty on kds in correction (hypothesis: null)
offset_corr = 0.1*kds                                                           # tolerance on kds (reasonably 10 %, as for the compensation)
delay_corr = np.array([24e-4, 26e-4])                                           # time shift due to filter (hypothesis: order 25 @ 10 kSa/s!!!)

## A/D conversion in NI9239: sampling and quantization
f_samp = 1e3                                                                    # sample frequency in Sa/s
n_bit = 24                                                                      # resolution in bit

dt = 1 / f_samp                                                                 # sample time in s
ADC_resolution = 2*range_acq/(np.power(2,n_bit)-1)                              # ADC resolution in V



#%% MONTE CARLO ITERATIONS

# start monitoring the simulation time
start_time = time.time()

# AC losses per each iteration and each number of cycles in the conditions of interest
AClosses_NOco = np.zeros((MC_iterations, num_cycles))                           # no compensation nor correction
AClosses_comp = np.zeros((MC_iterations, num_cycles))                           # compensation case
AClosses_corr = np.zeros((MC_iterations, num_cycles))                           # correction case

# time samples of interest (max number of cycles case)
t = np.arange(0, Nmax_cycles * period, dt)

## iterations
for i in range(MC_iterations):
    
    ### SIMULATION CORE --> implement the "del" at the end??
    ## path of magnet current: DCCT + acquisition module
    # add all the time delays here
    delay_current = np.random.uniform(low=delay_DCCT[0], high=delay_DCCT[1]) + np.random.uniform(low=delay_acq[0], high=delay_acq[1])
    current = generate_current_ramp(t, I_min, I_max, time_ramp, time_plateau, delay_current, period, f_samp)
    isamp = len(current)
    
    # adding the uncertainty of DCCT
    vDCCT = ((1+gain_DCCT*np.random.randn(isamp))*k_DCCT)*current + np.random.uniform(low=-offset_DCCT, high=+offset_DCCT, size=isamp)
    
    # adding the uncertainty of the acquisition module (it will be on CH3)
    CH3 = (1+gain_acq*np.random.randn(isamp))*vDCCT + range_acq*np.random.uniform(low=-offset_acq, high=+offset_acq, size=isamp)
    
    
    ## path of magnet voltage: isolation module + acquisition module
    # add the time delays of the voltage on the current entering the model
    delay_voltage = np.random.uniform(low=delay_isolation[0], high=delay_isolation[1]) + np.random.uniform(low=delay_acq[0], high=delay_acq[1])
    I = generate_current_ramp(t, I_min, I_max, time_ramp, time_plateau, delay_voltage, period, f_samp)
    
    # derivative of current
    dI_dt = np.concatenate((np.diff(I), [0]))/dt
    
    # magnet voltage in the time domain
    vMAGNET = (R_magnet * I + L_magnet * dI_dt)
    vsamp = len(vMAGNET)
    
    # adding the uncertainty of the isolation module
    vISOL = ((1+gain_isolation*np.random.randn(vsamp))/att_isolation)*vMAGNET + np.random.uniform(low=-offset_isolation, high=+offset_isolation, size=vsamp)
    
    # adding the uncertainty of the acquisition module (it will be on CH1)
    CH1 = (1+gain_acq*np.random.randn(vsamp))*vISOL + range_acq*np.random.uniform(low=-offset_acq, high=+offset_acq, size=vsamp)
    
    
    ## path of compensation voltage: derivative sensor + acquisition module
    # add the time delays of the current in a different way for the derivative sensor
    delay_sensor = np.random.uniform(low=delay_comp[0], high=delay_comp[1]) + np.random.uniform(low=delay_acq[0], high=delay_acq[1])
    Isens = generate_current_ramp(t, I_min, I_max, time_ramp, time_plateau, delay_sensor, period, f_samp)
    dIsens_dt = np.concatenate((np.diff(Isens), [0]))/dt
    isens = len(dIsens_dt)
    
    # apply the contribution of the derivative sensor
    vCOMP = ((1+gain_comp*np.random.randn(isens))*kds + np.random.uniform(low =-offset_comp, high=+offset_comp, size=isens))*dIsens_dt
    
    # apply contributions of the acquisition module (it will be on CH2)
    CH2 = (1+gain_acq*np.random.randn(isens))*vCOMP + range_acq*np.random.uniform(low=-offset_acq, high=+offset_acq, size=isens)
    
    
    ## path of correction voltage: using the measured current with delay of the filter
    # add the time delays of the current in a different way for the derivative sensor
    delay_Corr = np.random.uniform(low=delay_corr[0], high=delay_corr[1]) + delay_current
    Icorr = generate_current_ramp(t, I_min, I_max, time_ramp, time_plateau, delay_Corr, period, f_samp)
    dIcorr_dt = np.concatenate((np.diff(Icorr), [0]))/dt
    icorr = len(dIcorr_dt)
    
    # compute the correction voltage
    vCORR = ((1+gain_corr*np.random.randn(icorr))*kds + np.random.uniform(low =-offset_corr, high=+offset_corr, size=icorr))*dIcorr_dt
    
    # apply contributions of the acquisition module (it will be on CH2x)
    CH2x = (1+gain_acq*np.random.randn(icorr))*vCORR + range_acq*np.random.uniform(low=-offset_acq, high=+offset_acq, size=icorr)
    
        
    ## CALCULATION of instantaneous powers in different conditions (including quantization)
    inst_pow_NOco = calculate_instant_power(CH1*att_isolation,   0,     CH3/k_DCCT, ADC_resolution)
    inst_pow_comp = calculate_instant_power(CH1*att_isolation,   CH2,   CH3/k_DCCT, ADC_resolution)
    inst_pow_corr = calculate_instant_power(CH1*att_isolation,   CH2x,  CH3/k_DCCT, ADC_resolution)
    
    # cut at different number of cycles and retrieve the AC losses from instantaneous powers
    cut_idx = (cycles * period * f_samp).astype(int)
    
    AClosses_NOco[i, :] = np.array([np.mean(inst_pow_NOco[:end_idx]) for end_idx in cut_idx])
    AClosses_comp[i, :] = np.array([np.mean(inst_pow_comp[:end_idx]) for end_idx in cut_idx])
    AClosses_corr[i, :] = np.array([np.mean(inst_pow_corr[:end_idx]) for end_idx in cut_idx])
    
    
    ## cleaning and garbage collection
    del inst_pow_NOco, inst_pow_comp, inst_pow_corr
    gc.collect()
    
    print("Monte Carlo iteration n. ", i+1)


## save data on the file
save_results(file_name, cycles, AClosses_NOco, AClosses_comp, AClosses_corr)


## final garbage collection and simulation time
gc.collect()
print("Simulation completed in {:.1f} seconds.".format(time.time() - start_time))



#%% POST-PROCESSING
# process data just obtained from simulation, or loaded from file!

## test Gaussian distribution
##### CURRENTLY MISSING                 <<<<

## calculation of mean and standard deviation
m_NOco = np.mean(AClosses_NOco, axis=0)
m_comp = np.mean(AClosses_comp, axis=0)
m_corr = np.mean(AClosses_corr, axis=0)

std_NOco = np.std(AClosses_NOco, axis=0)
std_comp = np.std(AClosses_comp, axis=0)
std_corr = np.std(AClosses_corr, axis=0)


## plots
cust_plot_power(cycles, m_NOco, std_NOco, m_comp, std_comp, m_corr, std_corr, P_ac, save=0)
cust_hist_power(AClosses_NOco, cycle_index=4, filename="distr_NOco.svg", save=1)
# cust_hist_power(AClosses_comp, cycle_index=4, filename="distr_comp.svg", save=1)
# cust_hist_power(AClosses_corr, cycle_index=4, filename="distr_corr.svg", save=1)