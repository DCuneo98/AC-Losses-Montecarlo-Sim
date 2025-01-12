import numpy as np
import time
import gc
import os
from functions import generate_current_ramp, calculate_instant_power, save_results, std_sensitivity_analysis, write_sensitivity_analysis_array
from functions import cust_plot_current, cust_plot_power, cust_hist_power, plot_sensitivity_analysis


## NOTES ON FUTURE ENHANCEMENTS
# 1) in composing the time delays, I'm not sure we have to take different ones on different streams if it is the same contribution, hence the currently implemented one is a worst case scenario
# 2) we could define a variable with the inverse of the current k_DCCT, this would reduce the number of division, but it is kept as it is for compatibility with previous versions of the code
# 3) the names "gain_*" and "offset_*" could be changed into "u_*" and "D_*" respectively (just to shorten variable names)
# 4) make the "custom plot functions" less hardcoded... 
# 5) implement the "del" inside the iteration loop?
# 6) save data inside the iteration loop for deeper analysis (and also sensitivity)


## WARNINGS
# THE CURRENT DELAY FOR THE CORRECTION CASE, COULD BE UNREALISTICALLY TOO LOW
# WHEN DIFFERENTIATION, IF THE "0" IS BEFORE THE np.diff (AS I BELIEVE IT SHOULD), THE NOco POWER IS ABOVE THE REFERENCE, NOT BELOW!!!


#%% INPUTS FOR THE SIMULATION (DISCORAP case study)

## Monte Carlo simulation 
# parameters
MC_iterations = 2                                                              # number of Monte Carlo iterations
cycles = np.array([1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100])     # number of cycles of interest

Nmax_cycles = np.max(cycles)                                                    # N_values array for different cycles in the simulation
num_cycles = len(cycles)


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


## other parameters not affected by the sensitivity analysis
k_DCCT = 1/750                                                                  # DCCT DR500UX-10V/7500A, conversion factor in V/A
att_isolation = 10                                                              # isolation module ISOBLOCKv4, attenuation factor of the isolation module in V/V
range_acq = 10.52                                                               # acquisition module NI9239 in compactRIO 9049, acquisition module range in V

## A/D conversion in NI9239: sampling and quantization
f_samp = 1e3                                                                    # sample frequency in Sa/s
n_bit = 24                                                                      # resolution in bit

dt = 1 / f_samp                                                                 # sample time in s
ADC_resolution = 2*range_acq/(np.power(2,n_bit)-1)                              # ADC resolution in V


#%% SENSITIVITY ANALYSIS ON THE ONLY "CORRECTED" CASE                           >>> understand if instead the COMPENSATED case is more proper...

# there are 11 contributions to investigate, this number is currently hardcoded here
n_sensitivity = 11
file_name_array = []
for S in range(n_sensitivity):
    # name of the file with results: it changes for each sensitivity iteration
    file_name = "results_MC" + str(int(MC_iterations/1000)) + "k_cycles" + str(Nmax_cycles) + "__sensitivity_" + str(int(S+1)) + ".txt"
    file_name_array.append(file_name) 
    ## weights: they decide per each iteration of the sensitivity which uncertainty contribution is present, and it zeroes the others
    W = np.zeros(n_sensitivity)
    parameters_array = np.zeros((MC_iterations, n_sensitivity))
    W[S] = 1
    
    ### UNCERTAINTY CONTRIBUTIONS AND FURTHER PARAMETERS FROM THE MEASUREMENT CHAIN
    
    ## DCCT DR500UX-10V/7500A
    
    gain_DCCT = 37e-6                                                               # gaussiana
    # relative uncertainty
    offset_DCCT = 70e-6*W[1]                                                        # absolute tolerance in V, uniform
    delay_DCCT = np.array([1e-6, 5e-6])                                             # range of time delays in s, uniform with mean different than 0
    delay_DCCT_ave = np.average(delay_DCCT)
    
    ## isolation module ISOBLOCKv4
    gain_isolation = 2e-3                                                           # relative uncertainty, gaussian
    offset_isolation = 500e-6*W[4]                                                  # absolute tolerance in V, uniforme
    delay_isolation = np.array([0.0e-6, 5.7e-6])                                    # range of time delays in s, uniform with mean different than 0
    delay_isolation_ave  = np.average(delay_isolation)
    
    ## acquisition module NI9239 in compactRIO 9049
    gain_acq = 3e-4                                                                 # relative uncertainty, gaussian
    offset_acq = 8e-5*W[7]                                                          # relative tolerance, uniform
    delay_acq = np.array([199e-6, 201e-6])                                          # range of time delays in s, uniform with mean different than 0
    delay_acq_ave = np.average(delay_acq) 
    
    ## Compensation and correction part 
    # # sensor case
    # gain_comp = 63e-4                                                             # relative uncertainty on kds (obtained with a further MC simulation)
    # offset_comp = 19e-4                                                           # tolerance on kds (obtained with a further MC simulation)
    # delay_sens = np.array([0e-6, 6e-6])                                           # time shift due to sensor (hypothesis: small as the isolation module one)
    
    # derivative case
    gain_corr = 0e-4                                                                # relative uncertainty on kds in correction (hypothesis: ALREADY NULL), gaussian
    offset_corr = 0.1*kds*W[9]                                                      # tolerance on kds (reasonably 10 %, as for the compensation), uniform
    delay_filt = np.array([0e-4, 2e-4])                                             # time shift due to filter (hypothesis: order 25 +/- 1 % @ 10 kSa/s, then deterministic effect corrected!!!), uniform with mean different than 0
    delay_filt_ave = np.average(delay_filt)
    
    
    #### MONTE CARLO ITERATIONS
    
    # start monitoring the simulation time
    start_time = time.time()
    
    # AC losses per each iteration and each number of cycles in the conditions of interest
    # AClosses_NOco = np.zeros((num_cycles, MC_iterations))                             # no compensation nor correction
    # AClosses_comp = np.zeros((num_cycles, MC_iterations))                             # compensation case
    AClosses_corr = np.zeros((num_cycles, MC_iterations))                               # correction case
    
    # time samples of interest (max number of cycles case)
    t = np.arange(0, Nmax_cycles * period, dt)
    
     
    ## SIMULATION CORE
    for i in range(MC_iterations):
        
        ## path of magnet current: DCCT + acquisition module
        # add all the time delays here
        # the "sa" suffix stands for "sensitivity analysis"
        
        delay_DCCT_sa = (1-W[2])*delay_DCCT_ave + W[2]*np.random.uniform(low=delay_DCCT[0], high=delay_DCCT[1]) #when the weight is 1 it returns the average of the parameter value otherwise it returns 0
        delay_acq_sa = (1-W[8])*delay_acq_ave + W[8]*np.random.uniform(low=delay_acq[0], high=delay_acq[1])     #when the weight is 1 it returns the average of the parameter value otherwise it returns 0
        
        delay_current = delay_DCCT_sa +  delay_acq_sa
        
        current = generate_current_ramp(t, I_min, I_max, time_ramp, time_plateau, delay_current, period, f_samp)
        
        isamp = len(current)
        
        # apply the uncertainty of DCCT
        gain_DCCT_sa = gain_DCCT*((1-W[0])+W[0]*np.random.randn(isamp))
        
        vDCCT = ((1+gain_DCCT_sa)*k_DCCT)*current + np.random.uniform(low=-offset_DCCT, high=+offset_DCCT, size=isamp)
        
        # apply the uncertainty of the acquisition module (it will be on CH3)
        gain_acq_sa = gain_acq*((1-W[6])+W[6]*np.random.randn(isamp))
        
        CH3 = (1+gain_acq_sa)*vDCCT + range_acq*np.random.uniform(low=-offset_acq, high=+offset_acq, size=isamp)
        
        
        ## path of magnet voltage: isolation module + acquisition module
        # add the time delays of the voltage on the current entering the model
        delay_isolation_sa = (1-W[5])*delay_isolation_ave + W[5]*np.random.uniform(low=delay_isolation[0], high=delay_isolation[1])               
                                                                 
        delay_voltage = delay_isolation_sa + delay_acq_sa
        
        I = generate_current_ramp(t, I_min, I_max, time_ramp, time_plateau, delay_voltage, period, f_samp)
        
        # derivative of current
        dI_dt = np.concatenate(([0], np.diff(I)))/dt
        
        # magnet voltage in the time domain
        voltage = (R_magnet * I + L_magnet * dI_dt)
        vsamp = len(voltage)
        
        # apply the uncertainty of the isolation module
        gain_isolation_sa = gain_isolation*((1-W[3])+W[3]*np.random.randn(vsamp))
        
        vISOL = ((1+gain_isolation_sa)/att_isolation)*voltage + np.random.uniform(low=-offset_isolation, high=+offset_isolation, size=vsamp)
        
        # apply the uncertainty of the acquisition module (it will be on CH1)
        CH1 = (1+gain_acq_sa)*vISOL + range_acq*np.random.uniform(low=-offset_acq, high=+offset_acq, size=vsamp)
        

        # ## path of compensation voltage: derivative sensor + isolation module (for attenuation) + acquisition module
        # # add the time delays of the current in a different way for the derivative sensor
        # delay_sensor = np.random.uniform(low=delay_sens[0], high=delay_sens[1]) + np.random.uniform(low=delay_isolation[0], high=delay_isolation[1]) + np.random.uniform(low=delay_acq[0], high=delay_acq[1])
        # Isens = generate_current_ramp(t, I_min, I_max, time_ramp, time_plateau, delay_sensor, period, f_samp)
        # dIsens_dt = np.concatenate(([0], np.diff(Isens)))/dt
        # isens = len(dIsens_dt)
        
        # # apply the contribution of the derivative sensor
        # vCOMP = ((1+gain_comp*np.random.randn(isens))*kds + np.random.uniform(low =-offset_comp, high=+offset_comp, size=isens))*dIsens_dt
        
        # # apply the uncertainty of the isolation module used for attenuation
        # vCOMP_ISOL = ((1+gain_isolation*np.random.randn(isens))/att_isolation)*vCOMP + np.random.uniform(low=-offset_isolation, high=+offset_isolation, size=isens)
        
        # # apply contributions of the acquisition module (it will be on CH2)
        # CH2 = (1+gain_acq*np.random.randn(isens))*vCOMP_ISOL + range_acq*np.random.uniform(low=-offset_acq, high=+offset_acq, size=isens)
        
        
        ## path of correction voltage: using the measured current with delay of the filter
        # add the time delays of the current in a different way for the derivative sensor
        
        #delay_corr = np.random.uniform(low=delay_filt[0], high=delay_filt[1]) + delay_current
        delay_filt_sa = (1-W[10])*delay_filt_ave + W[10] *np.random.uniform(low=delay_filt[0], high=delay_filt[1])
        
        delay_corr = delay_filt_sa +delay_current
        Icorr = generate_current_ramp(t, I_min, I_max, time_ramp, time_plateau, delay_corr, period, f_samp)
        dIcorr_dt = np.concatenate(([0], np.diff(Icorr)))/dt
        icorr = len(dIcorr_dt)
        
        # compute the correction voltage by also considering uncertainties in the scaling factor
        # note that this should be done after the acquisition since this calculation happens on the acquisition device, but for simulation purposes it is anticipated
        vCORR = ((1+gain_corr*np.random.randn(icorr))*kds + np.random.uniform(low =-offset_corr, high=+offset_corr, size=icorr))*dIcorr_dt
        
        # apply contributions of the acquisition module (it will be on CH3x, it should be actually the same contributions of CH3, but this is a worse case)
        CH3x = (1+gain_acq_sa)*vCORR + range_acq*np.random.uniform(low=-offset_acq, high=+offset_acq, size=icorr)
        
            
        ## CALCULATION of instantaneous powers in different conditions (including quantization)
        # inst_pow_NOco = calculate_instant_power(CH1*att_isolation,   0,                 CH3/k_DCCT, ADC_resolution)
        # inst_pow_comp = calculate_instant_power(CH1*att_isolation,   CH2*att_isolation, CH3/k_DCCT, ADC_resolution)
        inst_pow_corr = calculate_instant_power(CH1*att_isolation,   CH3x,              CH3/k_DCCT, ADC_resolution)
        
        # cut at different number of cycles and retrieve the AC losses from instantaneous powers
        cut_idx = (cycles * period * f_samp).astype(int)
        
        # AClosses_NOco[:, i] = np.array([np.mean(inst_pow_NOco[:end_idx]) for end_idx in cut_idx])
        # AClosses_comp[:, i] = np.array([np.mean(inst_pow_comp[:end_idx]) for end_idx in cut_idx])
        AClosses_corr[:, i] = np.array([np.mean(inst_pow_corr[:end_idx]) for end_idx in cut_idx])
        
        parameters_array[i,:] = np.array([np.average(gain_DCCT_sa), offset_DCCT, np.average(delay_DCCT_sa), np.average(gain_isolation_sa), offset_isolation,np.average(delay_isolation_sa),
                                          np.average(gain_acq_sa), offset_acq, np.average(delay_acq_sa), offset_corr, np.average(delay_filt_sa)])
        
        ## cleaning and garbage collection
        del inst_pow_corr                                                       # IF YOU WANT TO CHANGE CASE FOR THE SENSITIVITY, EDIT HERE
        gc.collect()
        
        print("Monte Carlo iteration n. ", i+1)

        
    ## save sensitivity results on the file
    # overwrite file if already existing
    if os.path.exists(file_name):
        os.remove(file_name)
    
    with open(file_name, 'w') as file:
        # Save the array of investigated current cycles
        file.write("Investigated current cycles:\n")
        np.savetxt(file, cycles, delimiter=",")
        
        # Save the corrected AC losses
        file.write("\nAC losses values (corrected):\n")
        np.savetxt(file, AClosses_corr, delimiter=",")                          # IF YOU WANT TO CHANGE CASE FOR THE SENSITIVITY, EDIT HERE

        file.write("\nParameters used:\n")
        np.savetxt(file, parameters_array, delimiter=",")
    
    ## final garbage collection and simulation time
    gc.collect()
    print("Simulation completed in {:.1f} seconds.".format(time.time() - start_time))



# #%% POST-PROCESSING
# # process data just obtained from simulation, or loaded from file!

std_dv_array_from_files = std_sensitivity_analysis(len(cycles), file_name_array)
write_sensitivity_analysis_array(file_name = "sensitivity_analysis_std_dv_results.txt", std_dvs = std_dv_array_from_files)
# ## test Gaussian distribution
# ##### CURRENTLY MISSING             
# <<<<

# ## calculation of mean and standard deviation
# m_NOco = np.mean(AClosses_NOco, axis=1)
# m_comp = np.mean(AClosses_comp, axis=1)
m_corr = np.mean(AClosses_corr, axis=1)

# std_NOco = np.std(AClosses_NOco, axis=1)
# std_comp = np.std(AClosses_comp, axis=1)
std_corr = np.std(AClosses_corr, axis=1)


# ## plots
# cust_hist_power(AClosses_NOco, cycle_index=4, filename="distr_NOco.svg", save=1)
# # cust_hist_power(AClosses_comp, cycle_index=4, filename="distr_comp.svg", save=1)
# # cust_hist_power(AClosses_corr, cycle_index=4, filename="distr_corr.svg", save=1)
cust_plot_power(cycles, m_corr, std_corr, m_corr, std_corr, m_corr, std_corr, P_ac, save=0)         ### TEMPORARY!!
plot_sensitivity_analysis(cycles, std_dv_array_from_files, save_path="Sensitivity Analysis.svg") 


        
    
# %%
