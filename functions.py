import numpy as np
import matplotlib.pyplot as plt
import os


### Function needed to create smoother transitions at the beginning and end of ramps in the current cycle
# WARNING: CURRENTLY NOT USED IN THE "generate_current_ramp" VERSION BELOW
def sinusoidal_transition(x, x0, x1, y0, y1):
    t = (x - x0) / (x1 - x0)
    t = np.clip(t, 0, 1)
    return y0 + (y1 - y0) * 0.5 * (1 - np.cos(np.pi * t))


### Function used to create the current waveform with ramps
def generate_current_ramp(t, I_min, I_max, time_plateau, time_ramp, time_offset, period, f_samp):

   # consider the offset by shifting the time scale
    t_mod = np.mod(t - time_offset, period)
    
    # initialize current array
    I = np.zeros_like(t)
    n_samp = len(I)
        
    # calculate relevant quantities for the ramp (to speed up the code)  
    I_ramp_rate = (I_max - I_min) / time_ramp
    q = I_max + I_ramp_rate * (time_ramp + time_plateau)

    # generate the single cycle
    for i in range(n_samp):
        if 0 <= t_mod[i] < time_ramp:
            # Ramp up
            I[i] = I_min + I_ramp_rate * t_mod[i]
        elif time_ramp <= t_mod[i] < time_ramp + time_plateau:
            # Plateau at max
            I[i] = I_max
        elif time_ramp + time_plateau <= t_mod[i] < 2 * time_ramp + time_plateau:
            # Ramp down
            I[i] = q - I_ramp_rate * t_mod[i]
        else:
            # Plateau at min
            I[i] = I_min

    # cut to the number of points we need
    return I


### Function used to create the current waveform with ramps
def calculate_instant_power(voltage, voltage_correction, current, lsb):
    
    # quantization implemented here to have it just before calculation
    voltage = np.round(voltage / lsb) * lsb
    voltage_correction = np.round(voltage_correction / lsb) * lsb
    current = np.round(current / lsb) * lsb
    
    return (voltage - voltage_correction) * current


### Functions used to save relevant data into files
## current cycles and relevant conditions are highlighted
## the number of iterations is typically highlighted in the file name, but it corresponds to the number of rows of each matrix
def save_results(file_path, cycles, NO_comp_nor_corr, compensated, corrected):
    
    # overwrite file if already existing
    if os.path.exists(file_path):
        os.remove(file_path)
    
    with open(file_path, 'w') as file:
        # Save the array of investigated current cycles
        file.write("Investigated current cycles:\n")
        np.savetxt(file, np.transpose(cycles), delimiter=",")
        
        # Save the non-compensated AC losses
        file.write("\nAC losses values (not compensated nor corrected):\n")
        np.savetxt(file, NO_comp_nor_corr, delimiter=",")
        
        # Save the compensated AC losses
        file.write("\nAC losses values (compensated):\n")
        np.savetxt(file, compensated, delimiter=",")
        
        # Save the corrected AC losses
        file.write("\nAC losses values (corrected):\n")
        np.savetxt(file, corrected, delimiter=",")


### Functions used to save relevant data within single iterations
##
def salva_dati_iterazione(nome_file, vettore):
    # Converte il vettore in tipo float32 per ridurre l'uso della memoria
    vettore = vettore.astype(np.float32)
    
    # Apre il file in modalità 'append' per aggiungere una nuova riga
    with open(nome_file, 'a') as f:
        # Scrive il vettore nel file come una nuova riga
        np.savetxt(f, vettore.reshape(1, -1), delimiter=',', fmt='%.8e')  # Formato scientifico per float32
        
        
        
### PLOT FUNCTIONS
def cust_plot_current(t, I, save):
    # Custom plot for the current cycle
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, I, label='ramped current waveform')
    plt.xlabel('time / s')
    plt.ylabel('current / A')
    plt.title('current cycles')                                     # TEMPORARY, MUST BE REMOVED FROM FINAL VERSION FOR PAPER
    plt.grid(True)
    
    if (save == 1):
        plt.savefig('current_cycle.svg', format='svg')
    
    plt.show()


def cust_plot_power(cycles, m_NOco, std_NOco, m_comp, std_comp, m_corr, std_corr, ref_Pac, save): 
    # Custom plot for the AC losses as a function of cycles for different conditions

    plt.figure(figsize=(12, 6))
    
    # plot the reference AC losses value
    plt.axhline(y=ref_Pac, xmin=0, xmax=1, linestyle='dashed', color='k', label='AC power losses')
    
    # plot the results per case as a function of current cycles
    plt.errorbar(cycles, m_NOco, yerr=std_NOco, fmt='o', capsize=5, label='no compensation/correction')
    plt.errorbar(cycles, m_comp, yerr=std_comp, fmt='o', capsize=5, label='compensation case')
    plt.errorbar(cycles, m_corr, yerr=std_corr, fmt='o', capsize=5, label='correction case')
    
    plt.xlabel('number of exploited cycles')
    plt.ylabel('AC power losses / W')
    plt.title('power losses as a number of acquisition cycles')     # TEMPORARY, MUST BE REMOVED FROM FINAL VERSION FOR PAPER
    plt.legend()
    plt.grid(True)
    plt.xticks(cycles) 
    
    if (save == 1):
        plt.savefig('power_losses_statistics.svg', format='svg')
        
    plt.show()

    
def cust_hist_power(ACloss_dist, cycle_index, filename, save): 
    # Custom plot for the AC losses histogram
    
    # Extract the data corresponding to the desired cycles
    power_values = ACloss_dist[:, cycle_index]
    
    # Create the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(power_values, edgecolor='black')
    plt.xlabel('AC losses values / W')
    plt.ylabel('statistical frequency')
    plt.title('histogram for a specific number of cycles')          # TEMPORARY, MUST BE REMOVED FROM FINAL VERSION FOR PAPER
    plt.grid(True) 
    
    if (save == 1):  
        plt.savefig(filename, format='svg')
        
    plt.show()
  
