import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2
import os


# ### Function needed to create smoother transitions at the beginning and end of ramps in the current cycle
# # WARNING: CURRENTLY NOT USED IN THE "generate_current_ramp" VERSION BELOW, SURELY TO ADJUST
# def smooth_transition(x, x0, x1, y0, y1):
#     t = (x - x0) / (x1 - x0)
#     t = np.clip(t, 0, 1)
#     return y0 + (y1 - y0) * 0.5 * (1 - np.cos(np.pi * t))


### Function used to create the current waveform with ramps
def generate_current_ramp(t, I_min, I_max, time_ramp, time_plateau, time_offset, period, f_samp):

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
        np.savetxt(file, cycles, delimiter=",")
        
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
# def salva_dati_iterazione(nome_file, vettore):
#     # Converte il vettore in tipo float32 per ridurre l'uso della memoria
#     vettore = vettore.astype(np.float32)
    
#     # Apre il file in modalità 'append' per aggiungere una nuova riga
#     with open(nome_file, 'a') as f:
#         # Scrive il vettore nel file come una nuova riga
#         np.savetxt(f, vettore.reshape(1, -1), delimiter=',', fmt='%.8e')  # Formato scientifico per float32
        

### Function for a custom normality test
def chi_squared_normality_test(data, num_bins=15):
    # Calculate the mean and standard deviation of the data
    mu = np.mean(data)
    sigma = np.std(data)

    # Calculate observed frequencies
    counts, edges = np.histogram(data, bins=num_bins)

    # Calculate theoretical frequencies
    bin_centers = edges[:-1] + np.diff(edges) / 2  # Compute bin centers
    theoretical_probs = norm.pdf(bin_centers, mu, sigma) * np.diff(edges)  # Theoretical probabilities
    theoretical_freqs = theoretical_probs * len(data)  # Theoretical frequencies

    # Calculate the chi-squared statistic
    chi_squared = np.sum((counts - theoretical_freqs) ** 2 / theoretical_freqs)

    # Degrees of freedom (number of bins - 1 - number of estimated parameters, here 2: mean and sigma)
    dof = num_bins - 1 - 2

    # Calculate the p-value
    p_value = 1 - chi2.cdf(chi_squared, dof)

    # Output the results
    print("##################################################################################################\n")
    print(f"H0: the data DO NOT follow a normal distribution\n")
    print(f"Chi-squared statistic: {chi_squared:.2f}")
    print(f"Degrees of freedom: {dof}")
    print(f"p-value: {p_value:.3f}")

    # Interpretation
    if p_value < 0.05:
        print("The null hypothesis was rejected, the distribution is NORMAL with probability of false negative < 5 %.\n")
    else:
        print("There is no evidence to reject the null hypotesis (p >= 0.05).\n")
    
    print("##################################################################################################")

        
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
    power_values = ACloss_dist[cycle_index, :]
    
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
  

#### SOME FUNCTIONS THAT NEED TO BE ADJUSTED  
def std_sensitivity_analysis(cycles, filenames):
    results = np.zeros((cycles, len(filenames)))  # Per salvare le deviazioni standard per ogni file
    
    for i, filename in enumerate(filenames):  # Usa enumerate per ottenere l'indice i
        # Apre il file e legge tutte le righe
        with open(filename, "r") as file:
            lines = file.readlines()
        
        data_lines = lines[18:34]  
        
        data = np.genfromtxt(data_lines, delimiter=",")
        
        std_deviation = np.std(data, axis=1)
    
        results[:, i] = std_deviation
    
    return results       

def write_sensitivity_analysis_array(file_name, std_dvs): 
    if os.path.exists(file_name):
        os.remove(file_name)
    with open(file_name, 'w') as file:
        # Save the array of investigated current cycles
        file.write("Standard deviations for the evaluated parameters:\n")
        np.savetxt(file, std_dvs, delimiter=",")   

def data_from_file_chi_squared_test(file_path, keywords, index, data_row_index):
    base_keyword = 'AC losses values'  # Base string to search for
    if index < 0 or index >= len(keywords):
        raise ValueError(f"Invalid index {index}. Please select a value between 0 and {len(keywords) - 1}.")

    selected_keyword = keywords[index]
    search_string = f"{base_keyword} {selected_keyword}"
    print(f"Searching for: '{search_string}'")

    with open(file_path, 'r') as file:
        found = False
        for line in file:
            if search_string in line:
                found = True
                break  # Exit the loop when the keyword is found

        if not found:
            raise ValueError(f"Keyword '{search_string}' not found in the file.")

        # Skip `data_row_index` lines after the found line
        for _ in range(data_row_index):
            line = next(file, None)
            if line is None:
                raise ValueError("Reached the end of the file before finding the specified row.")

        print(f"Extracting data from line: {line.strip()}")
        # Convert the line to a numpy array
        data_array = np.array([float(value) for value in line.strip().split(',')])
        return data_array
    

def plot_sensitivity_analysis(cycles, standard_deviations, save_path = None):
    labels = ["Gain_DCCT", "Offset_DCCT", "Delay_DCCT", "Gain_isolation", "Offset_isolation", 
              "Delay_isolation", "Gain_acquisition", "Offset_acquisition", "Delay_acquisition", 
              "Offset_correction", "Delay_filt"]
    
    plt.figure(figsize=(10, 8))  # Imposta una figura con dimensioni più piccole

    # Primo plot (colonne 1, 2, 3)
    plt.subplot(2, 2, 1)  # 2 righe, 2 colonne, primo subplot
    for i in range(3):  # Prima, seconda e terza colonna
        label = labels[i]
        plt.plot(cycles, standard_deviations[:, i], label=label)
    plt.grid(True)
    plt.xlabel('Number of exploited cycles')
    plt.ylabel('Std. Dev AC power losses / W')
    plt.title('Sensitivity: Gain_DCCT, Offset_DCCT, Delay_DCCT')
    plt.legend()
    
    # Secondo plot (colonne 4, 5, 6)
    plt.subplot(2, 2, 2)  # 2 righe, 2 colonne, secondo subplot
    for i in range(3, 6):  # Quarta, quinta e sesta colonna
        label = labels[i]
        plt.plot(cycles, standard_deviations[:, i], label=label)
    plt.grid(True)
    plt.xlabel('Number of exploited cycles')
    plt.ylabel('Std. Dev AC power losses / W')
    plt.title('Sensitivity: Gain_isolation, Offset_isolation, Delay_isolation')
    plt.legend()
    
    # Terzo plot (colonne 7, 8, 9)
    plt.subplot(2, 2, 3)  # 2 righe, 2 colonne, terzo subplot
    for i in range(6, 9):  # Settima, ottava e nona colonna
        label = labels[i]
        plt.plot(cycles, standard_deviations[:, i], label=label)
    plt.grid(True)
    plt.xlabel('Number of exploited cycles')
    plt.ylabel('Std. Dev AC power losses / W')
    plt.title('Sensitivity: Gain_acquisition, Offset_acquisition, Delay_acquisition')
    plt.legend()
    
    # Quarto plot (colonne 10, 11)
    plt.subplot(2, 2, 4)  # 2 righe, 2 colonne, quarto subplot
    for i in range(9, 11):  # Decima e undicesima colonna
        label = labels[i]
        plt.plot(cycles, standard_deviations[:, i], label=label)
    plt.grid(True)
    plt.xlabel('Number of exploited cycles')
    plt.ylabel('Std. Dev AC power losses / W')
    plt.title('Sensitivity: Offset_correction, Delay_filt')
    plt.legend()
    
    # Ridurre lo spazio tra i subplots
    plt.subplots_adjust(hspace=0.3, wspace=0.3)  # Regola gli spazi tra righe (hspace) e colonne (wspace)
    
    plt.tight_layout()  # Ottimizza la disposizione
    if save_path:
        plt.savefig(save_path, format='svg')
        print(f"Figure saved as {save_path}")
    
    plt.show()