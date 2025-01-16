import numpy as np
import matplotlib.pyplot as plt

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

def plot_sensitivity_analysis(cycles, standard_deviations, save_path = None):
    labels = ["gain DCCT", "offset DCCT", "delay DCCT", "gain isolation", "offset isolation", 
              "delay isolation", "gain acquisition", "offset acquisition", "delay acquisition", 
              "offset correction", "delay correction"]
    
    plt.figure(figsize=(10, 8))  # Imposta una figura con dimensioni più piccole
    #colors = ['b', 'g', 'r']
    colors = plt.cm.Dark2.colors
    markers = ['o', '^', 's']
    common_xlabel = 'number of exploited cycles'
    common_ylabel = 'standard deviation of losses / W'
    common_lim = (1e-6, 1e+2)
    
    
    # Primo plot (colonne 1, 2, 3)
    plt.subplot(2, 2, 1)  # 2 righe, 2 colonne, primo subplot
    for i in range(3):  # Prima, seconda e terza colonna
        label = labels[i]
        plt.plot(cycles, standard_deviations[:, i], label=label, color=colors[i], marker=markers[i], markersize=5)
        plt.yscale('log')
    plt.grid(True)
    plt.xlabel(common_xlabel)
    plt.ylabel(common_ylabel)
    plt.ylim(common_lim)
    #plt.title('Sensitivity analysis: Gain DCCT, Offset DCCT, Delay DCCT')
    plt.legend()
    
    # Secondo plot (colonne 4, 5, 6)
    plt.subplot(2, 2, 2)  # 2 righe, 2 colonne, secondo subplot
    for i in range(3, 6):  # Quarta, quinta e sesta colonna
        label = labels[i]
        plt.plot(cycles, standard_deviations[:, i], label=label, color=colors[i-3], marker=markers[i-3], markersize=5)
        plt.yscale('log')
    plt.grid(True)
    plt.xlabel(common_xlabel)
    plt.ylabel(common_ylabel)
    plt.ylim(common_lim)
    #plt.title('Sensitivity analysis: Gain isolation, Offset isolation, Delay isolation')
    plt.legend()
    
    # Terzo plot (colonne 7, 8, 9)
    plt.subplot(2, 2, 3)  # 2 righe, 2 colonne, terzo subplot
    for i in range(6, 9):  # Settima, ottava e nona colonna
        label = labels[i]
        plt.plot(cycles, standard_deviations[:, i], label=label, color=colors[i-6], marker=markers[i-6], markersize=5)
        plt.yscale('log')
    plt.grid(True)
    plt.xlabel(common_xlabel)
    plt.ylabel(common_ylabel)
    plt.ylim(common_lim)
    #plt.title('Sensitivity analysis: Gain acquisition, Offset acquisition, Delay acquisition')
    plt.legend()
    
    # Quarto plot (colonne 10, 11)
    plt.subplot(2, 2, 4)  # 2 righe, 2 colonne, quarto subplot
    for i in range(9, 11):  # Decima e undicesima colonna
        label = labels[i]
        plt.plot(cycles, standard_deviations[:, i], label=label, color=colors[i-8], marker=markers[i-8], markersize=5)
        plt.yscale('log')
    plt.grid(True)
    plt.xlabel(common_xlabel)
    plt.ylabel(common_ylabel)
    plt.ylim(common_lim)
    #plt.title('Sensitivity analysis: Offset correction, Delay filt')
    plt.legend()
    
    # Ridurre lo spazio tra i subplots
    plt.subplots_adjust(hspace=0.3, wspace=0.3)  # Regola gli spazi tra righe (hspace) e colonne (wspace)
    
    plt.tight_layout()  # Ottimizza la disposizione
    if save_path:
        plt.savefig(save_path, format='svg')
        print(f"Figure saved as {save_path}")
    
    plt.show()
    
MC_iterations = 1000
cycles = np.array([1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100])
Nmax_cycles = 100
file_name_array = []    
n_sensitivity = 11
for S in range (n_sensitivity): 
    file_name = "results_MC" + str(int(MC_iterations/1000)) + "k_cycles" + str(Nmax_cycles) + "__sensitivity_" + str(int(S+1)) + ".txt"
    file_name_array.append(file_name)
    
std_dv_array_from_files = std_sensitivity_analysis(len(cycles), file_name_array)
plot_sensitivity_analysis(cycles, std_dv_array_from_files, save_path="power_losses_sensitivity.svg") 