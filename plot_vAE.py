### RETRIEVE DATA FROM FILES TO POST-PROCESS AND PLOT
## UNCERTAINTY ANALYSIS CASE FIRST
import numpy as np
import matplotlib.pyplot as plt
from functions import cust_plot_power

## File with data to load
file_name = 'results_MC0k_cycles100.txt'

## Load cycles info
cycles = []
with open(file_name) as infile:
    copy = False
    for line in infile:
        if line.strip() == "Investigated current cycles:":
            copy = True
            continue
        elif line.strip() == "AC losses values (not compensated nor corrected):":
            copy = False
            continue
        elif copy:
            cycles.append(line)
            
# convert to array and avoid the last null line
cycles = np.asarray(cycles[:-1]).astype(float)

# plot (for DEBUG)
plt.plot(cycles)


## Load AC losses values "NOco case"
AClosses_NOco_temp = []
with open(file_name) as infile:
    copy = False
    for line in infile:
        if line.strip() == "AC losses values (not compensated nor corrected):":
            copy = True
            continue
        elif line.strip() == "AC losses values (compensated):":
            copy = False
            continue
        elif copy:
            AClosses_NOco_temp.append(line)
            
# convert to array and avoid the last null line
AClosses_NOco_temp = AClosses_NOco_temp[:-1]
AClosses_NOco = np.array([list(map(float, row.split(','))) for row in AClosses_NOco_temp])


## Load AC losses values "compensation case"
AClosses_comp_temp = []
with open(file_name) as infile:
    copy = False
    for line in infile:
        if line.strip() == "AC losses values (compensated):":
            copy = True
            continue
        elif line.strip() == "AC losses values (corrected):":
            copy = False
            continue
        elif copy:
            AClosses_comp_temp.append(line)
            
# convert to array and avoid the last null line
AClosses_comp_temp = AClosses_comp_temp[:-1]
AClosses_comp = np.array([list(map(float, row.split(','))) for row in AClosses_comp_temp])


## Load AC losses values "correction case"
AClosses_corr_temp = []
with open(file_name) as infile:
    copy = False
    for line in infile:
        if line.strip() == "AC losses values (corrected):":
            copy = True
            continue
        elif line.strip() == "":                                # end of file actually
            copy = False
            continue
        elif copy:
            AClosses_corr_temp.append(line)
            
# convert to array, no last null line in this case
AClosses_corr = np.array([list(map(float, row.split(','))) for row in AClosses_corr_temp])


# plot (to ADJUST)
m_NOco = np.mean(AClosses_NOco, axis=1)
std_NOco = np.std(AClosses_NOco, axis=1)

m_comp = np.mean(AClosses_comp, axis=1)
std_comp = np.std(AClosses_comp, axis=1)

m_corr = np.mean(AClosses_corr, axis=1)
std_corr = np.std(AClosses_corr, axis=1)

cust_plot_power(cycles, m_NOco, std_NOco, m_comp, std_comp, m_corr, std_corr, 37.6, 0)