import numpy as np
from scipy.stats import norm, chi2

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
        print("Cannot reject the hypothesis that the data do NOT follow a normal distribution (p < 0.05).\n")
    else:
        print("There are no evidences to trust the null hypotesis (p >= 0.05).\n")
    
    print("##################################################################################################")

file_path = "results_MC10k_cycles100.txt"
additional_keywords = ['(not compensated nor corrected):\n', '(compensated):\n', '(corrected):\n']
index = int(input("Insert an index: \n 0: Not compensated power losses\n 1: Compensated power losses\n 2: Corrected power losses\n"))
data_row_index = int(input(
    "Prompt the index of the distribution to plot inserting a number from 1 to 15:\n"
    "1: 1 cycle\n"
    "2: 2 cycles\n"
    "3: 3 cycles\n"
    "4: 4 cycles\n"
    "5: 5 cycles\n"
    "6: 10 cycles\n"
    "7: 15 cycles\n"
    "8: 20 cycles\n"
    "9: 25 cycles\n"
    "10: 30 cycles\n"
    "11: 40 cycles\n"
    "12: 50 cycles\n"
    "13: 60 cycles\n"
    "14: 80 cycles\n"
    "15: 100 cycles\n"
))
print("The file path name:", file_path, "\n")
data = data_from_file_chi_squared_test(file_path,additional_keywords, index, data_row_index)
chi_squared_normality_test(data)
