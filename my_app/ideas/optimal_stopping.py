import numpy as np
from scipy.stats import norm

# Initial values
initial_trials = gpt_4_results.copy()
confidence_level = 0.95
z_value = norm.ppf((1 + confidence_level) / 2)
margin_of_error = 0.05

# Function to calculate margin of error
def calculate_margin_of_error(p, n, z_value):
    return z_value * np.sqrt(p * (1 - p) / n)

# Simulate sequential sampling
def sequential_sampling(initial_trials, margin_of_error, z_value):
    n = len(initial_trials)
    p_hat = np.mean(initial_trials)
    current_margin_of_error = calculate_margin_of_error(p_hat, n, z_value)
    
    while current_margin_of_error > margin_of_error:
        # Simulate next trial (in a real scenario, this would be a new data point)
        new_trial = np.random.choice([0, 1], p=[1 - p_hat, p_hat])
        initial_trials.append(new_trial)
        
        # Update mean and margin of error
        n += 1
        p_hat = np.mean(initial_trials)
        current_margin_of_error = calculate_margin_of_error(p_hat, n, z_value)
    
    return n, p_hat, current_margin_of_error

# Run the sequential sampling simulation
final_sample_size, final_p_hat, final_margin_of_error = sequential_sampling(initial_trials, margin_of_error, z_value)
final_sample_size, final_p_hat, final_margin_of_error

