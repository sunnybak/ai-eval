import random
import math

def simulate_coin_flips(p, batch_size=20, confidence_level=0.90, margin_of_error=0.1):
    z_95 = 1.96  # z-score for 95% confidence level
    z_90 = 1.645  # z-score for 90% confidence level
    p_hat = 0
    n_batches = 0

    while True:
        n_batches += 1
        successes = 0

        # Flip the coin batch_size times
        for _ in range(batch_size):
            success = random.random() < p
            successes += success

        # Update p_hat after each batch
        p_hat = (p_hat * (n_batches - 1) * batch_size + successes) / (n_batches * batch_size)

        margin = z_90 * math.sqrt(p_hat * (1 - p_hat) / (n_batches * batch_size))
        if margin <= margin_of_error:
            break

    return n_batches, p_hat

# Test the simulation with different values of p
p_values = [0.1, 0.3, 0.5, 0.7, 0.9]

for p in p_values:
    n_batches, p_hat = simulate_coin_flips(p)
    print(f"True p: {p}, Estimated p: {p_hat:.3f}, Number of batches: {n_batches}")