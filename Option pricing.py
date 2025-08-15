import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def d1(S0, K, T, r, sigma):
    return (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def d2(S0, K, T, r, sigma):
    return d1(S0, K, T, r, sigma) - sigma * np.sqrt(T)

def black_scholes_call(S0, K, T, r, sigma):
    d1_val = d1(S0, K, T, r, sigma)
    d2_val = d2(S0, K, T, r, sigma)
    return S0 * norm.cdf(d1_val) - K * np.exp(-r * T) * norm.cdf(d2_val)

def black_scholes_put(S0, K, T, r, sigma):
    d1_val = d1(S0, K, T, r, sigma)
    d2_val = d2(S0, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2_val) - S0 * norm.cdf(-d1_val)

def monte_carlo_call(S0, K, T, r, sigma, sims):
    Z = np.random.standard_normal(sims)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0)
    return np.exp(-r * T) * np.mean(payoff)

def monte_carlo_put(S0, K, T, r, sigma, sims):
    Z = np.random.standard_normal(sims)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(K - ST, 0)
    return np.exp(-r * T) * np.mean(payoff)

if __name__ == "__main__":
    S0, K, T, r, sigma = 100, 105, 1, 0.05, 0.2
    
    # Theoretical prices
    bs_call = black_scholes_call(S0, K, T, r, sigma)
    bs_put = black_scholes_put(S0, K, T, r, sigma)
    
    # Simulations for convergence
    simulation_counts = np.logspace(2, 7, 20, dtype=int)  # 100 to 10,000,000
    mc_calls = [monte_carlo_call(S0, K, T, r, sigma, sims) for sims in simulation_counts]
    mc_puts = [monte_carlo_put(S0, K, T, r, sigma, sims) for sims in simulation_counts]
    
    # Plot Call
    plt.figure(figsize=(8,5))
    plt.plot(simulation_counts, mc_calls, label="Monte Carlo Call")
    plt.axhline(bs_call, color="red", linestyle="--", label=f"BS Call Price ({bs_call:.4f})")
    plt.xscale("log")
    plt.xlabel("Number of Simulations")
    plt.ylabel("Call Option Price")
    plt.title("Monte Carlo Convergence to Black–Scholes Call Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot Put
    plt.figure(figsize=(8,5))
    plt.plot(simulation_counts, mc_puts, label="Monte Carlo Put")
    plt.axhline(bs_put, color="red", linestyle="--", label=f"BS Put Price ({bs_put:.4f})")
    plt.xscale("log")
    plt.xlabel("Number of Simulations")
    plt.ylabel("Put Option Price")
    plt.title("Monte Carlo Convergence to Black–Scholes Put Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
