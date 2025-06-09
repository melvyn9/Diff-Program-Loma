import matplotlib.pyplot as plt

# Data from user's strong scaling results
cores = [1, 2, 4, 6, 8]

# Execution time (seconds)
fwd_times = [0.012460, 0.011303, 0.003603, 0.004232, 0.003164]
rev_times = [0.007045, 0.003294, 0.001676, 0.001133, 0.001690]

# Rate (M calls/s)
fwd_rates = [80.257, 88.468, 277.549, 236.291, 316.096]
rev_rates = [141.936, 303.558, 596.787, 882.854, 591.856]

# Speedup (T1 / Tn)
fwd_speedup = [fwd_times[0] / t for t in fwd_times]
rev_speedup = [rev_times[0] / t for t in rev_times]
ideal_speedup = cores  # ideal linear speedup

# Plot 1: Execution Time vs Cores
plt.figure()
plt.plot(cores, fwd_times, marker='o', label='Forward Mode')
plt.plot(cores, rev_times, marker='s', label='Reverse Mode')
plt.xlabel("Number of Cores")
plt.ylabel("Execution Time (s)")
plt.title("Strong Scaling: Execution Time vs Cores")
plt.legend()
plt.grid(True)

# Plot 2: Speedup vs Cores
plt.figure()
plt.plot(cores, fwd_speedup, marker='o', label='Forward Mode')
plt.plot(cores, rev_speedup, marker='s', label='Reverse Mode')
plt.plot(cores, ideal_speedup, 'k--', label='Ideal Speedup')
plt.xlabel("Number of Cores")
plt.ylabel("Speedup (T1 / Tn)")
plt.title("Strong Scaling: Speedup vs Cores")
plt.legend()
plt.grid(True)

# Plot 3: Throughput (Rate) vs Cores
plt.figure()
plt.plot(cores, fwd_rates, marker='o', label='Forward Mode')
plt.plot(cores, rev_rates, marker='s', label='Reverse Mode')
plt.xlabel("Number of Cores")
plt.ylabel("Throughput (M calls/s)")
plt.title("Strong Scaling: Throughput vs Cores")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
