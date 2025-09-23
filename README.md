# Research Compiler – Parallel Automatic Differentiation  

## Overview  
This project extends a research compiler to support **automatic differentiation (AD)** in both **forward-mode** and **reverse-mode**, with parallel execution using **OpenMP** and **MPI**. The goal was to automatically generate efficient parallel C code for derivative computations, benchmark scaling across multicore clusters, and evaluate performance tradeoffs.  

## What I Did  
- **Implemented AD passes** in Python for both forward and reverse differentiation  
  - `my_forward_diff.py` – generates parallel C code for forward-mode AD  
  - `my_reverse_diff.py` – generates parallel C code for reverse-mode AD  
- Integrated AD with the compiler’s intermediate representation and code generator  
- Designed a runtime system to handle vectorized operations and memory management  
- Benchmarked strong and weak scaling on multicore clusters, analyzing performance across different problem sizes  
- Debugged compiler transformations and dependency handling for correctness under parallel execution  

## Key Features  
- Generates **parallelized C kernels** for differentiation using OpenMP/MPI  
- Supports **forward-mode AD** for functions with small input dimension  
- Supports **reverse-mode AD** for functions with large input dimension  
- Evaluation framework to compare runtime speedup and memory tradeoffs  

## Skills Highlighted  
- **Programming:** Python, C, MPI, OpenMP  
- **Systems/Compilers:** Compiler passes, code generation, IR transformations  
- **Parallel Computing:** Strong/weak scaling experiments, optimization for multicore clusters  
- **Debugging & Optimization:** Dependency management, performance bottleneck analysis  

## Project Structure  
- `my_forward_diff.py` – forward-mode AD implementation  
- `my_reverse_diff.py` – reverse-mode AD implementation  
- `benchmarks/` – performance tests and scaling experiments  
- `examples/` – sample input programs and generated code  

---

This is an educational compiler/programming language for differentiable programming, used for the course [CSE 291](https://cseweb.ucsd.edu/~tzli/cse291/) in UCSD.  