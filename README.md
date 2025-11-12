## Overview  
The project focuses on implementing and analyzing a **Sparse Matrix–Vector Multiplication (SpMV)** algorithm, exploring both a **sequential** and a **shared-memory parallel** approach using **OpenMP**.

The objectives are:
- Understand memory-bound computation in sparse linear algebra.
- Implement an efficient **COO → CSR conversion**.
- Develop a **reference sequential SpMV** kernel.
- Implement and tune **OpenMP parallel versions** with configurable scheduling.
- Measure performance using **p90 execution times** and compute effective **memory bandwidth**.
- Compare **Sequential vs OpenMP** implementations.

---

## How to Run on the UniTN HPC Cluster

>⚠️ Note: The matrix `Spielman_k200.mtx` is compressed as `Spielman_k200.rar`
>because it exceeds GitHub’s 100 MB limit.
>
>Before submit the job, extract it.

### Submit the job

```bash
qsub spmv.pbs
```  

### Check the job outputs
- `spmv.out` → standard output  
- `spmv.err` → error 
- `results.txt` → performance results


### Generate Plots

```
module load python-3.10.14_gcc91
python3 plot_scaling.py results.txt
```  

## Repository Structure
```
├── makefile                 # build & run commands
├── spmv.c                   # Sequential + OpenMP implementation
├── spmv.pbs                 # PBS script for UniTN HPC cluster
├── results.txt              # benchmark results
├── plot_scaling.py          # plot generation script
├── matrix/                  # test matrices (.mtx and .rar)
├── plot/                    # generated plots
└── lionello-253647-D1.pdf   # project report
```
---


