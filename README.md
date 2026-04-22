# Effect of Cell Density and Enzymatic Clearance on Information Propagation Speed in a Cell Culture

**Associated manuscript (in preparation):**
> Johannes Konrad, *"On the Role of Relay Cells in Information Propagation Speed for Dictyostelium Chemotaxis: An Agent-Based Modeling Study"*
> Institute for Digital Communications (LHFT), Friedrich-Alexander-Universität Erlangen-Nürnberg
> Contact: johannes.konrad@fau.de

---

## Overview

This repository contains the full simulation and analysis pipeline for studying how cell density and enzymatic cAMP clearance shape the speed at which biochemical information propagates through a chain of *Dictyostelium discoideum* cells.

The model treats the cell chain as a **molecular communication channel**: a transmitter cell (TX) secretes cyclic AMP (cAMP) in response to an intracellular signal, intermediate relay cells re-amplify and retransmit the signal, and a receiver cell (RX) is the downstream readout. Information propagation speed is quantified using **lagged mutual information (LMI)** between the TX cAMP output and the RX CAR1 receptor activity.


## Repository Structure

```
.
├── simulation.jl            # Main agent-based model (ABM) simulation
├── computekernel.jl         # 2D PDE impulse-response kernel computation and fitting
├── laggedMutalInformation.jl # KSG-based lagged CMI / MI analysis
├── odefit.jl                # ODE fitting of intracellular dynamics
├── install.jl               # Package installation script
├── LICENSE                  # GPL-3.0
└── README.md
```

## Dependencies

All code is written in [Julia](https://julialang.org/) (≥ 1.9 recommended). Key packages:

| Package | Purpose |
|---|---|
| `Agents.jl` | Agent-based model framework |
| `Catalyst.jl` | Symbolic reaction network definition |
| `JumpProcesses.jl` | Gillespie SSA / NRM integration |
| `ModelingToolkit.jl` | Symbolic system construction |
| `DifferentialEquations.jl` | ODE solvers |
| `DelayDiffEq.jl` | Delay differential equations |
| `NearestNeighbors.jl` | KD-tree for KSG MI estimator |
| `SpecialFunctions.jl` | Digamma function (KSG) |
| `LsqFit.jl` | Nonlinear least squares fitting |
| `CairoMakie.jl` | Plotting |
| `CSV.jl`, `DataFrames.jl` | Data I/O |
| `JSON3.jl` | Parameter serialization |
| `Distributions.jl` | Stochastic draws |
| `StaticArrays.jl` | Performance (fixed-size arrays) |

---

## License

GPL-3.0. See [LICENSE](LICENSE).

---

## Citation

If you use this code, please cite the associated manuscript (preprint/publication details to be added upon release).
