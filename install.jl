"""
================================================================================
Cell Density & Enzymatic Clearance: Information Propagation Analysis
Package Installation Script
================================================================================
"""

using Pkg

# List of required packages
packages = [
    "Catalyst",
    "ModelingToolkit",
    "DifferentialEquations",
    "GLMakie",
    "JSON3",
    "Printf",
    "Symbolics",
    "Agents",
    "CairoMakie",
    "Random",
    "JumpProcesses",
    "Distributions",
    "ProgressMeter",
    "StaticArrays",
    "CSV",
    "DataFrames",
    "DelayDiffEq",
    "LsqFit",
    "Dates",
    "ProgressBars"
]

println("Installing required packages...")
println("=" ^ 50)
Pkg.add(packages)

println("=" ^ 50)
println("✓ All packages installed successfully!")
