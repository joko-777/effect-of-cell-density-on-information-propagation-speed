"""
================================================================================
Package Installation Script
================================================================================
"""

using Pkg

packages = [
    "Agents",
    "CairoMakie",
    "Catalyst",
    "CSV",
    "DataFrames",
    "Dates",
    "DelayDiffEq",
    "DifferentialEquations",
    "Distributions",
    "GLMakie",
    "JSON3",
    "JumpProcesses",
    "LsqFit",
    "ModelingToolkit",
    "NearestNeighbors",
    "Plots",
    "Printf",
    "ProgressBars",
    "ProgressMeter",
    "Random",
    "SpecialFunctions",
    "StaticArrays",
    "Statistics",
    "Symbolics"
]

println("Installing required packages...")
println("="^50)

Pkg.add(packages)

println("="^50)
println("✓ All packages installed successfully!")
