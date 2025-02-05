# AI-Driven Pharmaceutical Manufacturing Digital Twin & Optimization

This project simulates a pharmaceutical manufacturing process using synthetic data generation, builds a digital twin for predictive modeling of product quality, and integrates generative AI for process parameter optimization.

## Project Structure

# Tablet Manufacturing Process Simulation

## Overview
This program simulates a tablet manufacturing process through a sequential pipeline of four key production steps:
1. **Mixing:**  
   Simulates the blending of raw materials with operator-specific biases. This step generates parameters like mixing time, mixing speed, and a uniformity index which serve as input for the next step.
   
2. **Granulation:**  
   Uses the output from the mixing process to simulate granule formation. Parameters such as granulation time, binder rate, and granule density are computed based on the mixing quality.
   
3. **Drying:**  
   Simulates the drying process of granules. The drying temperature and moisture content are determined by the properties of the granules, with a check for excessive moisture leading to potential failure.
   
4. **Compression:**  
   Models the final compression of dried granules into tablets. This step calculates parameters like compression pressure, tablet hardness, dissolution, and yield. The output depends on both the operator bias and the moisture content from the drying step.

## Key Features
- **Modular Design:**  
  Each production step is implemented in its own module (mixing, granulation, drying, and compression), ensuring clear separation of concerns and easier maintenance.
  
- **Sequential Workflow with Failure Propagation:**  
  The simulation executes the steps in sequence, where the output of one step serves as the input for the next. If a step fails (e.g., due to poor mixing quality or high moisture content), the simulation stops and reports the failure.
  
- **Operator Influence:**  
  Operator profiles, which include biases for each production step, influence the process parameters. This introduces variability and simulates real-world differences in operator performance.
  
- **Randomized Simulation with Reproducibility:**  
  Random values within defined ranges are used to simulate variability in the process. A fixed random seed (set via the utilities module) ensures that simulation runs can be reproduced for debugging and testing.
  
- **Logging and Debugging:**  
  Integrated logging throughout the modules provides detailed traceability of the simulation flow and intermediate calculations, helping to diagnose issues and fine-tune parameters.
  
- **Configurable Parameters:**  
  All process parameters (e.g., temperature ranges, pressure limits, time intervals) are defined in a dedicated parameters module, making it simple to adjust and experiment with different production scenarios.

## Running the Simulation
To execute the simulation, run the `simulator.py` module. The program will perform each production step sequentially, log the process details, and output the final result indicating whether the tablet manufacturing process was successful or if a failure occurred at a specific stage.

## Example
```bash
$ python data_generation/simulator.py
