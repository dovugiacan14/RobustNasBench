# RobustNASBench

A framework for evaluating and analyzing neural architecture search (NAS) algorithms with a focus on robustness metrics and multi-objective optimization.

## Overview

This project provides tools for:
- Evaluating neural architectures on NASBench-201 dataset
- Analyzing architecture robustness against various adversarial attacks
- Visualizing and comparing architecture performance metrics
- Computing correlation between different architecture metrics
- Multi-objective optimization of neural architectures using pymoo

## Project Structure

```
RobustNASBench/
├── algorithms/         # Implementation of search algorithms
├── config/            # Configuration files
├── data/              # Dataset and benchmark data
├── helpers/           # Utility functions and helper classes
├── operators/         # Search space operators
├── problems/          # Problem definitions
├── results/           # Experiment results
├── src/               # Core source code
├── synthesis_result/  # Synthesized architecture results
├── visualize.py       # Visualization tools
├── visualize_bokeh.py # Interactive visualization using Bokeh
├── get_stats_mo.py    # Multi-objective statistics collection
├── get_stats_so.py    # Single-objective statistics collection
└── constant.py        # Constants and configuration
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/RobustNASBench.git
cd RobustNASBench
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 🧬 Genetic Algorithm (GA)

1. **Run GA**
   - Use the `main.py` script with the following parameters:
     - `--problem`: choose one of `"SO-NAS201-1"`, `"SO-NAS201-2"`, or `"SO-NAS201-3"`.
     - `--algorithm_name`: set to `"GA"`.
   - For detailed information, refer to the [`constant.py`](./constant.py) file.

2. **Configure Dataset**
   - In the `.env` file, change the `NAS_ROBBENCH` variable to point to the dataset you want to use.
     Example:
     ```env
     NAS_ROBBENCH=config/cifar10
     ```
     if you want to use the NAS-RobBench-201 metrics on CIFAR-10.

3. **Save Results**
   - Results after running are saved at the path specified by `--path_results`.
   - For example, if you run with `--path_results results`, the `results/` directory will contain the outputs.

4. **Collect Result Metrics**
   - Open the file [`get_stats_so.py`](./get_stats_so.py) and set the `base_path` variable to point to your results directory (e.g., `base_path="results"`).
   - This script will automatically collect all GA metrics.

5. **Note**
   - You can run all 11 search metrics, save everything to the same `results/` directory, then collect data at once, instead of collecting individually.

---

### ♻️ NSGA-II (Multi-Objective)

1. **Run NSGA-II**
   - Use the `main.py` script with the following parameters:
     - `--problem`: choose one of `"MO-NAS201-1"`, `"MO-NAS201-2"`, or `"MO-NAS201-3"`.
     - `--algorithm_name`: set to `"NSGA-II"`.
     - **Recommendation**: keep `--objective=1` as default, no need to care about other values as results remain nearly unchanged.

2. **Collect Results**
   - After running, open the file [`get_stats_mo.py`](./get_stats_mo.py) and set the `base_path` variable to point to the directory containing results.
   - Don't forget to edit the dataset names at the beginning of this file to match your experiment.

---



## Configuration

The project uses several configuration files:
- `config/cifar10.json`: Configuration for CIFAR-10 experiments
- `config/imagenet.json`: Configuration for ImageNet experiments
- `constant.py`: Global constants and configuration

## Results

Results are stored in:
- `results/`: Main experiment results
- `synthesis_result/`: Synthesized architecture results
- `pareto_front/`: Pareto front results for multi-objective optimization

