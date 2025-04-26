# RobustNASBench

A framework for evaluating and analyzing neural architecture search (NAS) algorithms with a focus on robustness metrics.

## Overview

This project provides tools for:
- Evaluating neural architectures on NASBench-201 dataset
- Analyzing architecture robustness against various adversarial attacks
- Visualizing and comparing architecture performance metrics
- Computing correlation between different architecture metrics

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

## Key Features

1. **Multi-Objective Analysis**
   - Evaluate architectures on multiple objectives (accuracy, robustness)
   - Collect and analyze statistics across different metrics
   - Export results to Excel with multiple sheets

2. **Robustness Evaluation**
   - Support for various adversarial attacks (FGSM, PGD)
   - Autoattack evaluation
   - Robust validation accuracy metrics

3. **Visualization Tools**
   - Generation-wise accuracy plots
   - Comparison charts
   - Interactive visualizations using Bokeh

4. **Correlation Analysis**
   - Compute correlations between different architecture metrics
   - Analyze relationships between robustness and other metrics

## Usage

### Collecting Statistics

1. **Multi-Objective Statistics**
```python
python get_stats_mo.py
```
This will process results from the `results/` directory and generate an Excel file with multiple sheets containing statistics for different metrics.

2. **Single-Objective Statistics**
```python
python get_stats_so.py
```
This will collect and analyze single-objective experiment results.

### Visualization

1. **Basic Visualization**
```python
python visualize.py
```
This will generate various plots and charts for analyzing architecture performance.

2. **Interactive Visualization**
```python
python visualize_bokeh.py
```
This will create interactive visualizations using Bokeh.

## Dependencies

- Python 3.x
- pandas
- numpy
- openpyxl
- matplotlib
- bokeh
- scipy

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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Add your license information here] 