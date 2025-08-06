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
## 🧬 Genetic Algorithm (GA)

1. **Chạy GA**  
   - Sử dụng script `main.py` với tham số:
     - `--problem`: chọn một trong các giá trị `"SO-NAS201-1"`, `"SO-NAS201-2"` hoặc `"SO-NAS201-3"`.
     - `--algorithm_name`: đặt là `"GA"`.
   - Thông tin chi tiết có thể tham khảo trong file [`constant.py`](./constant.py).

2. **Cấu hình Dataset**  
   - Trong file `.env`, thay đổi giá trị biến `NAS_ROBBENCH` để trỏ tới dataset bạn muốn sử dụng.  
     Ví dụ:
     ```env
     NAS_ROBBENCH=config/cifar10
     ```
     nếu bạn muốn dùng bộ chỉ số NAS-RobBench-201 trên CIFAR-10.

3. **Lưu kết quả**  
   - Kết quả sau khi chạy được lưu tại đường dẫn chỉ định bởi `--path_results`.  
   - Ví dụ, nếu bạn chạy với `--path_results results`, thư mục `results/` sẽ chứa các output.

4. **Thu thập chỉ số kết quả**  
   - Mở file [`get_stats_so.py`](./get_stats_so.py) và đặt biến `base_path` trỏ đến thư mục kết quả của bạn (ví dụ: `base_path="results"`).
   - Script này sẽ tự động thu thập các chỉ số của GA.

5. **Lưu ý**  
   - Bạn có thể chạy hết 11 chỉ số search metric, lưu tất cả vào cùng thư mục `results/`, rồi mới thu thập dữ liệu một lần, thay vì thu thập riêng lẻ.

---

## ♻️ NSGA-II (Multi-Objective)

1. **Chạy NSGA-II**  
   - Sử dụng script `main.py` với tham số:
     - `--problem`: chọn một trong các giá trị `"MO-NAS201-1"`, `"MO-NAS201-2"` hoặc `"MO-NAS201-3"`.
     - `--algorithm_name`: đặt là `"NSGA-II"`.
     - **Khuyến nghị**: để mặc định `--objective=1`, không cần quan tâm đến các giá trị khác vì kết quả gần như không đổi.

2. **Thu thập kết quả**  
   - Sau khi chạy xong, mở file [`get_stats_mo.py`](./get_stats_mo.py) và đặt biến `base_path` trỏ đến thư mục chứa kết quả.
   - Đừng quên chỉnh sửa tên các dataset ở phần đầu file này cho phù hợp với thí nghiệm của bạn.

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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
