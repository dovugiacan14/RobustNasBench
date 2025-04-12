import os
import pickle
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, List
from constant import attack_method
from collections import defaultdict 
from bokeh.plotting import figure, output_file, show, save
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.models import Band, ColumnDataSource, Legend


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=str,
        default="results/SO-NAS201-1/SO-NAS201-1_GA_synflow_1104134422",
    )
    parser.add_argument(
        "--attack", type=int, default=0, help="type of attack", choices=range(0, 6)
    )
    parser.add_argument(
        "--option", type=int, default=0, help="type of attack", choices=range(0, 6)
    )
    return parser.parse_args()


def collect_acc_statistics(base_result_dir: str, best_arch_filename: str):
    """
    Collect mean and std of validation and test accuracy from experiment folders.

    Args:
        base_result_dir (str): Path to the base result directory.
        best_arch_filename (str): Filename containing architecture accuracy results.

    Returns:
        Tuple containing:
            - val_acc_mean_each_exp (List[float])
            - val_acc_std_each_exp (List[float])
            - test_acc_mean_each_exp (List[float])
            - test_acc_std_each_exp (List[float])
    """
    # convert to Path object
    base_result_dir = Path(base_result_dir)

    # initialize output lists
    val_acc_best_each_exp = []
    val_acc_mean_each_exp = []
    val_acc_std_each_exp = []

    test_acc_best_each_exp = []
    test_acc_mean_each_exp = []
    test_acc_std_each_exp = []

    if not base_result_dir.exists():
        raise FileNotFoundError(f"Base result directory not found: {base_result_dir}")

    # iterate over each subdirectory
    for subdir_name in os.listdir(base_result_dir):
        val_acc_each_exp = []
        test_acc_each_exp = []

        subdir_path = base_result_dir / subdir_name
        file_path = subdir_path / best_arch_filename.strip()

        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue

        with open(file_path, "rb") as f:
            data = pickle.load(f)

        # extract accuracy values
        nGens_each_run = data[1]
        for gen in nGens_each_run:
            val_acc = gen[0]["validation_accuracy"]
            test_acc = gen[0]["testing_accuracy"]
            val_acc_each_exp.append(val_acc)
            test_acc_each_exp.append(test_acc)

        # convert to numpy for stats
        val_acc_each_exp = np.array(val_acc_each_exp)
        test_acc_each_exp = np.array(test_acc_each_exp)

        val_acc_best_each_exp.append(np.max(val_acc_each_exp))
        val_acc_mean_each_exp.append(np.mean(val_acc_each_exp))
        val_acc_std_each_exp.append(np.std(val_acc_each_exp))

        test_acc_best_each_exp.append(np.max(test_acc_each_exp))
        test_acc_mean_each_exp.append(np.mean(test_acc_each_exp))
        test_acc_std_each_exp.append(np.std(test_acc_each_exp))

    return {
        "max_val": val_acc_best_each_exp,
        "mean_val": val_acc_mean_each_exp, 
        "std_val": val_acc_std_each_exp,
        "max_test": test_acc_best_each_exp, 
        "mean_test": test_acc_mean_each_exp, 
        "std_test": test_acc_std_each_exp  
    }

def collect_total_eval_attacking(base_dir: str, best_arch_filename: str): 
    results = defaultdict(list)
    results = defaultdict(list)
    for attack in os.listdir(base_dir):
        attack_dir = os.path.join(base_dir, attack)
        for folder in os.listdir(attack_dir):
            folder_path = os.path.join(attack_dir, folder)
            results[attack] = collect_acc_statistics(folder_path, best_arch_filename)
    return results

def visualize_best_attack_result(test_acc_max: list, test_acc_std: list, attack_name: str): 
    """Visualize best test accuracy across attack methods using Bokeh."""
    # prepare data 
    x = list(range(len(test_acc_max)))
    max_acc = np.array(test_acc_max)
    std_acc = np.array(test_acc_std)

    # create destination directory
    os.makedirs("htmls", exist_ok=True)
    save_path = os.path.join("htmls", f"{attack_name}.html")
    output_file(save_path, title=f"{attack_name} Visualization")

    # create source data for Bokeh 
    source = ColumnDataSource(data={
        'x': x,
        'max_acc': max_acc,
        'std_acc': std_acc,
        'lower': max_acc - std_acc,
        'upper': max_acc + std_acc
    })

    # create figure and touchative tool
    p = figure(
        title=f"{attack_name}",
        x_axis_label="n_runs",
        y_axis_label="Test Accuracy",
        width=800, 
        height=400,
        tools="pan,box_zoom,wheel_zoom,reset,save",
        toolbar_location="above"
    )

    # UI settings 
    p.title.text_font_size = "16pt"
    p.title.align = "center"
    p.xaxis.axis_label_text_font_size = "12pt"
    p.yaxis.axis_label_text_font_size = "12pt"
    p.xgrid.grid_line_color = "gray"
    p.ygrid.grid_line_color = "gray"

    p.xgrid.grid_line_dash = [6, 4]
    p.ygrid.grid_line_dash = [6, 4]
    p.xgrid.grid_line_alpha = 0.3
    p.ygrid.grid_line_alpha = 0.3

    # fill between 
    p.varea(
        x= "x", 
        y1= "lower", 
        y2= "upper", 
        source= source, 
        fill_color= "navajowhite", 
        fill_alpha= 0.5, 
        legend_label= "Accuracy ± Std"
    )

    # plot line 
    p.line(
        x='x',
        y='max_acc',
        source=source,
        line_color="darkorange",
        line_width=2,
        legend_label="Highest Test Accuracy"
    )

    # plot scatter 
    p.scatter(
        x='x',
        y='max_acc',
        source=source,
        size=8,
        color="orangered",
        marker="square",
        legend_label="Points"
    )

    # add tooltip when hover 
    hover = HoverTool()
    hover.tooltips = [
        ("Run", "@x"),
        ("Accuracy", "@max_acc{0.0000}"),
        ("Std", "@std_acc{0.0000}")
    ]
    p.add_tools(hover)

    # legend settings 
    p.legend.location = "bottom_right" 
    p.legend.click_policy = "hide"
    p.legend.background_fill_alpha = 0.8
    p.legend.border_line_color = "black"
    p.legend.border_line_alpha = 0.5

    show(p)
    print(f"Plot saved to {save_path}")

def visualize_synthetic_data(data_dict, save_dir: str= "htmls/"):
    """Visualize multiple test accuracy results with standard deviation as fill area using Bokeh."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "accuracy_comparison.html")
    
    # Default colors if none provided
    default_colors = {
        'val_fgsm_3.0_acc': {'fill': 'peachpuff', 'line': 'coral', 'point': 'darkred'},
        'val_fgsm_8.0_acc': {'fill': 'navajowhite', 'line': 'darkorange', 'point': 'orangered'},
        'val_pgd_3.0_acc': {'fill': 'lightblue', 'line': 'dodgerblue', 'point': 'navy'},
        'val_pgd_8.0_acc': {'fill': 'lightcyan', 'line': 'teal', 'point': 'darkcyan'},
        'autoattack': {'fill': 'lightgreen', 'line': 'forestgreen', 'point': 'darkgreen'},
        'val_acc': {'fill': 'lavender', 'line': 'purple', 'point': 'darkmagenta'}
    }
    
    # Use provided colors or default
    colors = default_colors
    
    # Create a Bokeh figure
    p = figure(
        width=800, height=400,
        x_axis_label="Run Index",
        y_axis_label="Test Accuracy",
        tools="pan,box_zoom,reset,save,wheel_zoom"  # Enable interactive tools
    )
    
    # Iterate through each key in the data dictionary
    for key in data_dict.keys():
        # Prepare data
        acc_values = np.array(data_dict[key]['max_test'])
        std_values = np.array(data_dict[key]['std_test'])
        x = list(range(len(acc_values)))
        
        # create ColumnDataSource for Bokeh
        source = ColumnDataSource(data={
            'x': x,
            'acc': acc_values,
            'std': std_values,
            'lower': acc_values - std_values,
            'upper': acc_values + std_values,
            'key': [key] * len(x)  # For hover tool
        })
        
        # add Band for standard deviation
        band = Band(
            base='x', lower='lower', upper='upper', source=source,
            fill_color=colors.get(key, default_colors['val_acc'])['fill'],
            fill_alpha=0.5
        )
        p.add_layout(band)
        
        # line plot for accuracy
        p.line(
            x='x', y='acc', source=source,
            line_color=colors.get(key, default_colors['val_acc'])['line'],
            line_width=2,
            legend_label=f"{key}"
        )
        
        # scatter plot with square markers
        p.scatter(
            x='x', y='acc', source=source,
            fill_color=colors.get(key, default_colors['val_acc'])['point'],
            size=8,
            marker="square",
            legend_label=f"{key} (points)"
        )
        
        # add hover tool for scatter points
        hover = HoverTool(
            tooltips=[
                ("Attack", "@key"),
                ("Run Index", "@x"),
                ("Test Accuracy", "@acc{0.0000}"),
                ("Std", "@std{0.0000}")
            ],
            mode='mouse',
            point_policy='snap_to_data'
        )
        p.add_tools(hover)
    
    # customize plot
    p.grid.grid_line_dash = [6, 4]
    p.grid.grid_line_alpha = 0.6
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"  # Allow toggling lines on/off
    p.legend.label_text_font_size = "8pt"  # Smaller legend text for clarity
    
    # save the plot as HTML
    output_file(save_path)
    save(p)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    best_arch_filename = "best_architecture_each_gen.p"
    args = parse_arguments()
    if args.option == 1: 
        _, _, test_best, test_std = collect_acc_statistics(
            args.base_dir, best_arch_filename
        )
        visualize_best_attack_result(
            test_acc_max= test_best, 
            test_acc_std= test_std, 
            attack_name= attack_method[args.attack], 
        )
    elif args.option == 2: 
        statistics = collect_total_eval_attacking("results/", best_arch_filename)
        visualize_synthetic_data(statistics)