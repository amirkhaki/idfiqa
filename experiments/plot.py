"""Plotting utility experiment."""
import csv
import os
from .registry import ExperimentBase, register_experiment

@register_experiment
class PlotExperiment(ExperimentBase):
    name = "plot"
    description = "Plot a scatter plot from a result CSV file and save it as an image"

    def add_arguments(self, parser):
        parser.add_argument("csv_file", type=str, help="Path to the result CSV file")

    def run(self, args, datasets, num_workers, force, device):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is not installed. Please install it to generate plots:")
            print("  pip install matplotlib")
            return

        if not os.path.exists(args.csv_file):
            print(f"Error: CSV file '{args.csv_file}' not found.")
            return

        with open(args.csv_file, newline="") as f:
            rows = list(csv.DictReader(f))
        
        if not rows:
            print(f"File {args.csv_file} is empty or invalid.")
            return

        # Attempt to find standard columns
        x_col, y_col = None, None
        if "mos_label" in rows[0]:
            x_col = "mos_label"
            
        if "score" in rows[0]:
            y_col = "score"
        elif "global_gram" in rows[0]:
            # fallback for geometric robustness csv
            y_col = "global_gram"
            
        if not x_col or not y_col:
            print(f"Could not find appropriate columns in the CSV. Available: {list(rows[0].keys())}")
            return

        x = [float(r[x_col]) for r in rows]
        y = [float(r[y_col]) for r in rows]

        from .utils import compute_metrics
        srcc, plcc = compute_metrics(y, x)

        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, alpha=0.6, color="blue", edgecolors="black")
        
        title_text = f"Scatter plot of {os.path.basename(args.csv_file)}\n"
        title_text += f"SRCC: {srcc:.4f} | PLCC: {plcc:.4f}"
        
        plt.title(title_text)
        plt.xlabel("Ground Truth (MOS)")
        plt.ylabel("Predicted Score")
        plt.grid(True, linestyle="--", alpha=0.7)
        
        out_path = args.csv_file.replace(".csv", "_scatter.png")
        if out_path == args.csv_file:
            out_path += ".png"
            
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        
        print(f"Scatter plot saved to: {out_path}")
