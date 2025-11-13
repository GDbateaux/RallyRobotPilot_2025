"""
Run manager for organizing GA training outputs.

Creates a clean directory structure per run:
ga_runs/
  └── run_TIMESTAMP/
      ├── config.txt          # Configuration used
      ├── plots/              # Visualization plots
      ├── controls/           # Evolved control sequences
      └── training.log        # Training log
"""

from pathlib import Path
from datetime import datetime
import json


class RunManager:
    """
    Manages directory structure and file organization for a GA training run.

    Creates a timestamped directory for each run with subdirectories for
    plots, controls, and logs.
    """

    def __init__(self, base_dir="ga_runs"):
        """
        Initialize run manager and create directory structure.

        Args:
            base_dir (str): Base directory for all runs (default: "ga_runs")
        """
        self.base_dir = Path(base_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create run directory
        self.run_dir = self.base_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.plots_dir = self.run_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

        self.controls_dir = self.run_dir / "controls"
        self.controls_dir.mkdir(exist_ok=True)

        # Log file path
        self.log_path = self.run_dir / "training.log"

    def get_plot_path(self, name):
        """
        Get path for saving a plot.

        Args:
            name (str): Plot name (e.g., "segment_0", "full_lap_summary")

        Returns:
            Path: Full path to save plot (with .png extension)
        """
        return self.plots_dir / f"{name}.png"

    def get_controls_path(self, name):
        """
        Get path for saving evolved controls.

        Args:
            name (str): Controls name (e.g., "segment_0", "evolved_full_lap")

        Returns:
            Path: Full path to save controls (with .npz extension)
        """
        return self.controls_dir / f"{name}.npz"

    def get_log_path(self):
        """
        Get path for training log file.

        Returns:
            Path: Full path to log file
        """
        return self.log_path

    def save_config(self, config_dict):
        """
        Save configuration used for this run.

        Args:
            config_dict (dict): Configuration parameters
        """
        config_path = self.run_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        # Also save human-readable text version
        config_txt_path = self.run_dir / "config.txt"
        with open(config_txt_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("GENETICS ALGORITHM CONFIGURATION\n")
            f.write("="*70 + "\n\n")
            for key, value in config_dict.items():
                f.write(f"{key:30s}: {value}\n")

        print(f"✓ Configuration saved: {config_txt_path}")

    def save_summary(self, results):
        """
        Save training summary to text file.

        Args:
            results (dict): Training results dictionary
        """
        summary_path = self.run_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("TRAINING SUMMARY\n")
            f.write("="*70 + "\n\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Total segments trained: {len(results['segment_results'])}\n")
            f.write(f"Original total frames: {results['total_original_frames']}\n")
            f.write(f"Evolved total frames: {results['total_evolved_frames']}\n")
            f.write(f"Total improvement: {results['total_improvement']} frames ")
            f.write(f"({results['improvement_percentage']:.1f}%)\n\n")

            f.write("Per-segment results:\n")
            f.write(f"{'Segment':>8} | {'Original':>8} | {'Evolved':>8} | {'Improvement':>12} | {'Fitness':>8}\n")
            f.write(f"{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*12}-+-{'-'*8}\n")

            for result in results['segment_results']:
                if result['original_length'] is not None and result['original_length'] > 0:
                    imp_pct = (result['improvement'] / result['original_length'] * 100)
                    f.write(f"{result['segment_id']:8d} | {result['original_length']:8d} | {result['evolved_length']:8d} | "
                           f"{result['improvement']:+4d} ({imp_pct:+5.1f}%) | {result['best_fitness']:8.2f}\n")
                else:
                    # Evolved segment - no comparison
                    f.write(f"{result['segment_id']:8d} | {'N/A':>8} | {result['evolved_length']:8d} | "
                           f"{'evolved':>17} | {result['best_fitness']:8.2f}\n")

        print(f"✓ Summary saved: {summary_path}")

    def __str__(self):
        """String representation showing directory structure."""
        return f"""RunManager:
  Run directory: {self.run_dir}
  - Plots: {self.plots_dir}
  - Controls: {self.controls_dir}
  - Log: {self.log_path}
"""


# Quick test
if __name__ == "__main__":
    # Create test run manager
    rm = RunManager()
    print(rm)

    # Test paths
    print("\nExample paths:")
    print(f"Segment 0 plot: {rm.get_plot_path('segment_0')}")
    print(f"Segment 0 controls: {rm.get_controls_path('segment_0')}")
    print(f"Training log: {rm.get_log_path()}")

    # Test config save
    test_config = {
        'max_segments': 1,
        'population_size': 5,
        'num_generations': 5,
        'mutation_rate': 0.25,
        'use_headless': True
    }
    rm.save_config(test_config)

    # Test summary save
    test_results = {
        'segment_results': [
            {'segment_id': 0, 'original_length': 120, 'evolved_length': 110, 'improvement': 10, 'best_fitness': 9890}
        ],
        'total_original_frames': 120,
        'total_evolved_frames': 110,
        'total_improvement': 10,
        'improvement_percentage': 8.3
    }
    rm.save_summary(test_results)

    print(f"\n✓ Test run created in: {rm.run_dir}")
