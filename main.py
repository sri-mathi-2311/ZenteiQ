"""
main.py
-------
Single entry point that runs the full pipeline in order:
    1. Data generation   (src/data_generation.py)
    2. Preprocessing     (src/preprocessing.py)
    3. Training          (src/train.py)
    4. Evaluation        (src/evaluate.py)

Run with:  python3 main.py
(Run from the project root, i.e. the folder containing this file.)
"""

import subprocess
import sys

STEPS = [
    ("Generating synthetic character dataset", "src/data_generation.py"),
    ("Preprocessing and splitting data", "src/preprocessing.py"),
    ("Training the from-scratch NumPy neural network", "src/train.py"),
    ("Evaluating on the test set", "src/evaluate.py"),
]


def run():
    """Execute each pipeline stage as a subprocess, in order, and stop on error."""
    for description, script in STEPS:
        print("\n" + "=" * 70)
        print(f"STEP: {description}  ({script})")
        print("=" * 70)
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"Pipeline stopped: {script} exited with code {result.returncode}")
            sys.exit(result.returncode)
    print("\nPipeline complete. See outputs/ for weights, plots, and metrics.")


if __name__ == "__main__":
    run()
