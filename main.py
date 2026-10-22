import os
import sys

# Ensure project root and src/ are in Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from run_model_evaluation import main as run_evaluation

if __name__ == "__main__":
    run_evaluation()
