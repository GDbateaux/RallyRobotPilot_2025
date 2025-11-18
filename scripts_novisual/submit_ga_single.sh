#!/bin/bash
#SBATCH --job-name=ga_single
#SBATCH --nodes=1                    # Single node
#SBATCH --ntasks=32                   # Single task (no MPI)
#SBATCH --cpus-per-task=1           # Use 20 CPUs for multiprocessing
#SBATCH --time=01:00:00              # Max runtime
#SBATCH --output=out/ga_single_%j.out
#SBATCH --error=out/ga_single_%j.err

# ============================================================================
# SLURM Configuration Summary
# ============================================================================
# Single node, single process (no MPI)
# Uses multiprocessing within Python (20 CPUs)
# ============================================================================

echo "=============================================================="
echo "SLURM Job Configuration"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "=============================================================="
echo ""

# Activate virtual environment (done manually before sbatch)
# source .venv/bin/activate
echo "Python: $(which python3)"
echo "Python version: $(python3 --version)"
echo ""

# Run the genetic algorithm
echo "Launching genetic algorithm (non-MPI version)..."
python3 scripts_novisual/ga_step_by_step.py

echo ""
echo "=============================================================="
echo "Job completed: $(date)"
echo "=============================================================="