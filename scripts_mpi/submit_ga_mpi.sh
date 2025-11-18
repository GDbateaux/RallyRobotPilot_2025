#!/bin/bash
#SBATCH --job-name=ga_mpi
#SBATCH --nodes=1                    # SINGLE NODE TESTING
#SBATCH --ntasks-per-node=12         # 12 MPI processes on single node
#SBATCH --cpus-per-task=1            # 1 CPU per MPI task (sequential within each rank)
#SBATCH --time=01:00:00              # Max runtime (adjust based on NUM_GENERATIONS)
#SBATCH --output=ga_mpi_%j.out       # Output file (includes job ID)
#SBATCH --error=ga_mpi_%j.err        # Error file (includes job ID)

# ============================================================================
# SLURM Configuration Summary
# ============================================================================
# SINGLE NODE TEST CONFIGURATION
# Total MPI processes: 1 node × 12 tasks = 12 MPI ranks
# Total parallel workers: 12 MPI ranks × 1 CPU = 12 workers
# Each rank evaluates sequentially (no multiprocessing within rank)
#
# With 60 individuals: each rank gets 5 individuals (60 ÷ 12 = 5)
# ============================================================================

echo "=============================================================="
echo "SLURM Job Configuration"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_TASKS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Total MPI processes: $(($SLURM_JOB_NUM_NODES * $SLURM_TASKS_PER_NODE))"
echo "Total parallel workers: $(($SLURM_JOB_NUM_NODES * $SLURM_TASKS_PER_NODE * $SLURM_CPUS_PER_TASK))"
echo "Start time: $(date)"
echo "=============================================================="
echo ""

# Activate virtual environment (if using one)
# source .venv/bin/activate

# Change to project directory
#cd $SLURM_SUBMIT_DIR

# Run MPI program
echo "Launching MPI genetic algorithm..."
mpirun python3 scripts_mpi/ga_mpi.py

echo ""
echo "=============================================================="
echo "Job completed: $(date)"
echo "=============================================================="