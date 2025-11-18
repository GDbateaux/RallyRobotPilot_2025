#!/bin/bash
#SBATCH --job-name=ga_mpi
#SBATCH --nodes=6                    # 2 nodes for distributed MPI
#SBATCH --ntasks-per-node=48         # 32 MPI processes per node
#SBATCH --cpus-per-task=1            # 1 CPU per MPI task (sequential within each rank)
#SBATCH --nodelist=calypso[3-8]      # Use idle nodes with same specs (64 CPUs each)
#SBATCH --time=01:00:00              # Max runtime (adjust based on NUM_GENERATIONS)
#SBATCH --output=out/ga_mpi_%j.out       # Output file (includes job ID)
#SBATCH --error=out/ga_mpi_%j.err        # Error file (includes job ID)

# ============================================================================
# SLURM Configuration Summary
# ============================================================================
# MULTI-NODE CONFIGURATION
# Total MPI processes: 4 nodes × 48 tasks = 128 MPI ranks
# Total parallel workers: 128 MPI ranks × 1 CPU = 128 workers
# Each rank evaluates sequentially (no multiprocessing within rank)
#
# With 128 individuals: each rank gets 1 individual (128 ÷ 128 = 1)
# ============================================================================

echo "=============================================================="
echo "SLURM Job Configuration"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Allocated nodes: $SLURM_JOB_NODELIST"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Total MPI processes: $SLURM_NTASKS"
echo "Total parallel workers: $SLURM_NTASKS"
echo "Start time: $(date)"
echo "=============================================================="
echo ""

# Activate virtual environment (if using one)
# source .venv/bin/activate

# Change to project directory
#cd $SLURM_SUBMIT_DIR

# Run MPI program with network interface configuration
echo "Launching MPI genetic algorithm..."
# MPI configuration:
# - btl_tcp_if_include eno1: ONLY use eno1 interface (avoids docker/k8s networks)
# - oob_tcp_if_include eno1: ONLY use eno1 for out-of-band communication
# - btl ^openib: Don't use InfiniBand
# - map-by ppr:32:node: Map 32 processes per node
mpirun -np $SLURM_NTASKS \
       --mca btl_tcp_if_include eno1 \
       --mca oob_tcp_if_include eno1 \
       --mca btl ^openib \
       --map-by ppr:48:node \
       python3 scripts_mpi/ga_mpi.py

echo ""
echo "=============================================================="
echo "Job completed: $(date)"
echo "=============================================================="

