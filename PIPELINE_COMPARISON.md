# Pipeline Comparison: ga_step_by_step.py vs ga_mpi.py

## Architecture Overview

### ga_step_by_step.py (Original - Single Node)
```
Main Process
  ├─ Load data, create checkpoints
  ├─ Create CollisionSystem (ONE instance in main)
  ├─ Create worker pool (if NUM_WORKERS > 1)
  │   └─ Each worker: _init_worker() creates NEW CollisionSystem
  └─ Evolution loop
      └─ evaluate_population_parallel()
          ├─ If NUM_WORKERS == 1: Use main's collision_system
          └─ If NUM_WORKERS > 1: Workers use their own collision_system
```

### ga_mpi.py (MPI Version - Multi Node)
```
Rank 0 (Master)
  ├─ Load data, create checkpoints
  ├─ Broadcast to all ranks
  └─ Evolution loop
      └─ evaluate_population_mpi()

All Ranks (0-N)
  ├─ Receive broadcast data
  ├─ Create CollisionSystem (ONE per rank) ⚠️
  └─ For each generation:
      ├─ Receive genome chunk
      ├─ If cpus_per_rank == 1: Use rank's collision_system
      └─ If cpus_per_rank > 1: Create worker pool
          └─ Each worker: _init_worker() creates NEW CollisionSystem
```

## Key Difference: CollisionSystem Initialization

### ga_step_by_step.py
- **Total CollisionSystem instances**: 1 (sequential) or NUM_WORKERS (parallel)
- **Created**: In main OR in worker processes (not both)
- **Example**: NUM_WORKERS=20 → 20 CollisionSystem instances

### ga_mpi.py
- **Total CollisionSystem instances**:
  - If cpus_per_rank=1: size instances (one per MPI rank)
  - If cpus_per_rank>1: size + (size × cpus_per_rank) instances
- **Created**: In every MPI rank + in worker processes (if cpus_per_rank>1)
- **Example**:
  - 32 MPI ranks × cpus_per_rank=1 → 32 CollisionSystem instances ✓
  - 32 MPI ranks × cpus_per_rank=4 → 32 + (32×4) = 160 CollisionSystem instances ⚠️

## Collision System Creation Locations

### ga_step_by_step.py
```python
# Location 1: Main process (line 161)
collision_system = CollisionSystem(TRACK_METADATA_PATH)

# Location 2: Worker processes (line 47, _init_worker)
_collision_system = CollisionSystem(track_path)

# Usage in evaluate_population_parallel:
if num_workers == 1:
    # Use main's collision_system (no new instances)
    sim_result = simulate_individual(genome, collision_system)
else:
    # Workers create their own (Location 2)
    results = pool.map(_evaluate_individual_worker, genomes)
```

### ga_mpi.py
```python
# Location 1: Every MPI rank (line 334)
collision_system = CollisionSystem(TRACK_METADATA_PATH)

# Location 2: Worker processes within each rank (line 95, _init_worker)
_collision_system = CollisionSystem(track_path)

# Usage in evaluate_population_mpi:
if cpus_per_rank > 1:
    # Workers create their own (Location 2)
    with ctx.Pool(processes=cpus_per_rank, initializer=_init_worker, ...):
        local_results = pool.map(_evaluate_individual_worker, local_genomes)
else:
    # Use rank's collision_system (Location 1)
    sim_result = simulate_individual(genome, collision_system)
```

## Problem Identified

### Issue: Panda3D ShowBase Multiple Instance Problem

**Panda3D limitation**: Cannot create multiple ShowBase instances in the same Python interpreter
- CollisionSystem internally uses Panda3D's ShowBase
- Error: "Exception: Attempt to spawn multiple ShowBase instances!"

**Why it works on local machine (12 ranks)**:
- Each MPI rank is a separate process with its own Python interpreter
- 12 processes × 1 CollisionSystem each = 12 independent instances ✓

**Why it might fail on HPC (32 ranks)**:
- Possible resource contention when 32 processes simultaneously initialize Panda3D
- Potential display/graphics issues even in headless mode
- File descriptor limits or shared memory issues

### Potential Issues on HPC

1. **Simultaneous initialization**: All 32 MPI ranks try to create CollisionSystem at the same time
   - May overwhelm shared resources (file descriptors, shared memory, etc.)
   - Panda3D might not handle concurrent initialization well

2. **Display/Graphics issues**: Even in headless mode, Panda3D might try to access display resources
   - HPC nodes typically have no X11/display
   - Need to ensure proper headless operation

3. **Resource limits**: HPC may have stricter resource limits
   - File descriptor limits
   - Memory limits per process
   - Shared memory limits

## Verification Steps

### 1. Check if all ranks reach collision system initialization
Look for this output in error file:
```bash
cat ga_mpi_JOBID.err
```

### 2. Check if initialization is synchronized
Current code has no barrier between broadcast and collision system creation:
```python
# Step 4: Broadcast
setup_data = broadcast_setup_data(...)
# No barrier here!
# Step 5: All ranks create collision system immediately
collision_system = CollisionSystem(TRACK_METADATA_PATH)
```

### 3. Test with reduced ranks
Try with fewer MPI ranks to see if it's a resource contention issue:
```bash
#SBATCH --ntasks-per-node=4  # Instead of 32
```

## Recommended Fixes

### Fix 1: Add synchronization barriers
```python
# After broadcast
comm.Barrier()  # Ensure all ranks have data before proceeding

# Before collision system creation
if is_master(rank):
    print("Step 5: Initializing collision system on all ranks...")
comm.Barrier()  # Synchronized start

# Suppress output on workers
if not is_master(rank):
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

collision_system = CollisionSystem(TRACK_METADATA_PATH)

if not is_master(rank):
    sys.stdout = old_stdout

comm.Barrier()  # Ensure all ranks complete before proceeding
```

### Fix 2: Sequential initialization across ranks
Instead of all ranks creating CollisionSystem simultaneously, do it one at a time:
```python
# Initialize collision systems sequentially (one rank at a time)
for i in range(size):
    if rank == i:
        if is_master(rank):
            print(f"Step 5: Initializing collision system on rank {rank}...")
        collision_system = CollisionSystem(TRACK_METADATA_PATH)
    comm.Barrier()  # Wait for current rank to finish before next rank starts
```

### Fix 3: Master-only collision system (for sequential path)
If cpus_per_rank=1, only master needs collision system (for sequential evaluation):
```python
if cpus_per_rank == 1:
    # Only master creates collision system (workers won't use it)
    if is_master(rank):
        collision_system = CollisionSystem(TRACK_METADATA_PATH)
    else:
        collision_system = None
else:
    # All ranks need collision system (for multiprocessing pool initialization)
    collision_system = CollisionSystem(TRACK_METADATA_PATH)
```

## Testing Strategy

1. **Test on local SLURM first**: Verify with 12 ranks (already works)
2. **Test on HPC with reduced ranks**: Try 4, 8, 16 ranks
3. **Check error output**: Look for Panda3D or initialization errors
4. **Add debug output**: Print when each rank completes initialization
5. **Monitor resource usage**: Check file descriptors, memory, etc.

## Current Status

✅ Works on local machine (12 MPI ranks)
❌ Fails on HPC (32 MPI ranks) - exits immediately after ursina imports
🔍 Need to check error file for actual failure reason
