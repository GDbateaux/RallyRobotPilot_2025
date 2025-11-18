"""
MPI Utilities for Distributed Genetic Algorithm

Provides helper functions for MPI-based distributed computation:
- MPI initialization and environment detection
- Broadcasting setup data to all workers
- Scattering population chunks across MPI ranks
- Gathering evaluation results from workers

Design:
- Master (rank 0): Coordinates work, creates generations, collects results
- Workers (rank 1-N): Receive work, evaluate individuals, return results
"""

from mpi4py import MPI
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


def init_mpi() -> Tuple[MPI.Comm, int, int]:
    """
    Initialize MPI environment and get communication info.

    Returns:
        comm: MPI communicator (MPI.COMM_WORLD)
        rank: Process rank (0 = master, 1-N = workers)
        size: Total number of MPI processes
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    return comm, rank, size


def is_master(rank: int) -> bool:
    """
    Check if current process is the master.

    Args:
        rank: MPI process rank

    Returns:
        True if rank 0 (master), False otherwise
    """
    return rank == 0


def broadcast_setup_data(comm: MPI.Comm, rank: int,
                         checkpoints: Optional[List] = None,
                         genome_length: Optional[int] = None,
                         track_path: Optional[str] = None,
                         finish_line_checkpoint_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Broadcast all setup data from master to workers.

    Master provides data, workers receive it.

    Args:
        comm: MPI communicator
        rank: Process rank
        checkpoints: Checkpoint data (master only)
        genome_length: Length of genome (master only)
        track_path: Path to track metadata (master only)
        finish_line_checkpoint_id: ID of finish line checkpoint (master only)

    Returns:
        Dictionary with all setup data (valid on all ranks)
    """
    if is_master(rank):
        # Master: package data
        data = {
            'checkpoints': checkpoints,
            'genome_length': genome_length,
            'track_path': track_path,
            'finish_line_checkpoint_id': finish_line_checkpoint_id
        }
    else:
        # Workers: receive data
        data = None

    # Broadcast to all ranks
    data = comm.bcast(data, root=0)

    return data


def scatter_population(comm: MPI.Comm, rank: int, size: int,
                       genomes: Optional[List] = None, verbose: bool = True) -> List:
    """
    Distribute population (genomes) across all MPI ranks.

    Master splits population into chunks and sends to workers.
    Each rank (including master) receives its chunk.

    Args:
        comm: MPI communicator
        rank: Process rank
        size: Total number of MPI processes
        genomes: List of genomes (master only)
        verbose: Print debug info about distribution

    Returns:
        local_chunk: Subset of genomes for this rank to evaluate
    """
    if is_master(rank):
        # Master: split population into chunks
        total = len(genomes)
        chunk_size = total // size
        remainder = total % size

        if verbose:
            print(f"\n  [DEBUG] Population distribution:")
            print(f"  Total individuals: {total}")
            print(f"  MPI ranks: {size}")
            print(f"  Base chunk size: {chunk_size}")
            print(f"  Remainder: {remainder} (first {remainder} ranks get +1 individual)")
            print(f"  Distribution:")

        # Create chunks (distribute remainder evenly)
        chunks = []
        start_idx = 0
        for i in range(size):
            # Give first 'remainder' ranks an extra individual
            current_chunk_size = chunk_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_chunk_size
            chunks.append(genomes[start_idx:end_idx])

            if verbose:
                print(f"    Rank {i}: individuals {start_idx}-{end_idx-1} ({current_chunk_size} total)")

            start_idx = end_idx
    else:
        # Workers: will receive chunk
        chunks = None

    # Scatter chunks to all ranks
    local_chunk = comm.scatter(chunks, root=0)

    return local_chunk


def gather_results(comm: MPI.Comm, rank: int,
                   local_sim_results: List,
                   local_fit_results: List) -> Tuple[Optional[List], Optional[List]]:
    """
    Collect evaluation results from all workers back to master.

    Each rank sends its results, master receives all.

    Args:
        comm: MPI communicator
        rank: Process rank
        local_sim_results: Simulation results from this rank's chunk
        local_fit_results: Fitness results from this rank's chunk

    Returns:
        (all_sim_results, all_fit_results) on master, (None, None) on workers
    """
    # Package local results
    local_results = {
        'sim': local_sim_results,
        'fit': local_fit_results
    }

    # Gather all results to master
    all_results = comm.gather(local_results, root=0)

    if is_master(rank):
        # Master: flatten gathered results (list of dicts -> two lists)
        all_sim_results = []
        all_fit_results = []
        for result_dict in all_results:
            all_sim_results.extend(result_dict['sim'])
            all_fit_results.extend(result_dict['fit'])
        return all_sim_results, all_fit_results
    else:
        # Workers: no need for full results
        return None, None


def print_mpi_info(comm: MPI.Comm, rank: int, size: int):
    """
    Print MPI environment information (master only).

    Args:
        comm: MPI communicator
        rank: Process rank
        size: Total number of MPI processes
    """
    if is_master(rank):
        print("=" * 70)
        print("MPI CONFIGURATION")
        print("=" * 70)
        print(f"Total MPI processes: {size}")
        print(f"Master rank: 0")
        print(f"Worker ranks: 1-{size-1}" if size > 1 else "No workers (single process)")
        print("=" * 70)
        print()