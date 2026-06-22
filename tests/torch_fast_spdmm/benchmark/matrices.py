import warnings
import numpy as np
import scipy.sparse as sp
from typing import List, Tuple

def generate_scattered_block_matrix(system_size: int, block_size: int, profiles: List[Tuple[int, int, float]]):
    """
    Generates a synthetic, highly-structured block-sparse matrix for HPC benchmarking.
    
    This generator builds matrices by dividing a global N x N grid into smaller 
    B x B blocks. It populates these blocks based on explicit sparsity profiles, 
    allowing for precise stress-testing of hardware registers, memory bandwidth, 
    and feature parallelism.

    Args:
        system_size (int): The global dimension (N) of the square matrix. 
            Must be perfectly divisible by `block_size`.
            Example: 262144 or 1048576.
            
        block_size (int): The dimension of the internal dense blocks (block_size x block_size).
            Example: 8 (for 8x8 blocks).
            
        profiles (list of tuples): A list of structural patterns to inject into the matrix.
            Each tuple must contain exactly three values: `(n_r, n_c, density)`.
            
            - n_r (int): The number of active rows inside the block (<= block_size).
            - n_c (int): The number of active columns inside the block (<= block_size).
            - density (float): The fraction of the *entire grid* (0.0 to 1.0) that 
              should be populated with this exact block profile.
              
            Example Profile: `(1, 8, 0.0005)` creates extremely sparse blocks 
            (only 1 active row out of 8) spanning 0.05% of the total available grid space.

    Returns:
        scipy.sparse.csr_matrix: A 32-bit floating point compressed sparse row matrix.
        
    Raises:
        ValueError: If `system_size` is not divisible by `block_size` (handled upstream).
    """
    grid_dim = system_size // block_size
    total_blocks = grid_dim * grid_dim
    all_rows, all_cols, all_data = [], [], []
    
    for (n_r, n_c, density) in profiles:
        num_blocks = int(total_blocks * density)
        if num_blocks == 0:
            raise ValueError(
                f"Profile (n_r={n_r}, n_c={n_c}, density={density}) produced 0 blocks "
                f"for system_size={system_size}, block_size={block_size}. Increase density or system_size."
            )
            
        # 1. Randomly sample block positions from the global grid
        block_idx = np.random.randint(0, total_blocks, size=num_blocks, dtype=np.int64)
        grid_r = block_idx // grid_dim
        grid_c =  block_idx % grid_dim
        
        # 2. Pick n_r/n_c distinct local indices per block without replacement.
        #    argsort(random) produces a random permutation for each block row; slicing [:n_r] takes the first n_r.
        local_rows = np.argsort(np.random.rand(num_blocks, block_size), axis=1)[:, :n_r]
        local_cols = np.argsort(np.random.rand(num_blocks, block_size), axis=1)[:, :n_c]
        
        # 3. Broadcast local coordinates to global matrix coordinates
        global_r = (grid_r[:, None] * block_size) + local_rows  
        global_c = (grid_c[:, None] * block_size) + local_cols  
        
        # 4. Form all (row, col) pairs within each block via broadcasting — gives n_r * n_c entries per block.
        r_mesh = np.broadcast_to(global_r[:, :, None], (num_blocks, n_r, n_c)).flatten()
        c_mesh = np.broadcast_to(global_c[:, None, :], (num_blocks, n_r, n_c)).flatten()
        
        # 5. Generate random float data
        data = np.random.randn(len(r_mesh)).astype(np.float32)
        
        all_rows.append(r_mesh)
        all_cols.append(c_mesh)
        all_data.append(data)
       
        
    # Combine and build the final Scipy CSR matrix
    final_rows = np.concatenate(all_rows)
    final_cols = np.concatenate(all_cols)
    final_data = np.concatenate(all_data)
    
    return sp.csr_matrix((final_data, (final_rows, final_cols)), shape=(system_size, system_size))


def generate_random_sparse(system_size: int, density: float):
    nnz = int(system_size * system_size * density)
    
    # Generate coordinates independently to avoid allocating an N^2 array
    rows = np.random.randint(0, system_size, size=nnz, dtype=np.int32)
    cols = np.random.randint(0, system_size, size=nnz, dtype=np.int32)
    data = np.random.randn(nnz).astype(np.float32)
    
    # Build COO then convert to CSR (automatically sums duplicate coordinates safely)
    return sp.coo_matrix((data, (rows, cols)), shape=(system_size, system_size)).tocsr()


def generate_diag_blocks_plus_noise(system_size: int, block_size: int = 64, diag_density: float = 0.5, off_diag_density: float = 0.00005):
    """
    Generates a sparse matrix with a dense block-diagonal structure and random global noise.
    Simulates real-world graph matrices with strong local clustering and sparse distant connections.
    """
    B = block_size

    if system_size % B != 0:
        raise ValueError(f"system_size ({system_size}) must be divisible by block_size ({B})")

    num_noise_elements = int(system_size * system_size * off_diag_density)

    temp_coo = sp.random(system_size, B, density=diag_density, format='coo', dtype=np.float32)
    all_rows = np.concatenate([temp_coo.row,
                               np.random.randint(0, system_size, size=num_noise_elements, dtype=np.int32)])
    all_cols = np.concatenate([(temp_coo.row // B) * B + temp_coo.col,
                               np.random.randint(0, system_size, size=num_noise_elements, dtype=np.int32)])
    all_data = np.concatenate([temp_coo.data,
                               np.random.randn(num_noise_elements).astype(np.float32)])

    return sp.csr_matrix((all_data, (all_rows, all_cols)), shape=(system_size, system_size))