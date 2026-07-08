import warnings
import numpy as np
import scipy.sparse as sp
from typing import List, Tuple, Literal

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


def generate_fem_2d(
    system_size: int,
    dofs_per_node: int = 4,
    coupling: Literal["isotropic", "anisotropic"] = "anisotropic",
    noise_density: float = 0.0,
    seed: int = None,
) -> sp.csr_matrix:
    """
    2D structured FEM sparse matrix — bilinear quad elements on a uniform n×n mesh.

    Physical model: a scalar Laplacian (or, with dofs_per_node>1, a vector field)
    discretised with bilinear quad elements.  Every non-zero entry sits inside an
    aligned dofs_per_node × dofs_per_node dense block (anisotropic coupling) or on
    the diagonal of one (isotropic), giving the block structure that PBR super-tile
    kernels are designed to exploit.

    Key SpMM property — unlike scattered random matrices, adjacent rows *always*
    share column bands: mesh node i and node i+1 both connect to the same set of
    neighbours (the patch of quads they share).  This enables simultaneous X-load
    and Y-flush amortisation in a blocks-of-blocks (tile-of-tiles) kernel.

    Sparsity: interior nodes have degree 8; boundary/corner nodes have degree 3–5.
    Each connected node-pair contributes one d×d block (anisotropic) or d diagonal
    entries (isotropic), where d = dofs_per_node.

    Exact sizes: system_size = k²·d gives N = system_size exactly.  The standard
    benchmark sizes all satisfy this for d=4:
        16384 = 4×64²  (n=64)
        65536 = 4×128² (n=128)
       262144 = 4×256² (n=256)

    Args:
        system_size:   Target matrix dimension N.  Actual N = n²·d,
                       n = floor(√(system_size / d)).  A UserWarning is emitted
                       when the actual N differs from system_size.
        dofs_per_node: DOFs per mesh node (d).  Sets the dense block size.
                       d=4 gives exact sizes for all standard benchmarks.
                       d=8 aligns with PBR block_size=8 (≤56-row size offset).
        coupling:      "anisotropic" (default) — full d×d blocks at each node-pair,
                           using a single random symmetric coupling tensor C scaled
                           per pair by K_e[a,b].  100% block fill; best for tile kernels.
                       "isotropic" — Kronecker(K_e, I_d); only diagonal DOF pairs
                           couple.  1/d block fill; lightweight assembly.
        noise_density: Fraction of N² entries to add as weak (×0.01) random
                       long-range connections (default 0 = pure mesh connectivity).
        seed:          NumPy Generator seed for reproducibility (default None).

    Returns:
        scipy.sparse.csr_matrix, float32, symmetric, shape (N, N).
    """
    rng = np.random.default_rng(seed)
    d = dofs_per_node

    n = int(np.sqrt(system_size / d))
    if n < 2:
        raise ValueError(
            f"system_size={system_size} too small for dofs_per_node={d}: "
            f"need n ≥ 2 mesh nodes per side (min system_size={4 * d})."
        )
    N = n * n * d
    if N != system_size:
        warnings.warn(
            f"generate_fem_2d: adjusted system_size {system_size} → {N} "
            f"(n={n} nodes/side, d={d} DOFs/node).  "
            f"For an exact match use system_size = k²·d "
            f"(e.g. {n**2 * d} or {(n + 1)**2 * d}).",
            UserWarning,
            stacklevel=2,
        )

    # ── Element connectivity ────────────────────────────────────────────────
    # Row-major node numbering: node(i, j) = i·n + j.
    # (n-1)² quad elements; each has four CCW-ordered corner nodes.
    n_elem = (n - 1) * (n - 1)
    ei = np.repeat(np.arange(n - 1, dtype=np.int64), n - 1)  # element row index
    ej = np.tile(np.arange(n - 1, dtype=np.int64), n - 1)    # element col index

    elem_nodes = np.column_stack([       # (n_elem, 4)  global node ids
        ei * n + ej,                     # 0: bottom-left  (i,   j  )
        ei * n + (ej + 1),               # 1: bottom-right (i,   j+1)
        (ei + 1) * n + (ej + 1),        # 2: top-right    (i+1, j+1)
        (ei + 1) * n + ej,              # 3: top-left     (i+1, j  )
    ])

    # ── Element stiffness (2D Laplacian, bilinear quad, unit square) ────────
    # K_e[a,b] = ∫∫ ∇Nₐ · ∇N_b dΩ  over [0,1]².
    # Properties: symmetric, row-sums = 0 (Galerkin consistency),
    #             diagonally dominant, all diagonal entries positive.
    K_e = np.array(
        [[ 4., -1., -2., -1.],
         [-1.,  4., -1., -2.],
         [-2., -1.,  4., -1.],
         [-1., -2., -1.,  4.]],
        dtype=np.float32,
    ) / 6.0

    # All 16 local node-pair combinations (a, b) ∈ {0,1,2,3}²
    a_idx, b_idx = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
    a_flat = a_idx.ravel()                   # (16,) local node index on A-side
    b_flat = b_idx.ravel()                   # (16,) local node index on B-side
    k_flat = K_e[a_flat, b_flat]            # (16,) stiffness scalar per pair

    node_a = elem_nodes[:, a_flat]           # (n_elem, 16)  global node for A-side
    node_b = elem_nodes[:, b_flat]           # (n_elem, 16)  global node for B-side

    # ── DOF-level assembly ──────────────────────────────────────────────────
    if coupling == "isotropic":
        # Kronecker(K_e, I_d): only DOFs with the same local index f couple.
        # Produces d entries per node-pair; PBR block fill = 1/d.
        # val[e, p, f] = k_flat[p]  with row = node_a[e,p]·d+f, col = node_b[e,p]·d+f.
        f_dof = np.arange(d, dtype=np.int64)
        row = (node_a[:, :, None] * d + f_dof[None, None, :]).ravel()   # (n_elem·16·d,)
        col = (node_b[:, :, None] * d + f_dof[None, None, :]).ravel()
        val = np.broadcast_to(
            k_flat[None, :, None], (n_elem, 16, d)
        ).ravel()

    else:  # anisotropic
        # Full d×d coupling block at every node-pair, scaled by K_e[a,b].
        # Uses a single random symmetric coupling tensor C (uniform material).
        # val[e, p, f1, f2] = k_flat[p] · C[f1, f2]
        # Produces d² entries per node-pair; PBR block fill = 100%.
        C_raw = rng.standard_normal((d, d)).astype(np.float32)
        C = (C_raw + C_raw.T) * 0.5                 # symmetric coupling tensor
        pair_blocks = k_flat[:, None, None] * C[None, :, :]  # (16, d, d)

        f1 = np.arange(d, dtype=np.int64)
        f2 = np.arange(d, dtype=np.int64)
        # row depends only on f1 (row DOF); col depends only on f2 (col DOF).
        row_2d = node_a[:, :, None] * d + f1[None, None, :]   # (n_elem, 16, d)
        col_2d = node_b[:, :, None] * d + f2[None, None, :]   # (n_elem, 16, d)
        row = np.broadcast_to(row_2d[:, :, :, None], (n_elem, 16, d, d)).ravel()
        col = np.broadcast_to(col_2d[:, :, None, :], (n_elem, 16, d, d)).ravel()
        val = np.broadcast_to(pair_blocks[None], (n_elem, 16, d, d)).ravel()

    mat = sp.csr_matrix((val, (row, col)), shape=(N, N), dtype=np.float32)

    # ── Optional long-range noise ───────────────────────────────────────────
    if noise_density > 0.0:
        n_noise = max(1, int(N * N * noise_density))
        r_n = rng.integers(0, N, n_noise, dtype=np.int64)
        c_n = rng.integers(0, N, n_noise, dtype=np.int64)
        v_n = (rng.standard_normal(n_noise) * 0.01).astype(np.float32)
        mat = mat + sp.csr_matrix((v_n, (r_n, c_n)), shape=(N, N), dtype=np.float32)

    # Enforce exact symmetry — floating-point duplicate summation can introduce
    # small asymmetries; (A + A.T) / 2 guarantees A[i,j] == A[j,i] exactly.
    mat = (mat + mat.T) * 0.5
    return mat.astype(np.float32)