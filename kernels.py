"""Student kernels for the SGEMM autograder assignment.

You implement K2 (GMEM coalescing), K3 (shared-memory blocking), K4 (1D
register tiling), and K5 (2D register tiling) inside this file. The launch
wrappers, tile-size constants, and signatures are provided — you only edit
the kernel bodies marked TODO.

K1 (naive) is given as a worked example so you have a reference for the
numba.cuda @cuda.jit signature every kernel must match.

To check correctness locally before submitting:
    python sanity_check.py

To submit: push your edits to the main branch of this assignment repo.
Each push that touches kernels.py triggers the autograder, which runs
on a Modal A100 40GB and posts your grade as a comment on the commit.
You have 5 graded submissions per assignment.
"""
import math

from numba import cuda, float32


# ── Tile constants ──────────────────────────────────────────────────
# These are tied to the launch shapes the autograder will use. Do not
# change them; the run_kN wrappers below depend on these values.

BLOCKSIZE = 32          # K1 + K2 tile

# K3 tile sizes
BM3, BN3, BK3 = 32, 32, 32

# K4 tile sizes
BM4, BN4, BK4 = 64, 64, 8
TM4 = 8

# K5 tile sizes
BM5, BN5, BK5 = 128, 128, 8
TM5, TN5 = 8, 8


# ── K1: naive (worked example, do not edit) ─────────────────────────

@cuda.jit
def sgemm_naive(A, B, C, M, N, K):
    """K1: one thread per output element. No tiling, no shared memory.
    Provided so you have a working numba.cuda kernel for reference.
    """
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x < M and y < N:
        tmp = float32(0.0)
        for i in range(K):
            tmp += A[x, i] * B[i, y]
        C[x, y] = tmp


# ── K2: GMEM coalescing ──────────────────────────────────────────────

@cuda.jit
def sgemm_coalesced(A, B, C, M, N, K):
    """K2: rewrite K1 so that 32 threads in a warp end up writing to 32
    *consecutive columns* of C (and reading 32 consecutive elements of B).
    The arithmetic is identical to K1

    Launch shape (run_k2 below uses this):
        block = (BLOCKSIZE * BLOCKSIZE,)        # 1024 threads, 1D
        grid  = (ceil(M / BLOCKSIZE), ceil(N / BLOCKSIZE))

    With a 1D block of 1024 threads, threadIdx.x runs 0..1023.
    Derive (row_in_tile, col_in_tile) from threadIdx.x using integer division
    and modulo by BLOCKSIZE.
    Be careful which one indexes the column.
    """
    tid = cuda.threadIdx.x

    # col varies fast: consecutive threads in a warp → consecutive columns
    # this is what makes B reads and C writes coalesced
    row_in_tile = tid // BLOCKSIZE
    col_in_tile = tid % BLOCKSIZE

    x = cuda.blockIdx.x * BLOCKSIZE + row_in_tile  # global row
    y = cuda.blockIdx.y * BLOCKSIZE + col_in_tile  # global col

    if x < M and y < N:
        tmp = float32(0.0)
        for i in range(K):
            tmp += A[x, i] * B[i, y]
        C[x, y] = tmp


# ── K3: shared-memory cache-blocking ────────────────────────────────

@cuda.jit
def sgemm_smem(A, B, C, M, N, K):
    """K3: stream the K dimension in chunks of BK3. Each block computes a
            BM3 x BN3 output tile by repeatedly:
        1. cooperatively loading a BM3 x BK3 slice of A and a BK3 x BN3
           slice of B into shared memory (one element per thread per slice),
        2. cuda.syncthreads(),
        3. dotting the row of As into the column of Bs to update one
           per-thread accumulator,
        4. cuda.syncthreads() before the next K-chunk.

    Launch shape (run_k3 below uses this):
        block = (BM3 * BN3,)                    # 1024 threads, 1D
        grid  = (ceil(M / BM3), ceil(N / BN3))

    Use cuda.shared.array((BM3, BK3), float32) for As and a similar
    (BK3, BN3) for Bs.
    Use 0.0 in the SMEM load when the global index is out of bounds.
    """
    As = cuda.shared.array((BM3, BK3), dtype=float32)
    Bs = cuda.shared.array((BK3, BN3), dtype=float32)

    tid = cuda.threadIdx.x
    local_row = tid // BN3   # row within this block's output tile
    local_col = tid % BN3    # col within this block's output tile

    tile_row = cuda.blockIdx.x   # which tile along M
    tile_col = cuda.blockIdx.y   # which tile along N

    global_row = tile_row * BM3 + local_row
    global_col = tile_col * BN3 + local_col

    acc = float32(0.0)

    num_chunks = (K + BK3 - 1) // BK3
    for chunk in range(num_chunks):
        # Step 1: cooperatively load As (BM3 x BK3), one element per thread
        a_col = chunk * BK3 + local_col
        if global_row < M and a_col < K:
            As[local_row, local_col] = A[global_row, a_col]
        else:
            As[local_row, local_col] = float32(0.0)

        # Cooperatively load Bs (BK3 x BN3), one element per thread
        b_row = chunk * BK3 + local_row
        if b_row < K and global_col < N:
            Bs[local_row, local_col] = B[b_row, global_col]
        else:
            Bs[local_row, local_col] = float32(0.0)

        # Step 2: wait for all loads to finish
        cuda.syncthreads()

        # Step 3: accumulate partial dot product from shared memory
        for dk in range(BK3):
            acc += As[local_row, dk] * Bs[dk, local_col]

        # Step 4: wait before next chunk overwrites shared memory
        cuda.syncthreads()

    if global_row < M and global_col < N:
        C[global_row, global_col] = acc


# ── K4: 1D register tiling ───────────────────────────────────────────

@cuda.jit
def sgemm_1d_tile(A, B, C, M, N, K):
    """K4: extend K3 by giving each thread TM4 = 8 rows in a single column
    of the BM4 x BN4 output tile.

    Note: blockIdx.x now indexes COLUMNS of the output.
    The run_k4 wrapper below already accounts for this, but you need to compute the global (row, col)
    start of your block accordingly.

    Launch shape (run_k4 below uses this):
        block = ((BM4 * BN4) // TM4,)           # 512 threads
        grid  = (ceil(N / BN4), ceil(M / BM4))  # x = col, y = row

    Cooperative loads here are tidy: A's tile is BM4 x BK4 = 512 elements,
    B's tile is BK4 x BN4 = 512 elements, and you have 512 threads so
    exactly one element per thread per tile (so no inner-load loop)

    Use cuda.local.array(TM4, float32) for the per-thread accumulator array.
    Initialize all entries to 0.0 before the K-loop.
    """
    As = cuda.shared.array((BM4, BK4), dtype=float32)
    Bs = cuda.shared.array((BK4, BN4), dtype=float32)

    # blockIdx.x = column tile, blockIdx.y = row tile (axis swap from K1-K3)
    tile_col = cuda.blockIdx.x
    tile_row = cuda.blockIdx.y

    tid = cuda.threadIdx.x  # 0..511

    # Each thread owns TM4=8 consecutive rows in one column
    thread_col = tid % BN4           # which column within tile (0..63)
    thread_row = (tid // BN4) * TM4  # starting row within tile (0, 8, 16, ...)

    # Per-thread accumulator: TM4 values for 8 rows
    acc = cuda.local.array(TM4, dtype=float32)
    for i in range(TM4):
        acc[i] = float32(0.0)

    num_chunks = (K + BK4 - 1) // BK4
    for chunk in range(num_chunks):
        # Cooperative load of As (BM4 x BK4 = 512 elements, 512 threads → 1 each)
        a_row = tid // BK4
        a_col = tid % BK4
        g_row = tile_row * BM4 + a_row
        g_col = chunk * BK4 + a_col
        if g_row < M and g_col < K:
            As[a_row, a_col] = A[g_row, g_col]
        else:
            As[a_row, a_col] = float32(0.0)

        # Cooperative load of Bs (BK4 x BN4 = 512 elements, 512 threads → 1 each)
        b_row = tid // BN4
        b_col = tid % BN4
        g_b_row = chunk * BK4 + b_row
        g_b_col = tile_col * BN4 + b_col
        if g_b_row < K and g_b_col < N:
            Bs[b_row, b_col] = B[g_b_row, g_b_col]
        else:
            Bs[b_row, b_col] = float32(0.0)

        cuda.syncthreads()

        # Each thread does TM4 FMAs: broadcast one Bs value, multiply by TM4 As values
        for dk in range(BK4):
            b_val = Bs[dk, thread_col]
            for tm in range(TM4):
                acc[tm] += As[thread_row + tm, dk] * b_val

        cuda.syncthreads()

    # Write TM4 output elements
    for tm in range(TM4):
        out_row = tile_row * BM4 + thread_row + tm
        out_col = tile_col * BN4 + thread_col
        if out_row < M and out_col < N:
            C[out_row, out_col] = acc[tm]


# ── K5: 2D register tiling ───────────────────────────────────────────

@cuda.jit
def sgemm_2d_tile(A, B, C, M, N, K):
    """K5: extend K4 to a TM5 x TN5 = 8 x 8 register tile per thread.
    Inside the inner-k loop, cache TM5 As values and TN5 Bs values into
    register arrays, then do the TM5 x TN5 outer-product update.

    Launch shape (run_k5 below uses this):
        block = ((BM5 * BN5) // (TM5 * TN5),)   # 256 threads
        grid  = (ceil(N / BN5), ceil(M / BM5))

    Cooperative loads now need a stride loop: the tile has more elements
    (BM5 * BK5 = 1024) than the block has threads (256), so each thread
    loads BM5 * BK5 / 256 = 4 elements of A per K-chunk and similarly for B.
    Pick the per-thread row stride so that consecutive threads touch
    consecutive memory addresses (= coalesced GMEM loads).

    For accumulators, use cuda.local.array((TM5, TN5), float32).
    Numba supports tuple-shaped local arrays!
    """
    # Transposed As (BK5 x BM5) avoids bank conflicts on column reads
    As = cuda.shared.array((BK5, BM5), dtype=float32)
    Bs = cuda.shared.array((BK5, BN5), dtype=float32)

    tile_col = cuda.blockIdx.x
    tile_row = cuda.blockIdx.y
    tid = cuda.threadIdx.x  # 0..255

    threads_per_row = BN5 // TN5   # 16
    thread_col = (tid % threads_per_row) * TN5
    thread_row = (tid // threads_per_row) * TM5

    # Flatten acc to 1D — Numba optimizes 1D local arrays into registers much better
    acc = cuda.local.array(TM5 * TN5, dtype=float32)
    reg_a = cuda.local.array(TM5, dtype=float32)
    reg_b = cuda.local.array(TN5, dtype=float32)

    for i in range(TM5 * TN5):
        acc[i] = float32(0.0)

    num_chunks = (K + BK5 - 1) // BK5
    for chunk in range(num_chunks):
        # Load As transposed: 4 elements per thread, coalesced global reads
        for load_idx in range(4):
            idx = tid + load_idx * 256
            a_row = idx // BK5
            a_col = idx % BK5
            g_row = tile_row * BM5 + a_row
            g_col = chunk * BK5 + a_col
            if g_row < M and g_col < K:
                As[a_col, a_row] = A[g_row, g_col]
            else:
                As[a_col, a_row] = float32(0.0)

        # Load Bs: 4 elements per thread
        for load_idx in range(4):
            idx = tid + load_idx * 256
            b_row = idx // BN5
            b_col = idx % BN5
            g_b_row = chunk * BK5 + b_row
            g_b_col = tile_col * BN5 + b_col
            if g_b_row < K and g_b_col < N:
                Bs[b_row, b_col] = B[g_b_row, g_b_col]
            else:
                Bs[b_row, b_col] = float32(0.0)

        cuda.syncthreads()

        for dk in range(BK5):
            for tm in range(TM5):
                reg_a[tm] = As[dk, thread_row + tm]
            for tn in range(TN5):
                reg_b[tn] = Bs[dk, thread_col + tn]
            for tm in range(TM5):
                for tn in range(TN5):
                    acc[tm * TN5 + tn] += reg_a[tm] * reg_b[tn]

        cuda.syncthreads()

    # Write 64 outputs
    for tm in range(TM5):
        for tn in range(TN5):
            out_row = tile_row * BM5 + thread_row + tm
            out_col = tile_col * BN5 + thread_col + tn
            if out_row < M and out_col < N:
                C[out_row, out_col] = acc[tm * TN5 + tn]


# ── Launch wrappers (provided — do not edit) ────────────────────────

def run_k1(A, B, C, M, N, K):
    grid = (math.ceil(M / BLOCKSIZE), math.ceil(N / BLOCKSIZE))
    block = (BLOCKSIZE, BLOCKSIZE)
    sgemm_naive[grid, block](A, B, C, M, N, K)


def run_k2(A, B, C, M, N, K):
    grid = (math.ceil(M / BLOCKSIZE), math.ceil(N / BLOCKSIZE))
    block = (BLOCKSIZE * BLOCKSIZE,)
    sgemm_coalesced[grid, block](A, B, C, M, N, K)


def run_k3(A, B, C, M, N, K):
    grid = (math.ceil(M / BM3), math.ceil(N / BN3))
    block = (BM3 * BN3,)
    sgemm_smem[grid, block](A, B, C, M, N, K)


def run_k4(A, B, C, M, N, K):
    # Axis swap: blockIdx.x indexes columns of C.
    grid = (math.ceil(N / BN4), math.ceil(M / BM4))
    block = ((BM4 * BN4) // TM4,)
    sgemm_1d_tile[grid, block](A, B, C, M, N, K)


def run_k5(A, B, C, M, N, K):
    grid = (math.ceil(N / BN5), math.ceil(M / BM5))
    block = ((BM5 * BN5) // (TM5 * TN5),)
    sgemm_2d_tile[grid, block](A, B, C, M, N, K)


# Graded kernels in the order the rubric uses (1/4 → C, 2/4 → B-, ...).
KERNELS = [
    ("k2_coalesce", run_k2),
    ("k3_smem",     run_k3),
    ("k4_1d_tile",  run_k4),
    ("k5_2d_tile",  run_k5),
]