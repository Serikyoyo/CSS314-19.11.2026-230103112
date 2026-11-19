import os
import sys
import json
import math
import hashlib
import subprocess
import numpy as np
from mpi4py import MPI
from numba import cuda, float64

# Compile-time constants for 2D shared-memory tiling (16x16 interior + 1-cell halo apron)
TILE_DIM = 16
SHARED_DIM = TILE_DIM + 2  # 18
GLOBAL_NX = 512
GLOBAL_NY = 512
LOCAL_NX = GLOBAL_NX // 2  # 256
LOCAL_NY = GLOBAL_NY // 2  # 256
PADDED_NX = LOCAL_NX + 2   # 258
PADDED_NY = LOCAL_NY + 2   # 258

STEPS = 400
LAMBDA_DAMP = 0.15

# Physical domain Omega = [0, 1] x [0, 1] with 512 interior points -> 513 intervals
DX = 1.0 / (GLOBAL_NX + 1)
DY = 1.0 / (GLOBAL_NY + 1)
KAPPA_UPPER = 0.24
DT = 0.8 * (DX * DX) / (4.0 * KAPPA_UPPER)


def get_gpu_uuid() -> str:
    """Query active NVIDIA GPU UUID via nvidia-smi."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        encoding="utf-8",
    )
    return out.strip().splitlines()[0].strip()


def get_gpu_name() -> str:
    """Query active NVIDIA GPU model name via nvidia-smi."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        encoding="utf-8",
    )
    return out.strip().splitlines()[0].strip()


def derive_student_seed(github_user: str, gpu_uuid: str) -> int:
    """Derive deterministic 64-bit integer seed bound to student GitHub handle and GPU UUID."""
    digest = hashlib.sha256(f"{github_user.strip()}::{gpu_uuid}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


# ==============================================================================
# TASK 1: Distributed Cartesian Topology & Hardware Context Initialization
# ==============================================================================
def init_distributed_context(github_user: str):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size != 4:
        raise RuntimeError(f"Expected exactly 4 MPI ranks (2x2 grid), got {size}.")

    # Bind process to GPU 0 on Google Colab
    cuda.select_device(0)
    gpu_uuid = get_gpu_uuid() if rank == 0 else None
    gpu_name = get_gpu_name() if rank == 0 else None
    gpu_uuid = comm.bcast(gpu_uuid, root=0)
    gpu_name = comm.bcast(gpu_name, root=0)

    instance_seed = derive_student_seed(github_user, gpu_uuid)

    # Non-periodic 2x2 Cartesian process topology
    dims = [2, 2]
    periods = [False, False]
    cart_comm = comm.Create_cart(dims=dims, periods=periods, reorder=False)
    cart_rank = cart_comm.Get_rank()
    row_rank, col_rank = cart_comm.Get_coords(cart_rank)

    # Axis 0 = vertical (rows: up/down), Axis 1 = horizontal (cols: left/right)
    nbr_up, nbr_down = cart_comm.Shift(0, 1)
    nbr_left, nbr_right = cart_comm.Shift(1, 1)

    ctx = {
        "comm": comm,
        "cart_comm": cart_comm,
        "rank": cart_rank,
        "size": size,
        "dims": dims,
        "row_rank": row_rank,
        "col_rank": col_rank,
        "nbr_up": nbr_up,
        "nbr_down": nbr_down,
        "nbr_left": nbr_left,
        "nbr_right": nbr_right,
        "gpu_uuid": gpu_uuid,
        "gpu_name": gpu_name,
        "instance_seed": instance_seed,
    }
    return ctx


# ==============================================================================
# TASK 2: Seeded Heterogeneous Subdomain Initialization (Rank-Local 258 x 258)
# ==============================================================================
def initialize_subdomains(ctx: dict):
    row_rank = ctx["row_rank"]
    col_rank = ctx["col_rank"]
    instance_seed = ctx["instance_seed"]

    # Map local indices 0..257 directly to global physical coordinates in [0, 1]^2
    # without allocating any global 512 x 512 host or device array.
    global_i = row_rank * LOCAL_NY + np.arange(PADDED_NY, dtype=np.float64)
    global_j = col_rank * LOCAL_NX + np.arange(PADDED_NX, dtype=np.float64)
    y_1d = global_i * DY
    x_1d = global_j * DX
    X, Y = np.meshgrid(x_1d, y_1d)

    # Spatially heterogeneous diffusivity kappa(x, y) strictly in [0.08, 0.24]
    kappa = (
        0.16
        + 0.05 * np.sin(2.0 * np.pi * X) * np.cos(2.0 * np.pi * Y)
        + 0.03 * np.cos(4.0 * np.pi * X + 3.0 * np.pi * Y)
    )

    # Initial thermal field composed of 4 seed-derived Gaussian pulses
    rng = np.random.default_rng(instance_seed)
    u0 = np.zeros((PADDED_NY, PADDED_NX), dtype=np.float64)
    for _ in range(4):
        cx = float(rng.uniform(0.25, 0.75))
        cy = float(rng.uniform(0.25, 0.75))
        amp = float(rng.uniform(0.6, 1.2))
        sigma = float(rng.uniform(0.06, 0.10))
        u0 += amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma ** 2))

    # Localized smooth heat source S(x, y)
    source = 0.05 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (2.0 * 0.08 ** 2))

    # Zero out all 4 ghost halo aprons initially:
    # - Outer physical boundaries (MPI.PROC_NULL) enforce homogeneous Dirichlet u = 0.0
    # - Interior rank boundaries start at 0.0 and must be populated via MPI halo exchange
    u0[0, :] = 0.0
    u0[-1, :] = 0.0
    u0[:, 0] = 0.0
    u0[:, -1] = 0.0

    return (
        np.ascontiguousarray(u0, dtype=np.float64),
        np.ascontiguousarray(kappa, dtype=np.float64),
        np.ascontiguousarray(source, dtype=np.float64),
    )


# ==============================================================================
# TASK 3: GPU Halo Pack/Unpack Kernels & Pinned-Memory Async MPI Halo Exchange
# ==============================================================================
@cuda.jit
def pack_halos_kernel(u, send_up, send_down, send_left, send_right, ny, nx):
    idx = cuda.grid(1)
    if idx < nx:
        send_up[idx] = u[1, idx + 1]
        send_down[idx] = u[ny, idx + 1]
    if idx < ny:
        send_left[idx] = u[idx + 1, 1]
        send_right[idx] = u[idx + 1, nx]


@cuda.jit
def unpack_halos_kernel(
    u,
    recv_up,
    recv_down,
    recv_left,
    recv_right,
    has_up,
    has_down,
    has_left,
    has_right,
    ny,
    nx,
):
    idx = cuda.grid(1)
    if idx < nx:
        u[0, idx + 1] = recv_up[idx] if has_up else 0.0
        u[ny + 1, idx + 1] = recv_down[idx] if has_down else 0.0
    if idx < ny:
        u[idx + 1, 0] = recv_left[idx] if has_left else 0.0
        u[idx + 1, nx + 1] = recv_right[idx] if has_right else 0.0


class HaloExchangeBuffers:
    """Pre-allocated device 1D halo vectors and page-locked (pinned) host staging buffers."""

    def __init__(self, ny: int, nx: int):
        self.ny = ny
        self.nx = nx

        # Device 1D buffers
        self.d_send_up = cuda.device_array((nx,), dtype=np.float64)
        self.d_send_down = cuda.device_array((nx,), dtype=np.float64)
        self.d_send_left = cuda.device_array((ny,), dtype=np.float64)
        self.d_send_right = cuda.device_array((ny,), dtype=np.float64)

        self.d_recv_up = cuda.device_array((nx,), dtype=np.float64)
        self.d_recv_down = cuda.device_array((nx,), dtype=np.float64)
        self.d_recv_left = cuda.device_array((ny,), dtype=np.float64)
        self.d_recv_right = cuda.device_array((ny,), dtype=np.float64)

        # Page-locked host staging buffers for asynchronous PCIe DMA
        self.h_send_up = cuda.pinned_array((nx,), dtype=np.float64)
        self.h_send_down = cuda.pinned_array((nx,), dtype=np.float64)
        self.h_send_left = cuda.pinned_array((ny,), dtype=np.float64)
        self.h_send_right = cuda.pinned_array((ny,), dtype=np.float64)

        self.h_recv_up = cuda.pinned_array((nx,), dtype=np.float64)
        self.h_recv_down = cuda.pinned_array((nx,), dtype=np.float64)
        self.h_recv_left = cuda.pinned_array((ny,), dtype=np.float64)
        self.h_recv_right = cuda.pinned_array((ny,), dtype=np.float64)

        self.h_recv_up.fill(0.0)
        self.h_recv_down.fill(0.0)
        self.h_recv_left.fill(0.0)
        self.h_recv_right.fill(0.0)


def post_halo_pack_and_d2h(d_u, bufs: HaloExchangeBuffers, stream):
    threads = 256
    blocks = (max(bufs.nx, bufs.ny) + threads - 1) // threads
    pack_halos_kernel[blocks, threads, stream](
        d_u,
        bufs.d_send_up,
        bufs.d_send_down,
        bufs.d_send_left,
        bufs.d_send_right,
        bufs.ny,
        bufs.nx,
    )
    bufs.d_send_up.copy_to_host(bufs.h_send_up, stream=stream)
    bufs.d_send_down.copy_to_host(bufs.h_send_down, stream=stream)
    bufs.d_send_left.copy_to_host(bufs.h_send_left, stream=stream)
    bufs.d_send_right.copy_to_host(bufs.h_send_right, stream=stream)


def post_mpi_nonblocking(ctx: dict, bufs: HaloExchangeBuffers):
    cart_comm = ctx["cart_comm"]
    nbr_up = ctx["nbr_up"]
    nbr_down = ctx["nbr_down"]
    nbr_left = ctx["nbr_left"]
    nbr_right = ctx["nbr_right"]

    if nbr_up == MPI.PROC_NULL:
        bufs.h_recv_up.fill(0.0)
    if nbr_down == MPI.PROC_NULL:
        bufs.h_recv_down.fill(0.0)
    if nbr_left == MPI.PROC_NULL:
        bufs.h_recv_left.fill(0.0)
    if nbr_right == MPI.PROC_NULL:
        bufs.h_recv_right.fill(0.0)

    reqs = [
        # Tag 10: upward traveling messages (sender -> nbr_up, receiver <- nbr_down)
        cart_comm.Irecv(bufs.h_recv_down, source=nbr_down, tag=10),
        # Tag 20: downward traveling messages (sender -> nbr_down, receiver <- nbr_up)
        cart_comm.Irecv(bufs.h_recv_up, source=nbr_up, tag=20),
        # Tag 30: leftward traveling messages (sender -> nbr_left, receiver <- nbr_right)
        cart_comm.Irecv(bufs.h_recv_right, source=nbr_right, tag=30),
        # Tag 40: rightward traveling messages (sender -> nbr_right, receiver <- nbr_left)
        cart_comm.Irecv(bufs.h_recv_left, source=nbr_left, tag=40),
        # Matching non-blocking sends
        cart_comm.Isend(bufs.h_send_up, dest=nbr_up, tag=10),
        cart_comm.Isend(bufs.h_send_down, dest=nbr_down, tag=20),
        cart_comm.Isend(bufs.h_send_left, dest=nbr_left, tag=30),
        cart_comm.Isend(bufs.h_send_right, dest=nbr_right, tag=40),
    ]
    return reqs


def complete_h2d_and_unpack(ctx: dict, d_u, bufs: HaloExchangeBuffers, stream):
    cuda.to_device(bufs.h_recv_up, stream=stream, to=bufs.d_recv_up)
    cuda.to_device(bufs.h_recv_down, stream=stream, to=bufs.d_recv_down)
    cuda.to_device(bufs.h_recv_left, stream=stream, to=bufs.d_recv_left)
    cuda.to_device(bufs.h_recv_right, stream=stream, to=bufs.d_recv_right)

    has_up = 1 if ctx["nbr_up"] != MPI.PROC_NULL else 0
    has_down = 1 if ctx["nbr_down"] != MPI.PROC_NULL else 0
    has_left = 1 if ctx["nbr_left"] != MPI.PROC_NULL else 0
    has_right = 1 if ctx["nbr_right"] != MPI.PROC_NULL else 0

    threads = 256
    blocks = (max(bufs.nx, bufs.ny) + threads - 1) // threads
    unpack_halos_kernel[blocks, threads, stream](
        d_u,
        bufs.d_recv_up,
        bufs.d_recv_down,
        bufs.d_recv_left,
        bufs.d_recv_right,
        has_up,
        has_down,
        has_left,
        has_right,
        bufs.ny,
        bufs.nx,
    )


def exchange_halos_async(ctx: dict, d_u, bufs: HaloExchangeBuffers, stream):
    """Standalone synchronous-wrapper halo exchange used in serialized baseline mode."""
    post_halo_pack_and_d2h(d_u, bufs, stream)
    stream.synchronize()
    reqs = post_mpi_nonblocking(ctx, bufs)
    MPI.Request.Waitall(reqs)
    complete_h2d_and_unpack(ctx, d_u, bufs, stream)


# ==============================================================================
# TASK 4: 2D Shared-Memory Tiled Stencil Kernel & Dual-Stream Orchestrator
# ==============================================================================
@cuda.jit
def tiled_stencil_kernel(
    u_curr,
    u_next,
    kappa,
    source,
    r_start,
    r_end,
    c_start,
    c_end,
    inv_dx2,
    inv_dy2,
    dt,
    lambda_damp,
):
    smem = cuda.shared.array(shape=(SHARED_DIM, SHARED_DIM), dtype=float64)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    row = r_start + cuda.blockIdx.y * TILE_DIM + ty
    col = c_start + cuda.blockIdx.x * TILE_DIM + tx

    ny_pad = u_curr.shape[0]
    nx_pad = u_curr.shape[1]

    # Load center cell into shared memory tile
    if row < ny_pad and col < nx_pad:
        smem[ty + 1, tx + 1] = u_curr[row, col]
    else:
        smem[ty + 1, tx + 1] = 0.0

    # Boundary threads load 1-cell halo apron
    if ty == 0:
        r_top = row - 1
        if r_top >= 0 and col < nx_pad:
            smem[0, tx + 1] = u_curr[r_top, col]
        else:
            smem[0, tx + 1] = 0.0

    if ty == TILE_DIM - 1:
        r_bot = row + 1
        if r_bot < ny_pad and col < nx_pad:
            smem[TILE_DIM + 1, tx + 1] = u_curr[r_bot, col]
        else:
            smem[TILE_DIM + 1, tx + 1] = 0.0

    if tx == 0:
        c_left = col - 1
        if row < ny_pad and c_left >= 0:
            smem[ty + 1, 0] = u_curr[row, c_left]
        else:
            smem[ty + 1, 0] = 0.0

    if tx == TILE_DIM - 1:
        c_right = col + 1
        if row < ny_pad and c_right < nx_pad:
            smem[ty + 1, TILE_DIM + 1] = u_curr[row, c_right]
        else:
            smem[ty + 1, TILE_DIM + 1] = 0.0

    # Synchronize all 256 threads in the 16x16 block before stencil arithmetic
    cuda.syncthreads()

    if row <= r_end and col <= c_end:
        sy = ty + 1
        sx = tx + 1

        u_c = smem[sy, sx]
        u_up = smem[sy - 1, sx]
        u_dn = smem[sy + 1, sx]
        u_lt = smem[sy, sx - 1]
        u_rt = smem[sy, sx + 1]

        k_c = kappa[row, col]
        k_up_half = 0.5 * (kappa[row - 1, col] + k_c)
        k_dn_half = 0.5 * (kappa[row + 1, col] + k_c)
        k_lt_half = 0.5 * (kappa[row, col - 1] + k_c)
        k_rt_half = 0.5 * (kappa[row, col + 1] + k_c)

        flux_y = (k_dn_half * (u_dn - u_c) - k_up_half * (u_c - u_up)) * inv_dy2
        flux_x = (k_rt_half * (u_rt - u_c) - k_lt_half * (u_c - u_lt)) * inv_dx2

        rhs = flux_y + flux_x - lambda_damp * (u_c * u_c * u_c) + source[row, col]
        u_next[row, col] = u_c + dt * rhs


def launch_stencil_box(
    d_u_curr,
    d_u_next,
    d_kappa,
    d_source,
    r_start: int,
    r_end: int,
    c_start: int,
    c_end: int,
    stream=0,
):
    num_rows = r_end - r_start + 1
    num_cols = c_end - c_start + 1
    if num_rows <= 0 or num_cols <= 0:
        return
    grid_x = (num_cols + TILE_DIM - 1) // TILE_DIM
    grid_y = (num_rows + TILE_DIM - 1) // TILE_DIM
    inv_dx2 = 1.0 / (DX * DX)
    inv_dy2 = 1.0 / (DY * DY)
    tiled_stencil_kernel[(grid_x, grid_y), (TILE_DIM, TILE_DIM), stream](
        d_u_curr,
        d_u_next,
        d_kappa,
        d_source,
        r_start,
        r_end,
        c_start,
        c_end,
        inv_dx2,
        inv_dy2,
        DT,
        LAMBDA_DAMP,
    )


def step_serialized(ctx: dict, d_u_curr, d_u_next, d_kappa, d_source, bufs: HaloExchangeBuffers, stream):
    """Serialized execution: halo exchange completes fully before full-subdomain stencil launch."""
    exchange_halos_async(ctx, d_u_curr, bufs, stream)
    launch_stencil_box(d_u_curr, d_u_next, d_kappa, d_source, 1, LOCAL_NY, 1, LOCAL_NX, stream=stream)
    stream.synchronize()


def step_overlapped(
    ctx: dict,
    d_u_curr,
    d_u_next,
    d_kappa,
    d_source,
    bufs: HaloExchangeBuffers,
    stream_interior,
    stream_boundary,
):
    """Dual-stream overlapped execution: interior [2..255, 2..255] runs concurrently with MPI halo exchange."""
    # Stage A: Pack border cells and initiate async D2H DMA on stream_boundary
    post_halo_pack_and_d2h(d_u_curr, bufs, stream_boundary)

    # Stage B: Synchronize D2H, post non-blocking MPI Irecv/Isend, and compute deep interior concurrently
    stream_boundary.synchronize()
    reqs = post_mpi_nonblocking(ctx, bufs)
    launch_stencil_box(
        d_u_curr,
        d_u_next,
        d_kappa,
        d_source,
        2,
        LOCAL_NY - 1,
        2,
        LOCAL_NX - 1,
        stream=stream_interior,
    )

    # Stage C: Complete MPI halo transfers, async H2D + unpack, and evaluate 4 outer border strips
    MPI.Request.Waitall(reqs)
    complete_h2d_and_unpack(ctx, d_u_curr, bufs, stream_boundary)

    # Top owned row (row 1, cols 1..256)
    launch_stencil_box(d_u_curr, d_u_next, d_kappa, d_source, 1, 1, 1, LOCAL_NX, stream=stream_boundary)
    # Bottom owned row (row 256, cols 1..256)
    launch_stencil_box(d_u_curr, d_u_next, d_kappa, d_source, LOCAL_NY, LOCAL_NY, 1, LOCAL_NX, stream=stream_boundary)
    # Left owned col excluding top/bottom corners (rows 2..255, col 1)
    launch_stencil_box(d_u_curr, d_u_next, d_kappa, d_source, 2, LOCAL_NY - 1, 1, 1, stream=stream_boundary)
    # Right owned col excluding top/bottom corners (rows 2..255, col 256)
    launch_stencil_box(d_u_curr, d_u_next, d_kappa, d_source, 2, LOCAL_NY - 1, LOCAL_NX, LOCAL_NX, stream=stream_boundary)

    stream_interior.synchronize()
    stream_boundary.synchronize()


# ==============================================================================
# TASK 5: Distributed Reductions, PTX Audit & Comparative Benchmark
# ==============================================================================
def compute_global_metrics(cart_comm, d_u_curr, d_u_prev=None):
    """Compute global thermal energy E and global L2 residual/field norm via MPI.Allreduce."""
    u_local = d_u_curr.copy_to_host()[1:-1, 1:-1]
    local_energy = float(np.sum(u_local) * DX * DY)
    global_energy = float(cart_comm.allreduce(local_energy, op=MPI.SUM))

    if d_u_prev is None:
        local_l2_sq = float(np.sum(u_local * u_local) * DX * DY)
    else:
        u_prev_local = d_u_prev.copy_to_host()[1:-1, 1:-1]
        res = (u_local - u_prev_local) / DT
        local_l2_sq = float(np.sum(res * res) * DX * DY)

    global_l2 = math.sqrt(float(cart_comm.allreduce(local_l2_sq, op=MPI.SUM)))
    return global_energy, global_l2


def run_ptx_audit():
    """Inspect compiled PTX assembly of tiled_stencil_kernel for shared memory & barrier instructions."""
    asm_dict = tiled_stencil_kernel.inspect_asm()
    ptx_text = "\n".join(asm_dict.values())
    st_shared_f64 = sum(1 for line in ptx_text.splitlines() if "st.shared.f64" in line)
    ld_shared_f64 = sum(1 for line in ptx_text.splitlines() if "ld.shared.f64" in line)
    bar_sync = sum(1 for line in ptx_text.splitlines() if "bar.sync" in line)
    passed = (st_shared_f64 > 0) and (ld_shared_f64 > 0) and (bar_sync > 0)
    return {
        "ptx_audit_passed": passed,
        "st_shared_f64": st_shared_f64,
        "ld_shared_f64": ld_shared_f64,
        "bar_sync": bar_sync,
    }


def run_solver(github_user: str):
    ctx = init_distributed_context(github_user)
    cart_comm = ctx["cart_comm"]
    rank = ctx["rank"]

    u0_host, kappa_host, source_host = initialize_subdomains(ctx)

    local_kappa_max = float(np.max(kappa_host[1:-1, 1:-1]))
    local_kappa_min = float(np.min(kappa_host[1:-1, 1:-1]))
    kappa_max = float(cart_comm.allreduce(local_kappa_max, op=MPI.MAX))
    kappa_min = float(cart_comm.allreduce(local_kappa_min, op=MPI.MIN))
    dt_cfl_max = (DX * DX) / (4.0 * kappa_max)

    d_kappa = cuda.to_device(kappa_host)
    d_source = cuda.to_device(source_host)
    bufs = HaloExchangeBuffers(LOCAL_NY, LOCAL_NX)

    stream_interior = cuda.stream()
    stream_boundary = cuda.stream()

    # Warmup JIT compilation across all kernels before timing
    d_warm_a = cuda.to_device(u0_host)
    d_warm_b = cuda.to_device(u0_host)
    for _ in range(3):
        step_serialized(ctx, d_warm_a, d_warm_b, d_kappa, d_source, bufs, stream_boundary)
        d_warm_a, d_warm_b = d_warm_b, d_warm_a
        step_overlapped(ctx, d_warm_a, d_warm_b, d_kappa, d_source, bufs, stream_interior, stream_boundary)
        d_warm_a, d_warm_b = d_warm_b, d_warm_a
    cuda.synchronize()

    # --------------------------------------------------------------------------
    # 1. Serialized Execution Benchmark (400 steps)
    # --------------------------------------------------------------------------
    d_u_curr = cuda.to_device(u0_host)
    d_u_next = cuda.to_device(u0_host)

    cart_comm.Barrier()
    ev_ser_start = cuda.event()
    ev_ser_end = cuda.event()
    ev_ser_start.record(stream_boundary)

    for _ in range(STEPS):
        step_serialized(ctx, d_u_curr, d_u_next, d_kappa, d_source, bufs, stream_boundary)
        d_u_curr, d_u_next = d_u_next, d_u_curr

    ev_ser_end.record(stream_boundary)
    ev_ser_end.synchronize()
    local_ser_ms = float(cuda.event_elapsed_time(ev_ser_start, ev_ser_end))
    serialized_ms = float(cart_comm.allreduce(local_ser_ms, op=MPI.MAX))
    u_serial_final = d_u_curr.copy_to_host()[1:-1, 1:-1]

    # --------------------------------------------------------------------------
    # 2. Dual-Stream Overlapped Execution & Distributed Energy/Residual Trace
    # --------------------------------------------------------------------------
    d_u_curr = cuda.to_device(u0_host)
    d_u_next = cuda.to_device(u0_host)

    energy_trace = []
    e0, l2_0 = compute_global_metrics(cart_comm, d_u_curr, d_u_prev=None)
    energy_trace.append({"step": 0, "global_energy": e0, "l2_norm": l2_0})

    cart_comm.Barrier()
    ev_ovl_start = cuda.event()
    ev_ovl_end = cuda.event()
    ev_ovl_start.record(stream_interior)

    for step in range(1, STEPS + 1):
        step_overlapped(
            ctx,
            d_u_curr,
            d_u_next,
            d_kappa,
            d_source,
            bufs,
            stream_interior,
            stream_boundary,
        )
        # Note: after step_overlapped, d_u_next holds step `step`, d_u_curr holds `step - 1`
        if step % 100 == 0:
            # Pause timer briefly around diagnostic Allreduce so timings remain pure
            ev_ovl_end.record(stream_interior)
            ev_ovl_end.synchronize()
            e_s, res_l2_s = compute_global_metrics(cart_comm, d_u_next, d_u_prev=d_u_curr)
            energy_trace.append({"step": step, "global_energy": e_s, "l2_residual": res_l2_s})
            if step < STEPS:
                # Accumulate split segment timings accurately
                pass
        d_u_curr, d_u_next = d_u_next, d_u_curr

    # Run a clean, uninterrupted 400-step timing pass for exact overlapped_ms
    d_u_curr = cuda.to_device(u0_host)
    d_u_next = cuda.to_device(u0_host)
    cart_comm.Barrier()
    ev_ovl_start.record(stream_interior)
    for _ in range(STEPS):
        step_overlapped(
            ctx,
            d_u_curr,
            d_u_next,
            d_kappa,
            d_source,
            bufs,
            stream_interior,
            stream_boundary,
        )
        d_u_curr, d_u_next = d_u_next, d_u_curr
    ev_ovl_end.record(stream_interior)
    ev_ovl_end.synchronize()

    local_ovl_ms = float(cuda.event_elapsed_time(ev_ovl_start, ev_ovl_end))
    overlapped_ms = float(cart_comm.allreduce(local_ovl_ms, op=MPI.MAX))
    u_overlap_final = d_u_curr.copy_to_host()[1:-1, 1:-1]

    # --------------------------------------------------------------------------
    # 3. Parity Verification & PTX Assembly Audit
    # --------------------------------------------------------------------------
    local_max_diff = float(np.max(np.abs(u_serial_final - u_overlap_final)))
    parity_max_diff = float(cart_comm.allreduce(local_max_diff, op=MPI.MAX))
    assert parity_max_diff < 1e-12, f"Parity check failed: max diff {parity_max_diff} >= 1e-12"

    ptx_report = run_ptx_audit()
    assert ptx_report["ptx_audit_passed"], f"PTX audit failed: {ptx_report}"

    # Print per-rank topology verification
    for r in range(ctx["size"]):
        if rank == r:
            print(
                f"[Rank {rank}] coords=({ctx['row_rank']},{ctx['col_rank']}) | "
                f"neighbors(U={ctx['nbr_up']}, D={ctx['nbr_down']}, "
                f"L={ctx['nbr_left']}, R={ctx['nbr_right']}) | local_shape={u0_host.shape}"
            )
        cart_comm.Barrier()

    if rank == 0:
        speedup = serialized_ms / overlapped_ms if overlapped_ms > 0 else 0.0
        manifest = {
            "GITHUB_USERNAME": github_user.strip(),
            "INSTANCE_SEED": ctx["instance_seed"],
            "gpu_name": ctx["gpu_name"],
            "gpu_uuid": ctx["gpu_uuid"],
            "mpi_ranks": ctx["size"],
            "cart_dims": ctx["dims"],
            "global_grid": [GLOBAL_NY, GLOBAL_NX],
            "local_interior_grid": [LOCAL_NY, LOCAL_NX],
            "local_padded_grid": [PADDED_NY, PADDED_NX],
            "steps": STEPS,
            "dx": DX,
            "dy": DY,
            "dt": DT,
            "dt_cfl_max": dt_cfl_max,
            "kappa_min": kappa_min,
            "kappa_max": kappa_max,
            "lambda_damp": LAMBDA_DAMP,
            "serialized_ms": round(serialized_ms, 4),
            "overlapped_ms": round(overlapped_ms, 4),
            "speedup": round(speedup, 4),
            "parity_max_diff": parity_max_diff,
            "ptx_audit_passed": ptx_report["ptx_audit_passed"],
            "ptx_counts": {
                "st.shared.f64": ptx_report["st_shared_f64"],
                "ld.shared.f64": ptx_report["ld_shared_f64"],
                "bar.sync": ptx_report["bar_sync"],
            },
            "energy_trace": energy_trace,
        }

        with open("submission_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print("\n=== DISTRIBUTED MPI + CUDA SOLVER TELEMETRY ===")
        print(f"GITHUB_USERNAME   : {manifest['GITHUB_USERNAME']}")
        print(f"GPU Device        : {manifest['gpu_name']} ({manifest['gpu_uuid']})")
        print(f"INSTANCE_SEED     : {manifest['INSTANCE_SEED']}")
        print(f"MPI Topology      : {manifest['mpi_ranks']} ranks in {manifest['cart_dims']} Cartesian grid")
        print(f"Diffusivity Range : kappa_min = {kappa_min:.8f}, kappa_max = {kappa_max:.8f}")
        print(f"Time-Step & CFL   : dx = {DX:.8e}, dt = {DT:.8e}, dt_max = {dt_cfl_max:.8e} (ratio = {DT/dt_cfl_max:.4f})")
        print(
            f"PTX Audit         : passed={ptx_report['ptx_audit_passed']} | "
            f"st.shared.f64={ptx_report['st_shared_f64']}, "
            f"ld.shared.f64={ptx_report['ld_shared_f64']}, "
            f"bar.sync={ptx_report['bar_sync']}"
        )
        print(f"Parity Max Diff   : {parity_max_diff:.6e} (< 1e-12)")
        print(f"Serialized Time   : {serialized_ms:.4f} ms")
        print(f"Overlapped Time   : {overlapped_ms:.4f} ms (speedup = {speedup:.4f}x)")
        print("\n--- Global Energy & L2 Residual Trace (MPI.Allreduce) ---")
        for entry in energy_trace:
            if entry["step"] == 0:
                print(f"Step {entry['step']:3d} | Energy = {entry['global_energy']:.10e} | Initial L2 Norm = {entry['l2_norm']:.10e}")
            else:
                print(f"Step {entry['step']:3d} | Energy = {entry['global_energy']:.10e} | L2 Residual = {entry['l2_residual']:.10e}")
        print("\nSaved submission_manifest.json successfully.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python3 distributed_cuda_solver.py <GITHUB_USERNAME>")
    run_solver(sys.argv[1])
