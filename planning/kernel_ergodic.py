"""Kernel ergodic metric + gradient-based trajectory optimization.

Implements the kernel ergodic metric from Sun et al., "Fast Ergodic Search
With Kernel Functions" (IEEE T-RO 2025), specialized to a 3D position search
space with auxiliary per-waypoint orientation state (yaw, pitch) whose info
content enters via a gaze reward rather than the ergodic term itself.

Key pieces:
  * trilinear_sample_field : differentiable sampling of a flat voxel field
      (D*H*W,) at world-frame xyz points, implemented via F.grid_sample.
  * kernel_ergodic_metric_pos : the two-term metric
      E_theta = -2/K * sum_k phi(pos_k)
              + 1/K^2 * sum_{i,j} exp(-||pos_i - pos_j||^2 / (2 sigma^2))
  * optimize_trajectory : gradient descent on the per-step control deltas u,
      with autograd gradients and best-checkpoint Armijo-style fallback.

The module intentionally stays planner-agnostic; ErgodicHorizon builds the
actual cost_fn closure with info-map, gaze, and collision terms and hands it
to optimize_trajectory.
"""

import torch
import torch.nn.functional as F

from .info_dynamics import causal_depletion_factors


# ---------------------------------------------------------------------------
# Differentiable field sampling
# ---------------------------------------------------------------------------
def trilinear_sample_field(field_flat, xyz, voxel_map):
    """Sample a flat voxel field at world-frame points with trilinear interp.

    field_flat : (Nx*Ny*Nz,) tensor on voxel_map.device
    xyz        : (..., 3) tensor of world coords
    returns    : (...,) tensor of sampled values, differentiable wrt xyz

    Points outside the voxel map bbox are zero-padded (grid_sample with
    padding_mode='zeros'), which is the desired behavior since phi has no
    support outside the explored volume anyway.
    """
    dim = voxel_map.dim
    # match xyz's device AND dtype so subsequent arithmetic stays in xyz.dtype
    bbox = voxel_map.bbox.to(device=xyz.device, dtype=xyz.dtype)
    Nx, Ny, Nz = int(dim[0]), int(dim[1]), int(dim[2])

    # grid_sample requires field and grid to share dtype; xyz (decision variable)
    # carries gradient, so we cast field to its dtype rather than the other way around.
    field_5d = field_flat.view(1, 1, Nx, Ny, Nz).to(xyz.dtype)

    extents = bbox[1] - bbox[0]
    # normalize xyz to [-1, 1] matching align_corners=False voxel-center
    # convention: voxel-center i (0..N-1) maps to (2i+1)/N - 1
    # => xyz - bbox_min in [0, extent) gives idx_cont = (xyz-bmin)/size in
    #    [0, N), and normalized = 2*idx_cont/N - 1 = 2*(xyz-bmin)/extent - 1.
    norm = 2.0 * (xyz - bbox[0]) / extents - 1.0  # (..., 3) in (x,y,z) order

    # grid_sample 3D expects last-dim order (W, H, D) = (z, y, x) here since
    # our reshape is (Nx, Ny, Nz) with x on D, y on H, z on W.
    grid = torch.stack([norm[..., 2], norm[..., 1], norm[..., 0]], dim=-1)

    shape = grid.shape[:-1]  # (...,)
    grid_5d = grid.reshape(1, 1, 1, -1, 3)  # (1,1,1,M,3)

    sampled = F.grid_sample(
        field_5d,
        grid_5d,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    # sampled shape: (1, 1, 1, 1, M) -> (M,) -> reshape
    return sampled.view(-1).reshape(shape)


# ---------------------------------------------------------------------------
# Kernel ergodic metric (position-only 3D)
# ---------------------------------------------------------------------------
def kernel_ergodic_metric_pos(pos, phi_flat, voxel_map, sigma):
    """Kernel ergodic metric over the 3D position trajectory.

    pos      : (K, 3) tensor (trajectory positions, differentiable)
    phi_flat : (Nx*Ny*Nz,) target info distribution on voxel grid (prob,
               i.e. already normalized to sum to 1)
    sigma    : scalar Gaussian kernel bandwidth (m)

    returns  : scalar cost to minimize
    """
    K = pos.shape[0]

    # MLE term: drive trajectory into high-phi regions
    phi_at_pos = trilinear_sample_field(phi_flat, pos, voxel_map)
    mle_term = -2.0 / K * phi_at_pos.sum()

    # Self-correlation term: drive trajectory toward uniform coverage
    # pairwise squared distances (K, K)
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # (K, K, 3)
    d2 = (diff * diff).sum(dim=-1)
    kernel = torch.exp(-d2 / (2.0 * sigma * sigma))
    self_term = kernel.sum() / (K * K)

    return mle_term + self_term


# ---------------------------------------------------------------------------
# Time-varying kernel ergodic metric with footprint depletion
# ---------------------------------------------------------------------------
def kernel_ergodic_metric_pos_footprint_depletion(
    pos,
    sensor_dir,
    phi_flat,
    voxel_map,
    sigma,
    *,
    eta,
    gaze_depths,
    footprint_sigma,
    dist_unsafe_flat=None,
    safe_decay=0.0,
):
    """Time-varying kernel ergodic metric with self-consistent φ depletion.

    Replaces the static MLE term -2/K · Σ φ(pos_k) with
        -2/K · Σ φ_k(pos_k),  φ_k(x) = φ_0(x) · ∏_{j<k} (1 - η v_jk)
    where v_jk is the FOOTPRINT-overlap depletion factor (see
    info_dynamics.causal_depletion_factors). Self-correlation term
    untouched.

    Footprint-based depletion (this version) measures whether two
    waypoints look at the same surface, instead of whether their free-
    space positions overlap — better aligned with active-GS physics.

    Optional dist_unsafe gating: when both `dist_unsafe_flat` and
    `safe_decay > 0` are given, footprint samples are weighted by
    exp(-safe_decay · dist_unsafe(target)). Samples behind walls / in
    unobservable space (high dist_unsafe) get weight ≈ 0 and don't
    contribute to overlap matching, eliminating false depletion from
    "ghost surface" matches in places the agent can't actually sense.
    Mirrors the gaze-side `gaze_safe` mask used in cost_fn.

    Inputs identical to kernel_ergodic_metric_pos plus:
      sensor_dir       : (K, 3) unit camera +z body axis at each waypoint
      eta              : depletion strength per visit ∈ (0, 1)
      gaze_depths      : (D,) tensor, depths along optical axis (m)
      footprint_sigma  : Gaussian bandwidth for footprint overlap (m)
      dist_unsafe_flat : optional flat dist-unsafe field (same layout as
                         phi_flat) for footprint gating
      safe_decay       : weight = exp(-safe_decay · dist_unsafe). 0 disables.
    """
    K = pos.shape[0]
    if not isinstance(gaze_depths, torch.Tensor):
        gaze_depths = torch.as_tensor(
            gaze_depths, device=pos.device, dtype=pos.dtype,
        )
    D = gaze_depths.shape[0]

    # Compute footprint targets once; reuse for dist_unsafe gating + depletion.
    targets = pos.unsqueeze(1) + gaze_depths.view(1, D, 1) * sensor_dir.unsqueeze(1)

    if dist_unsafe_flat is not None and safe_decay > 0:
        target_safe = trilinear_sample_field(
            dist_unsafe_flat, targets, voxel_map,
        )                                                            # (K, D)
        footprint_weight = torch.exp(-safe_decay * target_safe)     # (K, D)
    else:
        footprint_weight = None

    phi0_at_pos = trilinear_sample_field(phi_flat, pos, voxel_map)   # (K,)
    deplete = causal_depletion_factors(
        targets,
        eta=eta,
        footprint_sigma=footprint_sigma,
        footprint_weight=footprint_weight,
    )                                                                # (K,)
    mle_term = -2.0 / K * (phi0_at_pos * deplete).sum()

    diff = pos.unsqueeze(0) - pos.unsqueeze(1)
    d2 = (diff * diff).sum(dim=-1)
    kernel = torch.exp(-d2 / (2.0 * sigma * sigma))
    self_term = kernel.sum() / (K * K)

    return mle_term + self_term


# ---------------------------------------------------------------------------
# Trajectory optimization (Adam on control deltas with best-checkpoint)
# ---------------------------------------------------------------------------
def optimize_trajectory(
    u_init,
    cost_fn,
    n_iters=30,
    lr=0.05,
    lr_decay=0.95,
):
    """Gradient descent on per-step deltas u.

    u_init   : (K, 5) initial guess tensor (on device). Will be cloned.
    cost_fn  : callable u -> scalar cost (must be autograd-differentiable)
    n_iters  : number of optimizer steps
    lr       : Adam learning rate
    lr_decay : per-iter multiplicative decay on lr

    returns  : (u_best, cost_best, cost_history)
    """
    u = u_init.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([u], lr=lr)

    best_cost = float("inf")
    best_u = u.detach().clone()
    history = []

    for i in range(n_iters):
        opt.zero_grad()
        cost = cost_fn(u)
        if not torch.isfinite(cost):
            break
        cost.backward()
        # clip for stability
        torch.nn.utils.clip_grad_norm_([u], max_norm=10.0)
        opt.step()

        c = cost.item()
        history.append(c)
        if c < best_cost:
            best_cost = c
            best_u = u.detach().clone()

        for g in opt.param_groups:
            g["lr"] *= lr_decay

    return best_u, best_cost, history
