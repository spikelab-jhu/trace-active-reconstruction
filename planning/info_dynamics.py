"""Differentiable footprint-depletion model for time-ergodic search.

Motivation
----------
The static kernel-ergodic metric (kernel_ergodic.py) drives the trajectory
distribution toward a *snapshot* φ_0 of the info map at horizon start.
But φ is not static: as the robot flies, sensing at each waypoint depletes
unexplored / unbuilt mass in the swept region. Optimizing against φ_0 is
therefore an underestimate of clumping cost — K waypoints all sitting in
one high-φ blob each individually score "high info" against the snapshot,
even though only the first one would actually collect that information online.

This module gives a fully-differentiable forward predictor

    φ_k ≈ φ_0(pos_k) · ∏_{j<k} (1 - η · o(k, j))

where o(k, j) measures FOOTPRINT overlap: how much the surface patch seen at
waypoint k coincides with the one already seen at waypoint j (see
`causal_depletion_factors`, the model the planner actually uses). When the
planner evaluates
its MLE term -2/K · Σ_k φ_k(pos_k), depletion gives diminishing returns to
revisits and encourages coverage of distinct informative surfaces.

Why analytic instead of grid-based depletion
--------------------------------------------
We only ever need φ_k *evaluated at pos_k*, never the full grid. Evaluating
the depletion as a closed-form sum of K-1 overlap factors at one point is

    O(K) per waypoint  ->  O(K^2) per cost_fn call  (K=20 -> 400 evals)

and avoids materializing K full voxel grids in the autograd tape (which
for a 200x200x200 grid would be ~32 MB · K · backward-buffers).
"""

import torch


# Legacy free-space cone visibility. Retired from the planner in favour of
# footprint overlap (`causal_depletion_factors`); retained because the
# shipped smoke test exercises it.
def soft_visibility_factor(
    query_xyz,
    sensor_xyz,
    sensor_dir,
    depth_min,
    depth_max,
    cone_sigma,
    axial_softness,
):
    """Soft cone-frustum visibility v(q; p, d) ∈ [0, 1].

    Gaussian falloff perpendicular to sensor_dir, sigmoid soft-gate along
    the axial coord between (depth_min, depth_max). Behind the sensor
    (depth < 0) returns ≈ 0 via the near gate.

    All inputs broadcast on the leading dims; last dim is xyz.
    """
    rel = query_xyz - sensor_xyz
    depth = (rel * sensor_dir).sum(dim=-1)              # signed axial coord
    r2_total = (rel * rel).sum(dim=-1)
    radial2 = (r2_total - depth * depth).clamp(min=0.0)

    near_gate = torch.sigmoid((depth - depth_min) / axial_softness)
    far_gate = torch.sigmoid((depth_max - depth) / axial_softness)
    axial = near_gate * far_gate
    cone = torch.exp(-radial2 / (2.0 * cone_sigma * cone_sigma))
    return axial * cone


def causal_depletion_factors(
    targets,
    eta,
    footprint_sigma,
    *,
    footprint_weight=None,
):
    """Per-waypoint depletion multiplier from FOOTPRINT overlap.

    Footprint of waypoint k = D pre-computed surface samples in 3D. For each
    pair (k, j) with k > j, depletion fires if the closest pair of footprint
    samples (one from k, one from j) is within ~footprint_sigma — i.e. the
    two waypoints look at the same surface region.

    Replaces the legacy free-space cone-overlap depletion. The cone version
    asked "is pos_k inside pos_j's forward cone?" — a free-space proxy that
    fires for any nearby trajectory points regardless of what they actually
    look at. The footprint version directly measures surface-level
    redundancy, which aligns with active-GS information physics: two
    waypoints at similar positions but looking at different surfaces are
    NOT redundant; two waypoints at different positions but looking at the
    same surface ARE redundant.

    targets          : (K, D, 3) precomputed footprint samples in world frame
    eta              : depletion strength per visit ∈ (0, 1)
    footprint_sigma  : Gaussian kernel bandwidth (m) for footprint overlap.
                       Calibrate to the spatial scale below which two
                       surface samples count as "the same surface."
    footprint_weight : optional (K, D) per-sample mask. When provided, each
                       overlap[k, j, dk, dj] is multiplied by w[k,dk]·w[j,dj]
                       before the max, so samples behind walls / in
                       unobservable space (weight ≈ 0) don't trigger false
                       depletion. Caller fills this with e.g.
                       exp(-decay · dist_unsafe(target)).
    returns          : (K,) tensor; entry k is the multiplier on φ_0(pos_k)
    """
    K, D, _ = targets.shape

    # Pairwise squared distances WITHOUT materializing the (K, K, D, D, 3)
    # diff tensor. Use the identity ||a-b||² = ||a||² + ||b||² - 2 a·b on
    # flattened (M=K*D) footprint samples — matrix-multiply is O(M·M·3)
    # compute and O(M²) memory, vs. O(M²·3) for the broadcast-subtract.
    flat = targets.reshape(K * D, 3)                          # (M, 3)
    sq = (flat * flat).sum(dim=-1)                            # (M,)
    inner = flat @ flat.t()                                   # (M, M)
    d2_flat = (sq.unsqueeze(0) + sq.unsqueeze(1) - 2.0 * inner).clamp(min=0.0)
    d2 = d2_flat.view(K, D, K, D).transpose(1, 2)             # (K, K, D, D)

    if footprint_weight is None:
        # vis[k, j] = max over (dk, dj) of exp(-d²/2σ²)
        #          = exp(-min(d²)/2σ²)  (exp is monotone in -d²; saves the
        #          (K, K, D, D) overlap tensor that amax would otherwise
        #          need to retain for backprop).
        min_d2 = d2.amin(dim=(-1, -2))                        # (K, K)
        vis = torch.exp(-min_d2 / (2.0 * footprint_sigma * footprint_sigma))
    else:
        # Weighted version: max no longer factors through, so we materialize
        # the (K, K, D, D) weighted-overlap tensor and amax it.
        overlap = torch.exp(-d2 / (2.0 * footprint_sigma * footprint_sigma))
        # weighted[k, j, dk, dj] = overlap[k, j, dk, dj] * w[k, dk] * w[j, dj]
        w_k = footprint_weight.unsqueeze(1).unsqueeze(3)      # (K, 1, D, 1)
        w_j = footprint_weight.unsqueeze(0).unsqueeze(2)      # (1, K, 1, D)
        vis = (overlap * w_k * w_j).amax(dim=(-1, -2))        # (K, K)

    # Strictly lower-triangular causal mask: only j < k contributes.
    # Multiply *inside* log1p so masked entries become log(1-0)=0 cleanly,
    # without ever producing log(0) for the diagonal.
    causal = torch.tril(
        torch.ones(K, K, device=vis.device, dtype=vis.dtype),
        diagonal=-1,
    )
    log_keep = torch.log1p(-eta * vis * causal)     # (K, K)
    return torch.exp(log_keep.sum(dim=-1))          # (K,)
