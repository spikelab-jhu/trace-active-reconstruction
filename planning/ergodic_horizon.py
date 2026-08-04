import numpy as np
import torch
import time
from collections import deque

from .plan_base import PlanBase
from .utils import wp2path, cal_flight_time, rotation_from_z_batch
from .kernel_ergodic import (
    kernel_ergodic_metric_pos,
    kernel_ergodic_metric_pos_footprint_depletion,
    optimize_trajectory,
    trilinear_sample_field,
)
from utils.common import TextColors


class ErgodicHorizon(PlanBase):
    """Ergodic horizon planner based on the kernel ergodic metric.

    Follows Sun et al., "Fast Ergodic Search With Kernel Functions"
    (IEEE T-RO 2025): the ergodic metric is the sum of an information
    maximization term (MLE on phi along the trajectory) and a
    self-correlation term (Gaussian kernel between pairs of waypoints).
    Trajectory optimization is gradient-based (autograd + Adam) with
    per-plan-call warm-start for receding-horizon control.

    Decision variable is a (K, 5) tensor of per-step deltas
    [dx, dy, dz, dyaw, dpitch]; the trajectory is the cumulative sum
    starting from the current pose.
    """

    horizon_mode = True

    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.horizon_waypoints = cfg.horizon_waypoints
        self.max_step = cfg.max_step
        self.preview_pause = cfg.preview_pause
        self.batch_train_steps = cfg.batch_train_steps
        self.gaussian_samples_per_horizon = cfg.gaussian_samples_per_horizon

        # info-map params. version="v1" -> legacy 3-state (unexplored+frontier
        # +(1-conf)). version="v2" -> 4-state w/ explored_unbuilt + well_built
        # gating. v2-only params have safe defaults so v1 yamls stay valid.
        self.info_map_version = str(getattr(cfg, "info_map_version", "v1"))
        self.alpha_unexplored = cfg.alpha_unexplored
        self.alpha_frontier = cfg.alpha_frontier
        self.alpha_unbuilt = getattr(cfg, "alpha_unbuilt", 2.0)
        self.beta_gauss = cfg.beta_gauss
        self.well_built_count_thresh = getattr(cfg, "well_built_count_thresh", 3)
        self.well_built_conf_thresh = getattr(cfg, "well_built_conf_thresh", 0.8)
        self.info_aggregate_radius = cfg.info_aggregate_radius

        # Footprint depletion (off by default — falls back to
        # static-φ kernel metric). When enabled, the MLE term is replaced
        # with -2/K · Σ φ_k(pos_k) where φ_k is φ_0 multiplied by the
        # causal depletion product of soft-cone visibilities at prior
        # waypoints. See planning/info_dynamics.py.
        self.footprint_depletion_enabled = bool(
            getattr(cfg, "footprint_depletion_enabled", False)
        )
        self.footprint_depletion_rate = float(
            getattr(cfg, "footprint_depletion_rate", 0.6)
        )
        # Footprint-overlap depletion: see planning/info_dynamics.py.
        # The Gaussian bandwidth controls when two sampled surface regions
        # count as overlapping.
        self.footprint_sigma = float(
            getattr(cfg, "footprint_sigma", 0.25)
        )
        # Down-weight footprint samples behind walls / in unobservable
        # space using dist_unsafe. Mirrors gaze_safe_decay; default matches
        # gaze. Set to 0 to disable gating.
        self.footprint_safe_decay = float(
            getattr(cfg, "footprint_safe_decay", 15.0)
        )

        # iLQR / trajopt params
        self.ilqr_iters = cfg.ilqr_iters
        self.ilqr_lr = cfg.ilqr_lr
        self.ilqr_lr_decay = cfg.ilqr_lr_decay
        self.kernel_sigma = cfg.kernel_sigma
        self.pitch_max = cfg.pitch_max

        # running-cost weights
        self.gaze_depths = list(cfg.gaze_depths)
        self.gaze_lambda = cfg.gaze_lambda
        # Decay rate for the gaze "safe weight" exp(-decay * dist_unsafe).
        # 50.0 is sharp: a sample 5cm into unsafe space already gets ~8% weight,
        # so gaze stops crediting "looking through walls". Lower to soften.
        self.gaze_safe_decay = float(getattr(cfg, "gaze_safe_decay", 50.0))
        self.collision_lambda = cfg.collision_lambda

# ====collision segment samples====
        self.collision_segment_samples = int(
            getattr(cfg, "collision_segment_samples", -1)
        )
# ====collision segment samples====

        self.safety_truncate_enabled = bool(
            getattr(cfg, "safety_truncate_enabled", False)
        )

        self.ctrl_reg_pos = cfg.ctrl_reg_pos
        self.ctrl_reg_yaw = cfg.ctrl_reg_yaw
        self.ctrl_reg_pitch = cfg.ctrl_reg_pitch
        self.step_barrier_lambda = cfg.step_barrier_lambda
        self.pitch_barrier_lambda = cfg.pitch_barrier_lambda
        self.bbox_barrier_lambda = cfg.bbox_barrier_lambda

        # phi normalization. "sum" (default, legacy probability distribution)
        # dilutes max(phi) to ~1/N_free_voxels on big scenes, making MLE term
        # 50-100x weaker than self-correlation. "max" keeps max(phi)=1 so
        # MLE actually drives trajectory toward info.
        self.phi_norm = str(getattr(cfg, "phi_norm", "sum"))

        # RHC warm-start state (persists across plan() calls)
        self.u_nominal = None

        # Convergence detection (geometry-discovery rate).
        # See ergodic.yaml for semantics.
        self.geometry_window = int(cfg.geometry_window)
        self.geometry_eps = float(cfg.geometry_eps)

        # +1 so we can diff oldest vs newest within a window of N.
        self._gaussian_count_hist = deque(maxlen=self.geometry_window + 1)

    @torch.no_grad()
    def plan(self, map, simulator, recorder):
        gaussian_map, voxel_map = map
        t_start = time.time()
        # Reset waypoint-pose handoff each call. Only the main branch (full
        # ergodic optimization) repopulates it; init leaves it None so the
        # mapper falls back to its linspace subsample for that one call.
        self.last_waypoint_poses = None

        if not self.init:
            nbv = torch.eye(4)
            nbv[:3, :3] = self.pose[:3, :3]
            nbv_index = voxel_map.xyz_2_index(self.pose[:3, 3])
            nbv_xyz = voxel_map.index_2_xyz([nbv_index])[0].cpu()
            nbv[:3, 3] = nbv_xyz
            waypoints = torch.stack([self.pose[:3, 3], nbv_xyz])
            camera_path, path_length = wp2path(
                self.pose[:3, :3], nbv[:3, :3], waypoints,
            )
            best_cost = None
            self.init = True
        else:
            if self.info_map_version == "v2":
                phi, src_raw = voxel_map.build_info_map_v2(
                    gaussian_map,
                    alpha_unexplored=self.alpha_unexplored,
                    alpha_frontier=self.alpha_frontier,
                    alpha_unbuilt=self.alpha_unbuilt,
                    beta_gauss=self.beta_gauss,
                    well_built_count_thresh=self.well_built_count_thresh,
                    well_built_conf_thresh=self.well_built_conf_thresh,
                    radius_voxels=self.info_aggregate_radius,
                )
            else:
                phi, src_raw = voxel_map.build_info_map_v1(
                    gaussian_map,
                    alpha_unexplored=self.alpha_unexplored,
                    alpha_frontier=self.alpha_frontier,
                    beta_gauss=self.beta_gauss,
                    radius_voxels=self.info_aggregate_radius,
                )
            dist_unsafe = voxel_map.distance_to_safe()

            waypoints, rotations, best_cost = self._ilqr_plan(
                voxel_map, phi, src_raw, dist_unsafe,
            )
            if self.safety_truncate_enabled:
                n_before = waypoints.shape[0]
                waypoints, rotations = self._safety_truncate(
                    waypoints, rotations, voxel_map,
                )
                if waypoints.shape[0] < n_before:
                    self.u_nominal = None
            nbv = torch.eye(4)
            nbv[:3, :3] = rotations[-1]
            nbv[:3, 3] = waypoints[-1]

            camera_path, path_length = wp2path(
                self.pose[:3, :3],
                nbv[:3, :3],
                waypoints,
                intermediate_rotations=rotations,
            )

            # Expose K+1 waypoint poses for mapper's K+1 mesh-save fairness fix.
            wp_poses = torch.eye(4).repeat(waypoints.shape[0], 1, 1)
            wp_poses[:, :3, :3] = rotations
            wp_poses[:, :3, 3] = waypoints
            self.last_waypoint_poses = wp_poses

        self.pose = nbv

        # update convergence signals (coverage + quality)
        self._update_convergence_signals(gaussian_map, voxel_map)

        t_plan = time.time() - t_start
        cost_str = f" cost={best_cost:.3f}" if best_cost is not None else ""
        print(
            f" {TextColors.CYAN}ErgodicHorizon(kernel):{cost_str}"
            f" {len(waypoints)} waypoints,"
            f" {camera_path.shape[0]} interpolated poses,"
            f" length={path_length:.2f}m"
            f" t_plan={t_plan:.2f}s{TextColors.RESET}"
        )

        if recorder is not None:
            t_flight = cal_flight_time(path_length, flight_speed=self.flight_speed)
            recorder.update_time("planning", t_plan)
            recorder.update_time("flight", t_flight)
            recorder.update_path(camera_path, path_length)

        return camera_path

    # ------------------------------------------------------------------
    # iLQR-style trajectory optimization on kernel ergodic metric
    # ------------------------------------------------------------------
    def _ilqr_plan(self, voxel_map, phi, src_raw, dist_unsafe):
        K = self.horizon_waypoints
        device = self.device

        cur_xyz = self.pose[:3, 3].to(device)
        cur_R = self.pose[:3, :3].to(device)

        z_cur = cur_R[:, 2]
        yaw_cur = torch.atan2(z_cur[1], z_cur[0])
        pitch_cur = torch.asin(z_cur[2].clamp(-1.0, 1.0))

        # --- RHC warm-start: shift nominal by 1 if we have one from last call
        u_init = self._warmstart_nominal(K, device)

        # --- precompute tensors used in cost ---
        phi_eff = phi + 1e-4 / phi.numel()
        if self.phi_norm == "max":
            phi_eff = phi_eff / phi_eff.max().clamp(min=1e-12)
        else:
            phi_eff = phi_eff / phi_eff.sum()
        src_raw_det = src_raw.detach()
        dist_unsafe_det = dist_unsafe.detach()

        gaze_depths = torch.tensor(
            self.gaze_depths, dtype=torch.float32, device=device,
        )  # (D,)

        # bbox bounds for the out-of-bbox barrier (cost vanishes outside since
        # grid_sample uses padding_mode="zeros"; without this term the optimizer
        # escapes the bbox where collision/erg/gaze all read as zero)
        bbox_min = voxel_map.bbox[0].to(device=device, dtype=torch.float32)
        bbox_max = voxel_map.bbox[1].to(device=device, dtype=torch.float32)

# ====collision segment samples====
        if self.collision_segment_samples < 0:
            min_voxel = float(voxel_map.size.min().item())
            n_seg_samples = max(
                1, int(np.ceil(self.max_step / (0.5 * min_voxel))) - 1
            )
        else:
            n_seg_samples = self.collision_segment_samples
        if n_seg_samples > 0:
            # interior interpolation fractions, endpoints excluded (segment
            # endpoints are waypoints, already covered by the `pos` term)
            seg_ts = torch.linspace(
                0.0, 1.0, n_seg_samples + 2, device=device, dtype=torch.float32,
            )[1:-1]
        else:
            seg_ts = None
# ====collision segment samples====

        # closure capturing current pose + maps
        def cost_fn(u):
            # u: (K, 5) -> trajectory positions, yaws, pitches via cumsum
            pos = cur_xyz.view(1, 3) + torch.cumsum(u[:, :3], dim=0)   # (K, 3)
            yaw = yaw_cur + torch.cumsum(u[:, 3], dim=0)                # (K,)
            pitch = pitch_cur + torch.cumsum(u[:, 4], dim=0)            # (K,)

            # Optical axis (+z body axis) — needed for gaze and, optionally,
            # the footprint-depletion forward model. Hoisted above cost_erg
            # so the time-varying metric can consume it without recomputation.
            cos_p = torch.cos(pitch)
            sin_p = torch.sin(pitch)
            cos_y = torch.cos(yaw)
            sin_y = torch.sin(yaw)
            z_axes = torch.stack(
                [cos_p * cos_y, cos_p * sin_y, sin_p], dim=-1,
            )  # (K, 3)

            # 1) Kernel ergodic metric on positions. The depletion variant replaces
            #    the static MLE term with one that depletes φ along the
            #    planned trajectory; self-correlation is unchanged.
            if self.footprint_depletion_enabled:
                cost_erg = kernel_ergodic_metric_pos_footprint_depletion(
                    pos, z_axes, phi_eff, voxel_map, self.kernel_sigma,
                    eta=self.footprint_depletion_rate,
                    gaze_depths=gaze_depths,
                    footprint_sigma=self.footprint_sigma,
                    dist_unsafe_flat=dist_unsafe_det,
                    safe_decay=self.footprint_safe_decay,
                )
            else:
                cost_erg = kernel_ergodic_metric_pos(
                    pos, phi_eff, voxel_map, self.kernel_sigma,
                )

            # 2) Gaze reward: mean(phi_raw) along optical axis sampled at
            #    several depths per waypoint. Optical axis is the +z body
            #    axis of the camera frame (matches rotation_from_z_batch).

            D = gaze_depths.shape[0]
            gaze_pts = pos.unsqueeze(1) + z_axes.unsqueeze(1) * gaze_depths.view(
                1, D, 1,
            )  # (K, D, 3)
            gaze_vals = trilinear_sample_field(
                src_raw_det, gaze_pts, voxel_map,
            )  # (K, D)
            # Mask out gaze samples that pierce through walls (unreachable
            # space). Without this, the agent gets credit for "looking at"
            # unexplored voxels behind a wall (alpha_unexplored mass on
            # src_raw is large there) and ends up obsessively staring at
            # walls. dist_unsafe is 0 inside the safe free region and grows
            # as samples enter occ + occ-margin + behind-occ unexplored, so
            # exp(-K * dist_unsafe) cleanly weights the gaze.
            gaze_dist = trilinear_sample_field(
                dist_unsafe_det, gaze_pts, voxel_map,
            )  # (K, D), in meters
            gaze_safe = torch.exp(-self.gaze_safe_decay * gaze_dist)
            cost_gaze = -self.gaze_lambda * (gaze_vals * gaze_safe).mean()

            # 3) Collision soft penalty via trilinear-interpolated unsafe
            #    distance at waypoint positions.
            coll = trilinear_sample_field(dist_unsafe_det, pos, voxel_map)
            cost_coll = self.collision_lambda * (coll * coll).mean()

# ====collision segment samples====
            if seg_ts is not None:
                # chain = [current (fixed) pose, wp_1, ..., wp_K] -> K segments
                chain = torch.cat([cur_xyz.view(1, 3), pos], dim=0)  # (K+1, 3)
                seg_vec = chain[1:] - chain[:-1]                     # (K, 3)
                seg_pts = (
                    chain[:-1].unsqueeze(1)
                    + seg_vec.unsqueeze(1) * seg_ts.view(1, -1, 1)
                )                                                    # (K, S, 3)
                coll_seg = trilinear_sample_field(
                    dist_unsafe_det, seg_pts, voxel_map,
                )                                                    # (K, S)
                # mean of squared penetration (~ normalized line integral),
                # independent of sample count so collision_lambda stays
                # comparable to the waypoint-only term.
                cost_coll = cost_coll + self.collision_lambda * (
                    coll_seg * coll_seg
                ).mean()
# ====collision segment samples====

            # 4) Control regularization on per-step deltas
            cost_ctrl = (
                self.ctrl_reg_pos * (u[:, :3] * u[:, :3]).sum()
                + self.ctrl_reg_yaw * (u[:, 3] * u[:, 3]).sum()
                + self.ctrl_reg_pitch * (u[:, 4] * u[:, 4]).sum()
            )

            # 5) Soft step barrier: penalize ||dpos|| > max_step quadratically
            dpos_norm = torch.norm(u[:, :3], dim=-1)
            over = torch.clamp(dpos_norm - self.max_step, min=0.0)
            cost_step = self.step_barrier_lambda * (over * over).sum()

            # 6) Soft pitch barrier: penalize |pitch| > pitch_max
            pitch_over = torch.clamp(pitch.abs() - self.pitch_max, min=0.0)
            cost_pitch = self.pitch_barrier_lambda * (pitch_over * pitch_over).sum()

            # 7) Soft bbox barrier: penalize waypoint xyz outside the voxel map
            #    bbox per-axis quadratically. This is what stops the optimizer
            #    from escaping into the zero-padded region where every other
            #    spatial cost vanishes.
            lo = torch.clamp(bbox_min - pos, min=0.0)  # (K, 3)
            hi = torch.clamp(pos - bbox_max, min=0.0)  # (K, 3)
            oob = lo + hi  # (K, 3); per axis only one of lo/hi can be nonzero
            cost_bbox = self.bbox_barrier_lambda * (oob * oob).sum()

            return (
                cost_erg + cost_gaze + cost_coll
                + cost_ctrl + cost_step + cost_pitch + cost_bbox
            )

        # run optimizer
        with torch.enable_grad():
            u_best, best_cost, _hist = optimize_trajectory(
                u_init,
                cost_fn,
                n_iters=self.ilqr_iters,
                lr=self.ilqr_lr,
                lr_decay=self.ilqr_lr_decay,
            )

        # cache for next RHC call
        self.u_nominal = u_best.detach()

        # --- reconstruct waypoints + rotations from u_best ---
        with torch.no_grad():
            u = u_best
            pos = cur_xyz.view(1, 3) + torch.cumsum(u[:, :3], dim=0)
            yaw = yaw_cur + torch.cumsum(u[:, 3], dim=0)
            # wrap yaw to [-pi, pi] for numerical hygiene
            yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
            pitch = (pitch_cur + torch.cumsum(u[:, 4], dim=0)).clamp(
                -self.pitch_max, self.pitch_max,
            )

            waypoints = torch.cat(
                [cur_xyz.cpu().unsqueeze(0), pos.cpu()], dim=0,
            )

            cos_p, sin_p = torch.cos(pitch).cpu(), torch.sin(pitch).cpu()
            cos_y, sin_y = torch.cos(yaw).cpu(), torch.sin(yaw).cpu()
            z_axes = torch.stack(
                [cos_p * cos_y, cos_p * sin_y, sin_p], dim=-1,
            )
            R_all = torch.zeros(K + 1, 3, 3)
            R_all[0] = cur_R.cpu()
            R_all[1:] = rotation_from_z_batch(z_axes)

        return waypoints, R_all, best_cost

    def _safety_truncate(self, waypoints, rotations, voxel_map):
        """Hard post-optimization safety filter.

        Truncates the K-segment trajectory at the first segment whose
        sampled occupancy probability crosses voxel_map.occ_thres --
        a deterministic backstop for cases the soft collision cost doesn't
        reliably avoid (e.g. thin obstacles): unlike collision_lambda,
        this doesn't depend on tuning being "big enough". If even the
        first segment is unsafe, stay put this horizon.
        """
        device = self.device
        occ_prob = voxel_map.voxel_states.to(device)
        K = waypoints.shape[0] - 1
        n_check = 5
        ts = torch.linspace(0.0, 1.0, n_check + 2, device=device)[1:-1]

        for k in range(K):
            p0 = waypoints[k].to(device)
            p1 = waypoints[k + 1].to(device)
            seg_pts = p0.unsqueeze(0) + ts.unsqueeze(1) * (p1 - p0).unsqueeze(0)
            check_pts = torch.cat([seg_pts, p1.unsqueeze(0)], dim=0)
            occ_at = trilinear_sample_field(occ_prob, check_pts, voxel_map)
            if bool((occ_at > voxel_map.occ_thres).any()):
                if k == 0:
                    return (
                        waypoints[:1].repeat(2, 1),
                        rotations[:1].repeat(2, 1, 1),
                    )
                return waypoints[: k + 1], rotations[: k + 1]

        return waypoints, rotations

    def _warmstart_nominal(self, K, device):
        """Shift the cached nominal control by one step; zero-pad the tail.

        First call (no cache) returns a small-random init so the optimizer
        has non-degenerate gradients.
        """
        if self.u_nominal is None or self.u_nominal.shape[0] != K:
            return torch.randn(K, 5, device=device) * 0.01

        u_prev = self.u_nominal.to(device)
        u_init = torch.zeros(K, 5, device=device)
        u_init[:-1] = u_prev[1:]
        # tail: replicate last delta scaled down, keeps trajectory continuous
        u_init[-1] = u_prev[-1] * 0.5
        return u_init

    def _update_convergence_signals(self, gaussian_map, voxel_map):
        """Push the current gaussian count into the rolling window.

        Gaussian count grows only when a depth pixel reveals a surface that
        isn't already covered; its growth rate is therefore a direct proxy
        for "is there still new geometry discoverable from this trajectory".
        """
        if gaussian_map is not None:
            n = int(gaussian_map.get_means.detach().shape[0])
            self._gaussian_count_hist.append(n)

    def should_terminate(self):
        """Return True when no new geometry has been discovered for a while.

        rate = (count_now - count_window_ago) / count_now
        saturated when rate < geometry_eps over the full window.

        Returns False until the window is populated, so we never terminate
        before having a stable readout.
        """
        if len(self._gaussian_count_hist) < self.geometry_window + 1:
            return False

        n_old = self._gaussian_count_hist[0]
        n_new = self._gaussian_count_hist[-1]
        # rate normalized by current count: if total is 100k and we added
        # 500 in the last 5 horizons, rate = 0.005 -> saturated below eps=0.01
        rate = (n_new - n_old) / max(n_new, 1)

        if rate < self.geometry_eps:
            print(
                f" {TextColors.MAGENTA}[converge]"
                f" gaussian_count {n_old} -> {n_new}"
                f" (rate={rate:.4f} < {self.geometry_eps}"
                f" over last {self.geometry_window} horizons)"
                f"{TextColors.RESET}"
            )
            return True
        return False

    def cal_utility(self, *args, **kwargs):
        raise NotImplementedError("ErgodicHorizon bypasses utility evaluation")
