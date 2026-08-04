import torch
import time
import numpy as np
from tqdm import tqdm

from utils.operations import *
from utils.common import Camera, Mapper2Gui, FakeQueue, TextColors
from .gaussian_map import GaussianMap
from .voxel_map import VoxelMap


class IncrementalMapper:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

        # map instance
        self.gaussian_map = None
        self.voxel_map = None

        # gui related
        self.use_gui = False
        self.q_mapper2gui = FakeQueue()
        self.q_gui2mapper = FakeQueue()
        self.pause = False
        self.init = False

    @property
    def current_map(self):
        return self.gaussian_map, self.voxel_map

    def load_recorder(self, recorder):
        print("\n ----------load mission recorder----------")
        self.recorder = recorder

    def load_simulator(self, simulator):
        print("\n ----------load simulator----------")
        self.simulator = simulator

    def load_planner(self, planner):
        print("\n ----------load planner----------")
        self.planner = planner

    def init_map(self):
        print("\n ----------initialize map----------")
        self.gaussian_map = GaussianMap(self.cfg.gaussian_map, self.device)
        self.voxel_map = VoxelMap(self.cfg.voxel_map, self.simulator.bbox, self.device)

    def get_new_dataframe(self, i):
        # return way points to the nbv
        path = self.planner.plan(self.current_map, self.simulator, self.recorder)

        # for visualization only
        if self.use_gui:
            for pose in path:
                dataframe = self.simulator.simulate(pose)
                camera_frame = Camera.init_from_mapper(None, dataframe)
                self.q_mapper2gui.put(
                    Mapper2Gui(
                        current_frame=camera_frame,
                    )
                )
                time.sleep(0.05)

        # dataframe at nbv as keyframe
        dataframe = self.simulator.simulate(path[-1])
        camera_frame = Camera.init_from_mapper(i, dataframe)
        self.q_mapper2gui.put(
            Mapper2Gui(
                current_frame=camera_frame,
            )
        )
        return dataframe

    def run(self):
        torch.cuda.empty_cache()
        self.init_map()

        print(
            f"\n {TextColors.MAGENTA}----------Start Active Reconstruction----------{TextColors.RESET}"
        )

        if getattr(self.planner, "horizon_mode", False):
            self._run_horizon()
        else:
            self._run_nbv()

        print(
            f"\n {TextColors.MAGENTA}----------Finish Reconstruction Mission----------{TextColors.RESET}"
        )

    # ------------------------------------------------------------------
    # Original greedy-NBV loop
    # ------------------------------------------------------------------
    def _run_nbv(self):
        frame_id = 0
        while self.recorder is None or self.recorder.is_alive:
            if not self.q_gui2mapper.empty():
                data_gui2mapper = self.q_gui2mapper.get_nowait()
                self.pause = data_gui2mapper.flag_pause
            if self.pause:
                continue

            print(
                f"\n {TextColors.MAGENTA}----------Step {frame_id+1}----------{TextColors.RESET}"
            )

            print(f"\n {TextColors.GREEN}-----Planning:{TextColors.RESET}")
            dataframe = self.get_new_dataframe(frame_id)
            dataframe = {k: v.to(self.device) for k, v in dataframe.items()}

            print(f"\n {TextColors.GREEN}-----Mapping:{TextColors.RESET}")
            t_mapper_start = time.time()
            self.gaussian_map.update(dataframe)
            self.voxel_map.update(dataframe)
            t_mapper = time.time() - t_mapper_start
            frame_id += 1

            self.q_mapper2gui.put(
                Mapper2Gui(
                    gaussians=self.gaussian_map,
                    voxels=self.voxel_map,
                )
            )

            if self.recorder is not None:
                self.recorder.update_time("mapping", t_mapper)
                self.recorder.log()
                self.recorder.save_dataframe(dataframe, f"{frame_id:03}")
                if self.recorder.require_record:
                    self.recorder.save_map(self.gaussian_map, f"{frame_id:03}")
                    self.recorder.save_path()
            time.sleep(0.1)

    # ------------------------------------------------------------------
    # New: horizon / RHC loop.
    # 1) plan a horizon path (N waypoints, dense interpolation)
    # 2) preview in GUI for planner.preview_pause seconds
    # 3) walk the path; at each pose: simulate, update voxel map, push to GUI
    # 4) at horizon end: batch add_gaussians + single train(batch_train_steps)
    # ------------------------------------------------------------------
    def _run_horizon(self):
        frame_id = 0
        preview_pause = getattr(self.planner, "preview_pause", 2.0)
        batch_steps = getattr(self.planner, "batch_train_steps", 50)
        gaussian_samples = getattr(
            self.planner, "gaussian_samples_per_horizon", 10
        )

        while self.recorder is None or self.recorder.keep_running(frame_id):
            if not self.q_gui2mapper.empty():
                data_gui2mapper = self.q_gui2mapper.get_nowait()
                self.pause = data_gui2mapper.flag_pause
            if self.pause:
                continue

            print(
                f"\n {TextColors.MAGENTA}----------Horizon {frame_id+1}----------{TextColors.RESET}"
            )

            # ---- Plan one horizon ----
            print(f"\n {TextColors.GREEN}-----Planning:{TextColors.RESET}")
            path = self.planner.plan(
                self.current_map, self.simulator, self.recorder
            )

            # ---- Preview in GUI ----
            if self.use_gui:
                self.q_mapper2gui.put(Mapper2Gui(planned_path=path))
                print(
                    f" {TextColors.CYAN}Show planned path for"
                    f" {preview_pause:.1f}s...{TextColors.RESET}"
                )
                time.sleep(preview_pause)

            # ---- Execute: walk path, per-frame voxel update, collect frames ----
            print(f"\n {TextColors.GREEN}-----Executing horizon:{TextColors.RESET}")
            pending_frames = []
            last_cpu_frame = None
            t_mapper_total = 0.0
            pbar = tqdm(
                path, desc=" walk+voxel", ncols=80, leave=False, dynamic_ncols=False
            )
            for k, pose in enumerate(pbar):
                dataframe_cpu = self.simulator.simulate(pose)
                last_cpu_frame = dataframe_cpu

                # build GUI camera_frame on the CPU dataframe (needs .numpy())
                if self.use_gui:
                    camera_frame = Camera.init_from_mapper(None, dataframe_cpu)
                    self.q_mapper2gui.put(
                        Mapper2Gui(current_frame=camera_frame)
                    )
                    # time.sleep(0.05)

                dataframe = {
                    kk: v.to(self.device) for kk, v in dataframe_cpu.items()
                }

                # voxel map updates every frame (cheap + needed for safety).
                # Defensive: ergodic_horizon's slerp can occasionally produce
                # a near-singular rotation when consecutive waypoints have
                # ~antipodal view directions; skip such frames rather than
                # crash the mission.
                t0 = time.time()
                try:
                    self.voxel_map.update(dataframe, verbose=False)
                    pending_frames.append(dataframe)
                except torch._C._LinAlgError as e:
                    print(
                        f" {TextColors.YELLOW}[skip frame {k}: singular extrinsic"
                        f" -- {e}]{TextColors.RESET}"
                    )
                t_mapper_total += time.time() - t0
            pbar.close()

            # ---- Horizon end: batch Gaussian densify + train ----
            # wp10 patch: GS training uses K+1 waypoint-aligned frames
            # instead of np.linspace index-uniform sampling. Each gaussian_frame
            # is the pending_frame closest to a planner-output K+1 waypoint.
            # This makes GS training data consistent with the K+1 cameras
            # saved per horizon (see below), eliminating the planner-training
            # disconnect.
            waypoint_poses_for_gs = getattr(
                self.planner, "last_waypoint_poses", None
            )
            if waypoint_poses_for_gs is not None and len(pending_frames) > 0:
                import torch as _torch
                wp_xyz = waypoint_poses_for_gs[:, :3, 3].cpu()
                pending_xyz = _torch.stack(
                    [
                        f["extrinsic"][:3, 3].cpu()
                        if _torch.is_tensor(f["extrinsic"])
                        else _torch.tensor(f["extrinsic"][:3, 3])
                        for f in pending_frames
                    ]
                )
                # pairwise L2 distance, pick closest pending_frame per waypoint
                d = (
                    (wp_xyz.unsqueeze(1) - pending_xyz.unsqueeze(0))
                    .norm(dim=-1)
                )  # (K+1, N_pending)
                closest_idx = d.argmin(dim=1).tolist()
                # dedup while preserving order
                seen = set()
                closest_idx_dedup = [
                    i for i in closest_idx if not (i in seen or seen.add(i))
                ]
                gaussian_frames = [pending_frames[i] for i in closest_idx_dedup]
            elif (
                gaussian_samples > 0
                and len(pending_frames) > gaussian_samples
            ):
                # fallback for non-ergodic planners (no last_waypoint_poses)
                idxs = np.linspace(
                    0, len(pending_frames) - 1, gaussian_samples
                ).astype(int)
                gaussian_frames = [pending_frames[i] for i in idxs]
            else:
                gaussian_frames = pending_frames

            # add + train + post_processing run inline: plan() of the NEXT
            # horizon only starts after this returns, so it always sees a
            # fully-trained gaussian_map.
            print(
                f"\n {TextColors.GREEN}-----Gaussian update"
                f" ({len(gaussian_frames)}/{len(pending_frames)} frames,"
                f" {batch_steps} steps){TextColors.RESET}"
            )
            t_sgd = time.time()
            if len(gaussian_frames) > 0:
                self.gaussian_map.add_gaussians_batch(gaussian_frames)
            if batch_steps > 0:
                self.gaussian_map.train(
                    steps=batch_steps, do_post_processing=False,
                )
            # confidence must aggregate ALL N new frames -- not just the last
            # one. Pass the count so post_processing loops.
            self.gaussian_map.post_processing(
                num_new_frames=len(gaussian_frames),
            )
            # SGD time IS mission time -- credit it to mapping so mission_time
            # stays consistent with what the user sees.
            t_mapper_total += time.time() - t_sgd

            # publish keyframe (last pose) + map snapshot
            if self.use_gui and last_cpu_frame is not None:
                key_cam = Camera.init_from_mapper(frame_id, last_cpu_frame)
                self.q_mapper2gui.put(Mapper2Gui(current_frame=key_cam))
            self.q_mapper2gui.put(
                Mapper2Gui(
                    gaussians=self.gaussian_map,
                    voxels=self.voxel_map,
                )
            )

            frame_id += 1

            if self.recorder is not None:
                self.recorder.update_time("mapping", t_mapper_total)
                self.recorder.log()
                # K+1 camera save: append one camera_params entry per planner
                # waypoint, so cameras_XXX.pkl records K+1 views per horizon
                # instead of just last_cpu_frame.
                #
                # CRITICAL: do NOT call simulator.simulate(wp_pose) here.
                # habitat_simulator.simulate() consumes one np.random.normal
                # draw per call (line 119, depth noise), which corrupts the
                # global numpy RNG and breaks PSNR reproducibility against
                # Reference TRACE experiment. Downstream consumers re-render
                # RGB+depth from the trained Gaussian map and only need
                # extrinsic+intrinsic — append camera_params directly.
                waypoint_poses = getattr(
                    self.planner, "last_waypoint_poses", None,
                )
                if waypoint_poses is not None and len(waypoint_poses) > 0:
                    intrinsic_flat = (
                        self.simulator.intrinsic.view(-1).cpu().numpy().tolist()
                    )
                    for wp_pose in waypoint_poses:
                        extrinsic_flat = (
                            wp_pose.view(-1).cpu().numpy().tolist()
                        )
                        self.recorder.camera_params_list.append(
                            extrinsic_flat + intrinsic_flat
                        )
                elif last_cpu_frame is not None:
                    self.recorder.save_dataframe(
                        last_cpu_frame, f"{frame_id:03}"
                    )
                if self.recorder.should_save(frame_id):
                    self.recorder.save_map(self.gaussian_map, f"{frame_id:03}")
                    self.recorder.save_path()
            time.sleep(0.1)

            # planner-driven early termination (e.g. ergodic cost has flattened
            # near zero -> nothing left to explore). Optional hook; planners
            # without should_terminate keep running to budget.
            if getattr(self.planner, "should_terminate", lambda: False)():
                print(
                    f"\n {TextColors.MAGENTA}----------"
                    f"Planner reports convergence; ending mission early"
                    f"----------{TextColors.RESET}"
                )
                break

        # Mission ended (budget hit or planner convergence). Two cleanups:
        #   1. (optional) one polish-pass of SGD on the full keyframe set so
        #      late-added gaussians get more optimization steps; configured
        #      via planner.final_polish_steps (0 disables)
        #   2. final GUI publish so the viewer shows the fully-trained map
        #      instead of whatever was the last mid-mission snapshot
        polish_steps = int(getattr(self.planner, "final_polish_steps", 0))
        if polish_steps > 0:
            print(
                f"\n {TextColors.GREEN}-----Final polish train"
                f" ({polish_steps} steps on full keyframe set)...{TextColors.RESET}"
            )
            self.gaussian_map.train(steps=polish_steps)

        if self.use_gui:
            self.q_mapper2gui.put(
                Mapper2Gui(
                    gaussians=self.gaussian_map,
                    voxels=self.voxel_map,
                )
            )
            print(
                f"\n {TextColors.GREEN}-----Final map published to GUI"
                f"{TextColors.RESET}"
            )
