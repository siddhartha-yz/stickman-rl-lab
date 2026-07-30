# Development progress

Last updated: 2026-07-30

## Current verified status

The project is installable and runnable. The articulated PyMunk environment passes Gymnasium validation, PPO can train/save/load, headless and RGB rendering work, and deterministic stage-1 target reaching has been demonstrated. Stage-2 randomized-target training also improves generalization. The original full box-plus-platform course now has a verified deterministic majority-success policy: 54/80 successes across two independent 40-episode target sets, with 100% strict route completion. The browser interface now starts a separate PPO process from random weights, displays current Python/PyMunk body states and live metrics, and controls pause, resume, checkpoint save, and stop. Farthest-target robustness and natural upright walking remain unsolved.

## Environment and installation

- OS: Windows 11
- Python: 3.11.9
- Installed stack: Gymnasium 1.3.0, PyMunk 7.3.0, Pygame 2.6.1, Stable-Baselines3 2.9.0, PyTorch 2.13.0.
- The first dependency installation failed because the deeply nested workspace exceeded the Windows legacy path limit while installing PyTorch.
- Verified fix: install the environment at `C:\rlv`, then create `.venv` as a directory junction to that short path.

## Implemented systems

- Closed room with floor, walls, and ceiling.
- Ten-body stickman: head, torso, paired upper arms, forearms, thighs, and shins.
- Passive limited neck and eight actuated, angle-limited rotary joints.
- Continuous target-angular-velocity control with configurable speed and motor force.
- Normalized 49-value observation vector.
- Decomposed and individually logged reward components.
- Configuration-driven stages 0-5 plus easy/medium obstacle sub-curricula.
- Modular box, platform, wall, slope, and trench obstacles.
- Collision-free random target sampling; targets are rejected if they overlap obstacles or trench gaps.
- Configurable four-frame action repeat.
- Runtime reward annealing across stages.
- Environment override files for training, evaluation, validation, demos, and recording.
- PPO resume with synchronized learning rate, optimizer, clipping, discount, entropy, and variance settings.
- Periodic/latest/best checkpoints, final-vs-best evaluation, early stopping support, TensorBoard, config snapshots, runtime metadata, plots, trajectories, and GIFs.
- Version-2 trajectories containing every rigid body's position, angle, geometry, target, obstacles, waypoints, per-frame metrics, seed, and SHA-256 provenance.
- React/Vite live training console with from-scratch process creation, current PyMunk state rendering, rolling metrics and PPO losses, pause/resume/stop/save controls, run history, and FastAPI endpoints.
- Atomic worker state under `lab/runs/`, including control, status, live frame, episode/update metrics, logs, TensorBoard data, and checkpoints.
- Ruff static analysis and automated tests.

## Final verification

```text
Ruff: all checks passed
Pytest: 29 passed
Stage-0 Gymnasium/random stability: passed (1,000 decision steps)
Stage-3 Gymnasium/random stability: passed (1,000 decision steps)
PPO direct smoke train/save/reload: passed
Full train.py orchestration smoke: passed
Checkpoint final/best comparison: passed
Configuration/runtime snapshot writing: passed
RGB-array renderer: passed
Trajectory export: passed
Animated GIF export: passed
Reward annealing callback: passed; alpha reached 1.0 and final stage-4 weights were logged
```

## Training experiments

### Baselines

Before action repeat:

- Zero-action stage-1 baseline: 0% success, mean final distance 6.23.
- Random-action stage-1 baseline: 50% success over 10 episodes.

After four-frame action repeat:

- Random-action stage-1 baseline: 80% success over 10 episodes, mean final distance 0.65.

The high random baseline is expected for the deliberately unconstrained first stage. Deterministic success, completion time, distance, energy, and trajectory inspection are therefore recorded alongside success rate.

### Stage 1: fixed target

Early PPO runs produced stochastic successes but conservative deterministic means. Low entropy and variance annealing improved distance but did not solve the issue by themselves.

The decisive change was four-frame action repeat:

- Checkpoint: `checkpoints/repeat4-stage1/best/best_model.zip`
- Independent deterministic evaluation: 20/20 successes.
- Success rate: 100%.
- Mean episode length: 337 decision steps.
- Mean final distance: 0.346.
- Mean energy: 0.00459.
- Mean torso height: 0.855.

The final model from the same run later regressed to 0% deterministic success. Best-model retention and explicit final-vs-best comparison were retained because of this measured PPO regression.

### Stage 2: randomized targets

Stage-1 best policy evaluated directly on random targets:

- 45% success over 20 episodes.

After 32,768 additional stage-2 PPO steps:

- Best checkpoint: `checkpoints/stage2-random-targets/best/best_model.zip`
- Independent evaluation: 56% success over 50 random-target episodes.
- Mean final distance: 0.990.
- Mean episode length: 496.7 decision steps.
- Mean energy: 0.0177.

The final stage-2 checkpoint was weaker than the saved best checkpoint.

### Stage 3: obstacles

A serious environment bug was found during the first obstacle run: random targets could be generated inside the box or platform, making many episodes impossible. That run was stopped. Target sampling was fixed and a regression test was added.

Corrected full course:

- Target is sampled beyond both obstacles.
- Stage-2 best deterministic baseline: 0% over 20 episodes, mean final distance 3.09.
- Random-action baseline: 50% over 10 episodes.
- Standard PPO stochastic rollout success reached roughly 57%, but deterministic evaluation remained 0%.
- Variance-annealed fine-tuning improved deterministic mean final distance to about 2.80 but still produced 0% success.

Sub-curricula:

- Easy low box: existing stage-2 policy already achieved 90%, so it was too easy to justify training.
- Medium full-height box without platform: existing policy achieved 40% over 20 episodes.
- Standard and conservative fine-tuning both degraded deterministic performance; those runs were stopped rather than allowed to waste compute.

Current conclusion: the medium single-box course is solved, and the original full box-plus-platform course now has a verified deterministic majority-success policy documented below. The remaining Stage-3 bottleneck is robustness over the farthest target interval, especially `x >= 10.1`, rather than obstacle clearance itself.

### Stage 4 reward transition smoke

A 2,048-step resume run from stage 2 to stage 4 verified runtime reward annealing:

- `curriculum/reward_anneal_alpha` reached 1.0.
- Height, upright, foot contact, hand contact, smoothness, and energy weights reached their configured stage-4 values.
- Checkpoint save/load and evaluation completed successfully.

This validates the transition mechanism; it does not claim upright walking is solved.

## Agent iteration: obstacle rays and strict route shaping (2026-07-30)

This section records direct Agent-authored inspection, code changes, tests, training, and independent evaluation. The pre-existing `autonomous_review_round.py` background job was inspected but was not treated as Agent reasoning.

### Baseline inspection

- Active process and latest logs were checked through `local-shell-mcp`.
- Git is not installed in the workspace, so no repository diff was available. Evidence is recorded through exact changed files, config snapshots, hashes, test outputs, evaluation arrays, and generated reports.
- Pre-change baseline: Ruff passed and Pytest reported 10 passed.

### Increment 1: versioned obstacle-ray observations

Changed files:

- `src/stickman_rl/env.py`
- `configs/base.yaml`
- `configs/stage3_lidar.yaml`
- `configs/stage3_medium_lidar.yaml`
- `tests/test_env.py`
- `README.md`

Implementation:

- Default observation remains 49 values, preserving all old checkpoints.
- Sensor-enabled configs append nine world-fixed obstacle-ray proximity values, producing a 58-value observation.
- Rays query actual PyMunk obstacle shapes and exclude room walls, target sensors, and the stickman's own bodies.
- Geometry probe: maximum proximity increased from 0.389 at spawn to 0.746 after translating the intact pose closer to the box.
- Existing stage-1 checkpoint remained compatible and retained 5/5 deterministic success.

Same-seed 65,536-step comparison on the medium box course:

| Condition | Observation | Deterministic success, 30 episodes | Mean final distance | Mean reward |
|---|---:|---:|---:|---:|
| No rays | 49 | 0% | 3.655 | 9.343 |
| Nine rays | 58 | 0% | 4.313 | 6.281 |

Conclusion: the sensor implementation is valid and backward compatible, but rays alone did not break the obstacle local optimum in this run. Exact metrics are stored in `reports/agent-medium-lidar-seed17-comparison.json`.

### Increment 2: route-aware progress and strict obstacle clearance

Changed files:

- `src/stickman_rl/env.py`
- `src/stickman_rl/evaluation.py`
- `configs/base.yaml`
- `configs/stage3_medium_lidar_waypoint.yaml`
- `tests/test_env.py`
- `README.md`

Implementation:

- Configurable navigation waypoints can temporarily replace the final target in observation and dense progress reward.
- Final success and `final_distance` always remain tied to the red target.
- Evaluation now records route-completion rate, mean waypoints completed, and maximum torso x-position.
- The first waypoint implementation used `within radius OR past x-boundary`. It falsely reported 100% route completion while mean maximum torso x was only 5.106, still before the box was cleared.
- The bug was corrected: when `advance_x` is configured, the torso must actually cross that boundary. The waypoint was moved behind and above the box to `[6.0, 1.25]`, with clearance threshold `x >= 5.85`.
- A regression test verifies that entering the waypoint radius at `x=5.5` does not complete the route.

Training evidence:

- Starting policy under the strict rule: 0% route completion, mean maximum torso x 5.479.
- Continued PPO evaluation success rates across checkpoints: 0%, 90%, 40%, 100%, 0%, 100%, 20%, 100%.
- This confirms severe PPO regression even after the task becomes learnable.
- Frozen verified checkpoint: `checkpoints/agent-medium-lidar-clearance-seed17-65k/best_snapshot_90112.zip`.
- Current `best/best_model.zip` is byte-identical to that snapshot.
- SHA-256: `1d4431f1e3667dba774c80f1be7b1c168dfb7f7d2f851dc4a79229fe2680ec2e`.

Independent deterministic evaluation over 30 fresh medium-course episodes:

- Success: 30/30, 100%.
- Route completion: 100%.
- Mean episode length: 333.5 decisions.
- Mean final distance: 0.436.
- Mean reward: 161.737.
- Mean energy: 0.00941.
- Mean maximum torso x: 7.917.

Final verification after all code changes:

```text
Ruff: all checks passed
Pytest: 13 passed
Medium lidar waypoint Gymnasium/random stability: passed, 1,000 decision steps
Legacy stage-1 checkpoint: 5/5 deterministic success
```

This solves the medium single-box deterministic curriculum. It does not solve the original full box-plus-platform course.

## Agent iteration: full-course demonstrations and actor stabilization (2026-07-30)

This section records direct Agent-authored code changes, data collection, model transformation, independent evaluation, and failed branches after the medium-course solution.

### Full route transfer and PPO failure modes

- Added a second strict waypoint for the platform in `configs/stage3_lidar_waypoints.yaml`.
- Checkpoint `checkpoints/agent-full-lidar-waypoints-seed17-65k/best_snapshot_114688.zip` completed both obstacle waypoints in 20/20 episodes and reached mean maximum torso x 8.166, but final-target success remained 0%.
- Multiple PPO continuation branches were tested with target-relative waypoints, near-target curricula, smaller learning rates down to `1e-5`, fewer epochs, narrower clipping, and `target_kl` limits.
- These branches repeatedly destroyed the already learned obstacle route after the first update. Failed branches were stopped and retained in logs rather than presented as progress.
- A post-route target-progress multiplier was added without changing the 58-value observation schema. The source policy retained 100% route completion, but PPO fine-tuning still regressed.

### Successful-rollout distillation evidence

Added:

- `scripts/distill_successful_rollouts.py`
- reusable `.npz` input datasets
- phase-balanced batch sampling through `--final-batch-fraction`
- actor-only source anchoring
- `tests/test_distillation_sampling.py`

Measured comparisons on the same 40 deterministic episodes:

| Model | Success | Route completion | Mean final distance |
|---|---:|---:|---:|
| Full-route PPO source | 0/40 | 100% | 1.566 |
| Two-success-trajectory smoke distillation | 0/40 | 100% | 1.267 |
| Twelve-trajectory, 20-epoch distillation | 0/40 | 0% | 3.344 |

The overtrained distillation dataset contained 4,416 samples but only 656 final-phase samples (14.9%). It regressed to one completed waypoint. This motivated explicit phase-balanced batches and conservative actor interpolation rather than larger unconstrained distillation updates.

Verified complete-distillation artifacts:

- Model: `checkpoints/distill-full-16/model.zip`
- Model SHA-256: `8ce678692c10fccfe756d7805bdaf69c3221d53c9cec3a7ed6358ef84dfffdf8`
- Dataset: `trajectories/distill-full-16.npz`
- Dataset SHA-256: `6111398a3b4836382133b58902f6dbcea14ace75ef6d3dd2ec2f02bcf80fda4b`
- Report: `reports/distill-full-16.json`
- Report SHA-256: `086ce41108b7d9c29d288238860180c9b55499b5e8bdf0052e34ed5969f92781`

### Far-target demonstrations and actor interpolation

Detailed evaluation showed the first distilled deterministic actor succeeded only for nearer targets. A far-target curriculum (`x=9.2` to `10.5`) confirmed:

- Deterministic source: 0% far-target success.
- Stochastic source: 19/60 far-target successes, spanning target x 9.439 to 10.440.

A phase-balanced collection run gathered 12 real far-target successes in 36 attempts:

- Dataset: `trajectories/distill-far-balanced-v1.npz`
- Samples: 4,124 total, 909 post-route samples.
- Collection success rate: 33.3%.
- Dataset SHA-256: `875e2ad171f5c77e1b3120ffb5bc9c908df10a41b0bd429e1574fb1f9814dfa2`
- Distilled model SHA-256: `614737654c11814e9f1e98d1d448b48231de217a0ab288335de02f88466d2064`
- Training report SHA-256: `e76a8704f9cae45dc80fd7ee00334e1dcb9613a48c73152c69a11933ab716ede`

The complete distilled actor again regressed. `scripts/interpolate_actor.py` was therefore added to move only actor parameters by a controlled fraction while preserving the critic, action variance, and source checkpoint. Tests verify that only the policy network and action head change.

The policy basin was extremely discontinuous: parameter coefficients differing by 0.0005 could change from one-waypoint failure to majority success. A coarse-to-ultrafine scan found coefficient `0.4005` along the far-demonstration actor direction.

### Recommended full-course checkpoint

Recommended model:

- `checkpoints/stage3-full-recommended/model.zip`
- SHA-256: `6db7068817e096452f6c63b496abedc8801591f48ce71be628c2c475de11ecc4`
- Environment: `configs/stage3_lidar_waypoints_goalboost.yaml`
- Provenance/evaluation report: `reports/stage3-full-breakthrough.json`

Independent deterministic evaluation:

| Target seed range | Success | Route completion | Mean final distance | Mean reward |
|---|---:|---:|---:|---:|
| 1000-1039 | 30/40 (75.0%) | 100% | 0.491 | 165.419 |
| 5000-5039 | 24/40 (60.0%) | 100% | 0.553 | 150.722 |
| Combined | 54/80 (67.5%) | 100% | - | - |

Target-bin diagnostics across both sets:

- `x < 9.5`: 100% success in every populated bin.
- `9.5 <= x < 9.8`: approximately 71-75% success.
- `9.8 <= x < 10.1`: mixed success.
- `x >= 10.1`: 0% success in the measured sets.

The model therefore solves obstacle clearance and most targets, but the farthest interval remains an explicit unresolved subproblem. A two-expert target router was also implemented and tested, but performed worse than this single checkpoint and is not recommended.

### Verification after these changes

```text
Ruff: all checks passed
Pytest: 23 passed
Full route/goal-boost environment: 1,000 random decision steps passed
Detailed deterministic and stochastic evaluation JSON/CSV export: passed
Actor-only interpolation regression test: passed
Phase-balanced distillation batch tests: passed
Target-position expert routing test: passed
```

## Experiment observer GUI (2026-07-30)

The first usable browser interface was implemented as an offline, reproducible experiment observer. It deliberately replays body states recorded by the Python/PyMunk environment instead of reproducing the dynamics in JavaScript, so the visualized pose is the same state that generated the recorded rewards and metrics.

### Architecture and files

- `frontend/`: React 19 and Vite production interface.
- `scripts/lab_server.py`: FastAPI experiment manifest and trajectory API plus production static-file serving.
- `scripts/run_lab.py`: single-command production launcher.
- `scripts/export_lab_assets.py`: fixed-seed batch trajectory exporter driven by the manifest.
- `lab/experiments.json`: ordered learning timelines and verified experiment metadata.
- `scripts/record_trajectory.py`: version-2 physics trajectory exporter retaining all legacy array keys.
- `tests/test_lab_observer.py`: manifest, path, serialization, and API contract coverage.

### Implemented interface

- Eight-run experiment timeline grouped into Stage-1 learning and full double-obstacle evolution.
- Actual ten-body stickman reconstruction from recorded PyMunk circle, segment, and polygon geometry.
- Room, obstacles, red target, strict waypoints, active waypoint state, success overlay, and optional recent torso trail.
- Play/pause, frame stepping, scrubber, and 0.25x to 4x playback.
- Synchronized cumulative-reward, target-distance, and torso-height curves.
- Per-frame eight-joint action display.
- Independent verified metrics alongside trajectory-local results.
- Seed, stage, checkpoint path, environment path, model/config/trajectory SHA-256, and trajectory format provenance.
- Responsive desktop and narrow-screen layout.

The manifest's eight version-2 trajectories were regenerated from real environments and checkpoints. Representative fixed-seed evidence:

| Timeline entry | Steps | Success | Final distance | Maximum torso x | Total reward |
|---|---:|---:|---:|---:|---:|
| Stage-1 random baseline | 397 | yes | 0.368 | 11.128 | 163.64 |
| Stage-1 16,384-step checkpoint | 337 | yes | 0.346 | 7.721 | 163.16 |
| Full-route source | 900 | no | 1.715 | 8.165 | 38.09 |
| Two-demo distillation smoke | 900 | no | 1.424 | 8.276 | 49.35 |
| Actor interpolation 0.995 | 900 | no | 0.894 | 8.762 | 66.49 |
| Far actor 0.4000 | 900 | no | 0.979 | 8.958 | 62.44 |
| Recommended full-course model | 787 | yes | 0.530 | 9.198 | 197.92 |

The random Stage-1 baseline can succeed because Stage 1 is intentionally unconstrained and has a historically high random-action baseline; it remains useful for comparing motion quality and completion time rather than as evidence that learning is unnecessary.

### Verification evidence

```text
Frontend Vitest: 3 passed
Frontend Vite production build: passed
Python Ruff: all checks passed
Python Pytest: 26 passed
HTTP /: 200
HTTP /api/experiments: 200, 8/8 trajectories available
HTTP /api/experiments/full-recommended/trajectory: 200, 787 frames
System Edge browser: loaded production UI, 8 timeline items, 1198x621 Canvas
Playback interaction: frame advanced from 1/787 to 10/787
Experiment switching: random baseline and recommended model both loaded
Mobile Edge viewport: 390px layout, 348x261 Canvas, no horizontal overflow
Browser console/page errors: none
```

Browser acceptance screenshot:

- `reports/lab-observer-home.png`
- `reports/lab-observer-mobile.png`

Production launch:

```bat
C:\rlv\Scripts\python.exe scripts\run_lab.py
```

Open `http://127.0.0.1:8000`.

This offline observer was the first interface iteration. It has now been superseded as the main page by the live training console documented below; its trajectory format and APIs remain useful for reproducible post-run analysis.

## Live from-scratch training console (2026-07-30)

The main browser page was rebuilt after the offline replay interface was judged insufficient. The new interface creates and controls actual PPO jobs. Each job starts `scripts/live_train_worker.py` in a separate Python process with no resume checkpoint, so the policy network is randomly initialized.

### Architecture

- `StickmanReachEnv.live_snapshot()` exports the current ten-body positions, angles, shape geometry, target, obstacles, waypoints, episode info, and action names without advancing or rendering the environment.
- `scripts/live_train_worker.py` creates PPO, writes live frames/status/metrics atomically, handles pause/resume/stop/save requests inside an SB3 callback, creates periodic/manual/final checkpoints, and records TensorBoard output.
- `scripts/training_api.py` validates configs, creates jobs, launches worker subprocesses, lists run history, returns current frames and metrics, and writes control commands.
- `frontend/src/App.jsx` was replaced with a training form, current-episode Canvas, real-time progress and metric cards, reward/success/distance/loss curves, joint actions, session details, and process controls.
- `configs/train_live.yaml` uses one environment and 512-step rollouts to produce more frequent UI updates than the production-tuned 2,048-step config.

### Verified control path

A 64-step direct worker smoke completed from random weights and produced a 10-body frame, two PPO updates, and `final_model.zip`. A separate HTTP/API run verified the complete control loop:

```text
Created: ui-20260730-111758-f0e257
Running: 32 steps, 10 live rigid bodies
Paused: stable at 65 steps for 0.8 seconds
Manual checkpoint: manual-65.zip
Resumed: advanced to 96 steps
Stopped: 130 steps
Stopped checkpoint: stopped_model.zip
```

System Edge then performed the same workflow through visible page buttons using a 5,000-step request:

```text
Pause: 72 / 5000 remained 72 / 5000 after 0.8 seconds
Resume: advanced to 96 / 5000
Stop: page reached status-stopped
Canvas: 1112 x 584 CSS pixels
Browser console errors: none
```

The page originally used `step=64` on the timestep input, which silently prevented submission of the default value 100,000 through native browser validation. This was found during browser acceptance and corrected to allow any integer at least 64.

### Verification after the live-console change

```text
Python Ruff: all checks passed
Python Pytest: 29 passed
Frontend Vitest: 3 passed
Frontend Vite production build: passed
Direct live worker smoke: passed
HTTP create/pause/save/resume/stop: passed
System Edge UI create/pause/save/resume/stop: passed
Browser console errors: none
```

Acceptance screenshot: `reports/live-training-console.png`.

### Live frame-rate upgrade

The initial live console was limited by a 280 ms browser poll and 80 ms trainer snapshot interval, producing only about 3-5 visible source frames per second. The transport was replaced in stages:

- browser status/physics polling was replaced by a FastAPI WebSocket push stream;
- static room and rigid-body geometry moved to one `metadata.json` file;
- compact dynamic frames fell from about 5.9 KB to about 1.8 KB;
- the trainer now injects high-frequency frames directly into FastAPI memory through a localhost WebSocket, while disk snapshots remain a 5 Hz recovery fallback;
- localhost ingest explicitly bypasses system proxies, after the configured SOCKS proxy incorrectly intercepted `127.0.0.1`;
- Windows atomic JSON replacement now retries brief sharing-lock failures;
- the Canvas interpolates adjacent real trainer states with `requestAnimationFrame`.

Verified Stage-1 smoke measurement:

```text
Real physics states received: 148
Measurement duration: 5.018 seconds
Source state rate: 29.49 FPS
PPO training throughput: 201.29 steps/s
Internal stream: connected, 161 messages sent, one initial connection, no error
```

The source rate varies during PPO gradient updates because the environment does not advance while the policy optimizer is running. Browser rendering continues at the display refresh cadence by interpolating only between actual trainer states; it does not invent episode outcomes or alter physics.

## Desktop application migration (2026-07-30)

### Goal 1: native trainer transport and process controller

Acceptance criterion: a non-web Python controller must launch a randomly initialized PPO worker, receive structured metadata/frame/status/metrics events through the worker's standard output, complete a real 64-step smoke run, and save a reloadable final checkpoint without FastAPI or a browser.

Implemented:

- `scripts/live_train_worker.py` supports `--stream-stdout` line-delimited events and an explicit `--run-dir`.
- `stickman_rl.desktop.DesktopTrainingController` launches, observes, pauses, resumes, saves, stops, and snapshots one trainer directly.
- Configuration validation prevents environment YAML files from being used as PPO training configs.
- The web transport remains temporarily available until the native UI is independently verified.

Verification:

```text
Desktop controller Ruff: passed
Desktop controller tests: 2 passed
Real PPO smoke: 64/64 steps, process exit 0
Received event types: metadata, frame, status, metrics
Final checkpoint: created
```

Checkpoint commit: `f33c8411f89b04ca83dc669c0e37b971b172de29`.

### Goal 2: native desktop training console

Acceptance criterion: a Windows-native window must launch a random-weight PPO run, render the current ten-body PyMunk state without a browser, show live training metrics and action outputs, and expose pause, resume, save, and stop controls.

Implemented:

- `stickman_rl.desktop.app` provides a DPI-aware Tk desktop window with stage/config controls, run history, live rigid-body Canvas rendering, progress/metric tiles, PPO charts, action inspection, and process state.
- `scripts/run_desktop.py` is the direct launcher and includes a reproducible `--smoke` mode.
- The worker-to-desktop path is a structured stdout pipe; no FastAPI server or localhost WebSocket is required.
- Window closing explicitly stops and saves an active child trainer rather than leaving a broken stdout pipe.
- Windows DPI awareness and explicit layout constraints prevent the native window and physics room from being clipped by display scaling.
- Desktop control JSON writes retry brief Windows sharing-lock conflicts, verified after a real `WinError 5` failure was reproduced by the full suite.

Verification:

```text
Desktop UI/helper tests: 3 passed
Desktop controller lifecycle tests: 3 passed
Pause: timesteps remained unchanged for 0.4 seconds
Manual save while paused: checkpoint created
Resume: timesteps advanced
Stop: process exit 0 and stopped_model.zip created
Native UI smoke: completed, 64/64 steps
Live frames received: 8
Rigid bodies in frame metadata: 10
Desktop child process exit: 0
Full Ruff: passed
Full Pytest: 35 passed
Screenshot: reports/desktop-training-console.png
Machine-readable report: reports/desktop-smoke.json
```

Checkpoint commit: pending `goal_checkpoint.ps1` result.

## Generated artifacts

- Random/pre-training GIF: `videos/random-before.gif`
- Successful deterministic stage-1 GIF: `videos/stage1-best.gif`
- Renderer smoke GIF: `videos/renderer-smoke.gif`
- Random trajectory: `trajectories/random-before.npz`
- Successful stage-1 trajectory: `trajectories/stage1-best.npz`
- Stage-1 reward plot: `reports/stage1-repeat4-reward.png`
- Stage-1 success plot: `reports/stage1-repeat4-success.png`
- Medium obstacle comparison: `reports/agent-medium-lidar-seed17-comparison.json`
- Verified medium obstacle trajectory: `trajectories/stage3-medium-clearance-90112.npz`
- Verified medium obstacle GIF: `videos/stage3-medium-clearance-90112.gif`
- Full-course breakthrough report: `reports/stage3-full-breakthrough.json`
- Recommended full-course detailed evaluations: `reports/stage3-full-recommended-detailed-seed1000.json` and `reports/stage3-full-recommended-detailed-seed5000.json`
- Far-target successful demonstrations: `trajectories/distill-far-balanced-v1.npz`
- Recommended full-course trajectory: `trajectories/stage3-full-recommended-seed1000.npz`
- Recommended full-course GIF: `videos/stage3-full-recommended-seed1000.gif`
- Experiment observer desktop acceptance screenshot: `reports/lab-observer-home.png`
- Experiment observer mobile acceptance screenshot: `reports/lab-observer-mobile.png`
- Observer learning timelines: `trajectories/lab/*.npz`
- Live training console acceptance screenshot: `reports/live-training-console.png`
- High-frame-rate direct-stream screenshot: `reports/live-training-direct-stream.png`
- UI-controlled live runs and checkpoints: `lab/runs/*`

## Current limitations

- Stage 1 permits crawling, rolling, wall use, and non-human movement by design.
- Stage 2 is improved but not robust across all target positions.
- Full-course deterministic traversal is verified at 54/80 successes across two independent target sets, but is not yet 100% robust; failures concentrate in the farthest target interval.
- Stages 4-5 provide architecture, reward schedules, and verified transition machinery, but natural upright walking has not been demonstrated.
- PPO policy regression remains possible; best checkpoints and optional evaluation-based early stopping mitigate but do not eliminate it.

## Next technically justified work

1. Improve the recommended full-course policy on the remaining `x >= 10.1` target interval without losing its 100% strict route completion.
2. Add route-completion and target-bin metrics directly to checkpoint selection rather than selecting only by mean reward.
3. Repeat the successful-distillation/interpolation process with independently collected datasets to measure sensitivity to demonstration seeds.
4. Introduce obstacle dimensions continuously instead of switching between discrete YAML files.
5. Only after near-robust full-course traversal, anneal into upright and natural-gait rewards.

## Autonomous review round `autonomous-now-20260730-020156`

- Completed at: 2026-07-30T03:46:14
- Wall-clock duration: 104.3 minutes
- Execution mode: unattended multi-branch implementation/training/evaluation review

| Branch | Exit | Success | Mean reward | Final distance | Recommended checkpoint |
|---|---:|---:|---:|---:|---|
| autonomous-now-20260730-020156-medium-deterministic | 0 | 0.600 | 115.690 | 0.385 | `checkpoints\autonomous-now-20260730-020156-medium-deterministic\best\best_model.zip` |
| autonomous-now-20260730-020156-full-obstacles | 0 | 0.200 | 55.320 | 2.664 | `checkpoints\autonomous-now-20260730-020156-full-obstacles\final_model.zip` |
| autonomous-now-20260730-020156-upright-anneal | 0 | 1.000 | 231.792 | 0.338 | `checkpoints\autonomous-now-20260730-020156-upright-anneal\best\best_model.zip` |
| autonomous-now-20260730-020156-random-target-retention | 0 | 0.600 | 104.444 | 1.390 | `checkpoints\autonomous-now-20260730-020156-random-target-retention\best\best_model.zip` |
| autonomous-now-20260730-020156-continuation-01 | 0 | 0.000 | 24.668 | 2.208 | `checkpoints\autonomous-now-20260730-020156-continuation-01\best\best_model.zip` |
| autonomous-now-20260730-020156-continuation-02 | 0 | 0.000 | 24.668 | 2.208 | `checkpoints\autonomous-now-20260730-020156-continuation-02\best\best_model.zip` |

This entry is generated from real `summary.json` files. Failed branches are retained rather than hidden.
