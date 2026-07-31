# Development progress

Last updated: 2026-07-31

## Current verified status

The project is installable and runnable. The articulated PyMunk environment passes Gymnasium validation, PPO can train/save/load, headless and RGB rendering work, and deterministic stage-1 target reaching has been demonstrated. Stage-2 randomized-target training also improves generalization. The original full box-plus-platform course now has a verified deterministic majority-success policy: 54/80 successes across two independent 40-episode target sets, with 100% strict route completion. The native Tk desktop interface starts a separate PPO process from random weights, displays current Python/PyMunk body states and live metrics, and controls pause, resume, checkpoint save, and stop without a web server. Farthest-target robustness and natural upright walking remain unsolved.

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
- Native Tk desktop training console with from-scratch process creation, current PyMunk state rendering, rolling metrics and PPO losses, pause/resume/stop/save controls, and local run history.
- Atomic worker state under `lab/runs/`, including control, status, live frame, episode/update metrics, logs, TensorBoard data, and checkpoints.
- Ruff static analysis and automated tests.

## Final verification

```text
Ruff: all checks passed
Pytest: 34 passed
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

## Legacy WebUI history (removed 2026-07-30)

The repository previously contained a React/Vite experiment observer and a FastAPI/WebSocket live training console. Those versions established useful requirements—real PyMunk state rendering, reproducible run records, live metrics, and process controls—but accumulated browser lifecycle, polling, proxy, dependency, and layout failure modes.

After the native transport checkpoint `f33c8411f89b04ca83dc669c0e37b971b172de29` and native window checkpoint `dc62e432d4911c0bb46a5b716695045353d497d8` passed independent PPO smoke and control lifecycle tests, the WebUI source, server, web-only tests, manifest, dependencies, and screenshots were removed. Historical RL experiment metrics remain unchanged.

## Desktop application migration (2026-07-30)

### Goal 1: native trainer transport and process controller

Acceptance criterion: a non-web Python controller must launch a randomly initialized PPO worker, receive structured metadata/frame/status/metrics events through the worker's standard output, complete a real 64-step smoke run, and save a reloadable final checkpoint without FastAPI or a browser.

Implemented:

- `scripts/live_train_worker.py` supports `--stream-stdout` line-delimited events and an explicit `--run-dir`.
- `stickman_rl.desktop.DesktopTrainingController` launches, observes, pauses, resumes, saves, stops, and snapshots one trainer directly.
- Configuration validation prevents environment YAML files from being used as PPO training configs.
- The initial checkpoint retained the Web transport temporarily; it was removed after the native UI passed its independent checkpoint.

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

Checkpoint commit: `dc62e432d4911c0bb46a5b716695045353d497d8`.

### Goal 3: remove the WebUI stack

Acceptance criterion: the repository must contain no React/Vite frontend, FastAPI/Uvicorn server, browser WebSocket transport, web-only manifest/tests, or web runtime dependencies, while the native desktop smoke still completes a real PPO run.

Implemented:

- Removed `frontend/`, FastAPI server/API launchers, web-only asset exporter, observer manifest, web-only tests, and legacy screenshots.
- Removed FastAPI, Uvicorn, and WebSockets from runtime requirements.
- Simplified `live_train_worker.py` to structured stdout events plus low-frequency disk recovery snapshots.
- Added a distribution guard test that fails if WebUI files or dependencies return.
- README and project layout now expose the native desktop application as the sole training UI.

Verification:

```text
WebUI distribution guards: 4 passed
Desktop UI/controller regression tests: 6 passed
Post-removal native PPO smoke: completed, 64/64 steps
Post-removal live frames: 7
Post-removal rigid-body count: 10
Post-removal child process exit: 0
Full Ruff: passed
Full Pytest: 34 passed
```

Checkpoint commit: `ea0d2e3953d82effe559791d344667fc2fe2ebac`.

## Stability-only maintenance (2026-07-30)

### Goal: persist trainer diagnostics and harden cross-process JSON I/O

Acceptance criterion: the desktop controller must persist non-event worker stdout/stderr and controller lifecycle records to each run's `worker.log`; trainer spawn failures must become durable `failed` runs; transient Windows sharing locks while reading control/status JSON must not crash training or desktop controls.

Observed baseline:

- Recent desktop runs had empty `worker.log` files because non-structured output existed only in the live in-memory queue.
- `subprocess.Popen` failures escaped as `OSError`, while the run directory retained a stale `starting` status.
- Reusing a controller did not explicitly clear per-run event/stderr buffers before a new launch.
- The first guarded checkpoint run reproduced a worker crash while reading `control.json`: `PermissionError` killed the PPO process during stop.
- A five-round lifecycle stress test then reproduced the same sharing-lock race in the desktop controller while reading `status.json`, breaking resume.

Implemented:

- Persist controller launch, PID, process exit, non-event stdout, malformed event lines, and stderr with timestamps.
- Convert trainer spawn failures into `RuntimeError`, write `state=failed`, and preserve the original error in `worker.log` for the existing Tk error handler.
- Reset process/event/stderr/thread state before every new trainer launch.
- Add real subprocess regression tests for stdout/stderr persistence and a missing-interpreter startup failure.
- Retry transient `PermissionError`/decode failures in desktop status, metrics, frame, save, and control-state reads.
- Retry worker `control.json` reads and preserve the last valid paused/stop/save state if retries are exhausted, preventing accidental unpause.
- Add deterministic regression tests for both controller-side and worker-side sharing locks.

Verification:

```text
Desktop controller + worker reliability tests: 8 passed
Pause/save/resume/stop lifecycle stress: 5/5 repeated runs passed
Real native desktop smoke: completed, 64/64 steps
Smoke live frames: 5
Smoke rigid bodies: 10
Smoke child exit code: 0
worker.log: launch command, PID, and exit code 0 persisted
Full Ruff: passed
Full Pytest: 39 passed
```

Checkpoint commit: `9764b7482c8cc2f351ba087a53689337e810e789`.

### Goal: use one resilient JSON reader across the desktop UI

Acceptance criterion: refreshing or selecting desktop run history while a worker atomically updates `status.json`/`metrics.json` must not raise `PermissionError`, and the UI must use the same tested retry semantics as the controller instead of duplicate direct reads.

Implemented:

- Exposed one `read_json_file()` helper with bounded retries for transient sharing locks and JSON replacement windows.
- Replaced direct `Path.read_text()` calls in run-history summaries and history selection with the shared helper.
- Retained immediate default behavior for legitimately absent optional files, avoiding unnecessary polling delays.
- Added a deterministic history regression test that injects two `PermissionError` failures before a successful status read.

Verification:

```text
Desktop app + controller tests: 10 passed
Live concurrent history stress: 2,478 reads during PPO training
Stress worker exit code: 0
Stress final state: stopped
Full Ruff: passed
Full Pytest: 40 passed
```

Checkpoint commit: `51980d20cb9dfa2d78144705ac5b33efe47dcd85`.

### Goal: isolate reader/watcher threads between sequential runs

Acceptance criterion: when a second trainer starts immediately after the first process exits, late stdout/stderr, structured events, and process-exit records from the first trainer must remain attached to the first run and never enter the second run's log or UI event queue.

Observed baseline:

- Reader and watcher methods dereferenced mutable `self.process`, `self.run_dir`, `self._events`, and `self._stderr` after thread creation.
- A real two-run stress reproduction emitted 12,000 lines per worker and confirmed the first run ID appeared in the second run's `worker.log`.

Implemented:

- Bind each stdout reader, stderr reader, and watcher to the exact `Popen`, run directory, event queue, and stderr buffer created for that run.
- Make log writes accept an explicit run directory rather than following the controller's current mutable run.
- Track exit records per process object so `wait()` and watcher completion cannot duplicate or redirect an exit line.
- Add a sequential high-output regression test that immediately starts run two after run one exits.

Verification:

```text
Desktop controller tests: 7 passed
High-output reproduction: 12,003 log lines per run
First run ID in second log: false
Second run ID in first log: false
Exit records per run: exactly 1
Full Ruff: passed
Full Pytest: 41 passed
```

Checkpoint commit: `0b88b5190aae0f1ae441f3d746d87843821f57a5`.

### Goal: close cleanly while a terminal-status process is still exiting

Acceptance criterion: if a worker publishes `completed`, `stopped`, or `failed` before the Python process fully exits, closing the desktop app must not fail because a stop command is rejected; cleanup must continue through the existing bounded wait, terminate, and kill sequence.

Observed baseline:

- A fake worker published `state=completed` and then remained alive for two seconds.
- `stop_and_wait()` saw an active process, called `control("stop")`, and raised `RuntimeError: Run is already completed`, which could escape the Tk close callback.

Implemented:

- Treat a rejected stop command during process exit as a cleanup race rather than a fatal control error.
- Persist the skipped-stop reason in the run's `worker.log`.
- Continue waiting for natural exit and retain the existing terminate/kill fallback if the timeout expires.
- Add a regression test with terminal status visible while the child process is still alive.

Verification:

```text
Desktop controller tests: 8 passed
Terminal-status reproduction: process alive before close
stop_and_wait exit code: 0
Process alive after close: false
Skipped-stop reason persisted: true
Full Ruff: passed
Full Pytest: 42 passed
```

Checkpoint commit: `93c7fd4556ba2f078692d5edebac44f24384c590`.

### Goal: preserve real progress in worker failure status

Acceptance criterion: if the trainer fails after making progress, the final `failed` status must retain the last durable timestep, episode, rolling metrics, and requested total instead of resetting `num_timesteps` to zero; a missing or corrupt previous status may still fall back to zero.

Observed baseline:

- The top-level worker exception handler always wrote `num_timesteps: 0`.
- The earlier real control-file crash occurred after the lifecycle test had observed progress, yet its persisted failed status incorrectly reported zero steps.

Implemented:

- Build failure payloads by merging the last durable `status.json` and then overriding state, error, traceback, timestamp, and PID.
- Normalize the retained timestep to an integer and safely fall back to zero for absent, locked, malformed, or invalid status data.
- Add tests for preserving 384 steps plus episode/rolling metrics and for corrupt-status fallback.

Verification:

```text
Worker failure-status tests: 4 passed
Preserved num_timesteps: 384
Preserved episode and rolling reward: true
Corrupt status fallback: 0 steps
Full Ruff: passed
Full Pytest: 44 passed
```

Checkpoint commit: `bf6e83a62bb1146ea6bd9c7d1435eae5c32639bd`.

### Goal: close the environment when worker initialization fails

Acceptance criterion: once the vectorized Gym/PyMunk environment has been created, any later failure—including PPO construction, callback setup, training, or model saving—must close the environment before the worker exits.

Observed baseline:

- The existing `try/finally` began only immediately before `model.learn()`.
- Injecting a PPO constructor failure after `make_vec_env()` produced `env_closed=False`.

Implemented:

- Move policy kwargs, PPO construction, callback creation, training, final save, and final status inside the environment's `try/finally` scope.
- Keep environment creation outside the block so `close()` is called only for a successfully created environment.
- Add a regression test with a fake environment and injected PPO constructor failure.

Verification:

```text
Worker initialization tests: 5 passed
Injected PPO constructor failure: caught
Environment closed after failure: true
Full Ruff: passed
Full Pytest: 45 passed
```

Checkpoint commit: `f55aa25681f069f1b35595c8014acaaa3dd9c005`.

### Goal: finalize unexpected trainer process exits as failed runs

Acceptance criterion: if the child process exits before `live_train_worker.py` publishes `completed`, `stopped`, or `failed`, the controller must replace stale `starting`/`running` status with a durable failure containing the exit code and drained stderr tail, and emit that status to the current UI event queue.

Observed baseline:

- A fake worker printed `fatal import-style failure` to stderr and exited with code 7 before entering the worker's exception handler.
- The process was dead and stderr was captured, but `status.json` remained `state=starting`, leaving the UI logically active forever.

Implemented:

- Add one per-process exit finalizer owned by the controller.
- Wait briefly for stdout/stderr reader threads to drain before constructing failure diagnostics.
- Preserve existing terminal worker statuses; otherwise write `failed`, `process_exit_code`, stderr tail, error text, and timestamp.
- Push the synthesized failed status into the exact run's event queue.
- Keep finalization idempotent when both the watcher and explicit `wait()` observe exit.

Verification:

```text
Desktop controller tests: 9 passed
Hard-exit reproduction: exit code 7
Final state: failed
Recorded process exit code: 7
stderr tail preserved: fatal import-style failure
Full Ruff: passed
Full Pytest: 46 passed
```

Checkpoint commit: `16d7777628d4ef4620bcc6a14856f38d7c1eaf0e`.

### Goal: preserve failure progress through transient worker JSON locks

Acceptance criterion: a temporary sharing lock or atomic replacement window on `status.json` during exception handling must be retried so the failed run retains its last durable progress; worker control and failure-status reads must share one bounded retry implementation.

Observed baseline:

- `_control()` contained its own 20-attempt retry loop.
- `build_failure_status()` performed one direct `status.json` read and immediately fell back to zero on the first `PermissionError`, reintroducing progress loss exactly at failure time.

Implemented:

- Add one worker-side `read_json_with_retry()` helper for missing, locked, or temporarily malformed JSON files.
- Use the shared helper for both control commands and failure-status recovery.
- Preserve the last valid control state if all retries are exhausted.
- Add a regression test that injects two transient status-file locks and confirms 512 steps plus episode 4 survive.

Verification:

```text
Worker reliability tests: 6 passed
Transient status lock attempts before success: 2
Preserved failure progress: 512 steps
Preserved episode: 4
Full Ruff: passed
Full Pytest: 47 passed
```

Checkpoint commit: `e969d64612af7fc48ec432b7bd6731ef4cb69d47`.

### Goal: fail desktop run initialization transactionally

Acceptance criterion: a directory or initial JSON write failure before trainer spawn must be surfaced as the `RuntimeError` handled by the Tk UI, must not start a child process, and should persist a failed run plus diagnostic log whenever the run directory remains writable.

Observed baseline:

- Injecting `PermissionError` on the second initialization write escaped directly from `DesktopTrainingController.start()`.
- The partial run contained only `request.json`; no durable failed status or worker diagnostic existed.
- The Tk start callback catches `RuntimeError`, not arbitrary filesystem exceptions, so this error could escape the UI callback.

Implemented:

- Convert run-directory creation and initial request/control/status write failures into descriptive `RuntimeError` values.
- Persist a best-effort failed status and `worker.log` diagnostic without attempting to launch the trainer.
- Reuse the same guarded failure persistence for `subprocess.Popen` errors.
- Add a deterministic regression test that fails the second initialization JSON write.

Verification:

```text
Desktop controller tests: 10 passed
Injected failure before fix: PermissionError, partial request.json only
Injected failure after fix: RuntimeError
Child process started: false
Durable final state: failed
Durable files: request.json, status.json, worker.log
Full Ruff: passed
Full Pytest: 48 passed
```

Checkpoint commit: `889cb2871e3bef5b0ce37cb168c7fb9244b31b65`.

### Goal: keep trainer streams alive when worker.log is unavailable

Acceptance criterion: `worker.log` is diagnostic-only; an append failure must not block trainer spawn, kill stdout/stderr reader threads, or prevent process-exit finalization. Structured and plain output must continue through the in-memory event queue.

Observed baseline:

- Forcing every append open of `worker.log` to raise `PermissionError` caused `DesktopTrainingController.start()` to fail before `Popen`.
- The run remained `starting` and no trainer process was created.
- The same unguarded write path was used by stdout/stderr readers and the process watcher.

Implemented:

- Make `_append_worker_log()` best-effort and return a success flag instead of propagating file-system errors.
- Preserve all existing event-queue behavior even when durable diagnostic logging is unavailable.
- Add a fake-worker regression test that locks `worker.log` while emitting stdout, stderr, and a completed status.

Verification:

```text
Desktop controller tests: 11 passed
Locked-log baseline: PermissionError before process spawn
Locked-log real PPO smoke after fix: completed, 64/64 steps
Locked-log child exit code: 0
worker.log exists: false, as injected
Full Ruff: passed
Full Pytest: 49 passed
```

Checkpoint commit: `03a4a829827f754c31cdc92410794d62a416ad00`.

### Goal: isolate manual checkpoint failures from PPO training

Acceptance criterion: a user-requested manual checkpoint failure must produce a structured failed save result and mark that request handled without raising from the PPO callback or stopping the training loop. Final model persistence remains strict.

Observed baseline:

- Injecting `PermissionError` from `model.save()` escaped directly from `_manual_save()`.
- In `_handle_control()`, the exception would terminate `model.learn()` and fail the entire run.
- No `last_save.json` result existed for the UI or later diagnosis.

Implemented:

- Convert manual save exceptions into a `state=failed` result containing request ID, timestep, timestamp, exception type, and message.
- Persist the save result when possible and emit the existing checkpoint event regardless of success or failure.
- Mark the save request handled so a persistent control file does not retry the same failing save every environment step.
- Keep final checkpoint save failures fatal because a completed run without its final model is not valid.

Verification:

```text
Worker reliability tests: 7 passed
Injected manual save failure before fix: PermissionError escaped
Callback continues after fix: true
Callback state after failure: running
Handled request ID: save-request-1
Failed save progress retained: 320 steps
Full Ruff: passed
Full Pytest: 50 passed
```

Checkpoint commit: `9a99dd9964d5bf30d931cb227083875cb6742583`.

### Goal: bound desktop event backlog during UI stalls

Acceptance criterion: when the Tk main loop temporarily stops consuming events, high-frequency metadata/status/metrics/frame/checkpoint updates must not accumulate without bound. The UI must resume from the latest state and latest frame, while ordinary diagnostic events retain a bounded recent window.

Observed baseline:

- A fake worker emitted 20,000 structured frame events while the UI did not call `drain_events()`.
- All 20,000 frames remained queued in memory, from index 0 through 19,999.
- A modal dialog, window drag, system suspend, or slow render could therefore turn a 30 FPS stream into sustained memory growth.

Implemented:

- Replace the unbounded `queue.Queue` with a thread-safe `DesktopEventBuffer`.
- Coalesce metadata, status, metrics, frame, and checkpoint events to the latest value per type.
- Retain a bounded deque of 500 ordinary log/unknown events.
- Drain latest state first so the UI recovers immediately instead of replaying stale frames.
- Preserve per-run buffer isolation for sequential trainer processes.

Verification:

```text
Desktop controller tests: 12 passed
20,000-frame backlog before fix: 20,000 queued frames
20,000-frame backlog after fix: 1 queued frame
Retained frame index: 19,999
Full Ruff: passed
Full Pytest: 51 passed
```

Checkpoint commit: `90f18b127e827c4885b772471a2d6d80a61f88c8`.

### Goal: normalize persisted run timestamps before history sorting

Acceptance criterion: desktop history refresh must tolerate numeric timestamps, ISO strings, missing values, and malformed JSON values without raising during sorting. Invalid values should fall back to the run directory modification time.

Observed baseline:

- One run used an ISO string `updated_at`; another used an integer epoch timestamp.
- `sorted()` compared the raw values and raised `TypeError: '<' not supported between instances of 'int' and 'str'`.
- A single legacy or corrupted run could therefore prevent the desktop application from building its history list.

Implemented:

- Convert finite numeric timestamps directly to floating-point epoch values.
- Parse ISO timestamps, including `Z` UTC suffixes, through `datetime.fromisoformat()`.
- Fall back to the run directory mtime for unsupported, invalid, non-finite, or absent values.
- Keep the public run summary payload unchanged while sorting with a private numeric key.

Verification:

```text
Desktop app tests: 5 passed
Mixed numeric/string baseline: TypeError
Mixed numeric/string/invalid after fix: 3 summaries returned
Numeric year-2100 timestamp sorted first: true
Full Ruff: passed
Full Pytest: 52 passed
```

Checkpoint commit: `a100b7df7fa20273ee9098ee1abba27a73ad9fee`.

### Goal: reject non-object persisted run JSON safely

Acceptance criterion: valid JSON with the wrong top-level shape must not crash desktop history or history-detail loading. request/status/metrics payloads must be normalized to objects before any `.get()` access.

Observed baseline:

- A run with list-valued `request.json` and string-valued `status.json` parsed successfully.
- `run_summaries()` then called `.get()` on the string and raised `AttributeError`.
- One malformed legacy run could therefore prevent the desktop interface from opening its history panel.

Implemented:

- Add one shared `_json_object()` boundary that accepts dictionaries and downgrades all other JSON values to `{}`.
- Apply the same normalization in run summaries and selected-run status/metrics loading.
- Skip unusable history details without altering other valid runs.

Verification:

```text
Desktop app tests: 6 passed
Non-object baseline: AttributeError
Non-object after fix: 1 summary returned
Normalized request: {}
Normalized status: {}
Full Ruff: passed
Full Pytest: 53 passed
```

Checkpoint commit: `bc090b250cc9d7c597fa9e9a35ea9fd3aba9115b`.

### Goal: tolerate invalid percentage metrics in desktop refresh

Acceptance criterion: malformed persisted or streamed success-rate values must not raise from the Tk refresh loop. Invalid values should render as the same em-dash placeholder used for missing numeric metrics, while valid percentages remain unchanged.

Observed baseline:

- `format_percent("not-a-number")` raised `ValueError`.
- `format_percent({"bad": 1})` raised `TypeError`.
- `_refresh_ui()` calls this function every tick for `rolling_success_rate`, so one bad value could stop all subsequent desktop updates.

Implemented:

- Match `format_number()` error handling by catching conversion `TypeError` and `ValueError`.
- Return `—` for invalid percentage inputs without changing valid output formatting.
- Add regression assertions for both string and object inputs.

Verification:

```text
Desktop app tests: 6 passed
Invalid string before fix: ValueError
Invalid object before fix: TypeError
Invalid string/object after fix: —
Valid 0.625 after fix: 62.5%
Full Ruff: passed
Full Pytest: 53 passed
```

Checkpoint commit: `5db8520a0e77dfdbcd38728dfc6fd8a71ed9b6f7`.

### Goal: bound in-memory training metric history

Acceptance criterion: long-running training must retain only the same recent metric window that is persisted to `metrics.json`; episode and PPO-update histories must not grow without bound in memory.

Observed baseline:

- `episodes` and `updates` were ordinary lists.
- Appending 20,000 entries left all 20,000 resident in memory, even though `_write_metrics()` sliced only the latest 500 for disk output.
- Multi-hour or multi-day runs would therefore accumulate historical dictionaries indefinitely.

Implemented:

- Define one `METRIC_HISTORY_LIMIT = 500` constant.
- Store episode and update histories in `deque(maxlen=500)` containers.
- Persist the full bounded deques directly, keeping memory and disk retention semantics identical.
- Add a 20,000-entry pressure regression that verifies both ends of the retained window.

Verification:

```text
Worker reliability tests: 8 passed
20,000-entry baseline lengths: 20,000 episodes, 20,000 updates
After fix lengths: 500 episodes, 500 updates
Retained episode/update range: 19,500–19,999
Full Ruff: passed
Full Pytest: 54 passed
```

Checkpoint commit: `e76d87bb47a782754e8d71b650d8048fad3a5758`.

### Goal: tolerate invalid integer state values in desktop refresh

Acceptance criterion: malformed persisted or streamed timestep, episode, and waypoint integer values must not raise from the Tk refresh or physics-frame paths. Invalid and negative values should degrade to a non-negative default while valid integers remain unchanged.

Observed baseline:

- A minimal real `_refresh_ui()` call with object-valued `total_timesteps` raised `TypeError`.
- Direct `int()` conversions were also used for frame episode and active waypoint indices.
- One damaged status or frame payload could therefore stop subsequent desktop updates.

Implemented:

- Add one tested `nonnegative_int()` conversion boundary.
- Catch overflow, type, and value conversion errors and clamp negative results to zero.
- Apply the boundary to total/current timesteps, frame episode, and active waypoint index.
- Leave user form parsing strict so invalid launch input still produces a visible validation error.

Verification:

```text
Desktop app tests: 7 passed
Damaged status baseline: TypeError
Full _refresh_ui() with damaged timesteps after fix: completed
Valid integer/string conversion: 128 / 256
Invalid object, text, infinity, and negative values: 0
Full Ruff: passed
Full Pytest: 55 passed
```

Checkpoint commit: `6157962ae230458b1d6840d82e3a93660c129180`.

### Goal: sanitize malformed metric series before desktop charting

Acceptance criterion: persisted or streamed metric arrays containing non-object rows, non-numeric reward/loss values, or non-finite values must not raise from the Tk refresh loop. Valid metric records and numeric points should retain their order.

Observed baseline:

- A string entry in `episodes` caused `_refresh_ui()` to call `.get()` and raise `AttributeError`.
- Invalid reward or value-loss payloads would subsequently raise during Sparkline's direct `float()` conversion.
- One damaged metrics row could therefore stop all live charts and status updates.

Implemented:

- Add `metric_records()` to retain only dictionary entries from episode/update lists.
- Add `finite_float()` to accept convertible finite chart values and reject invalid, object, infinity, and NaN values.
- Make Sparkline use the shared numeric boundary.
- Keep valid record order and rolling-success calculation unchanged.

Verification:

```text
Desktop app tests: 8 passed
Damaged metrics baseline: AttributeError
Full _refresh_ui() with damaged metrics after fix: completed
Retained reward points: [7.5]
Retained value-loss points: [2.25]
Full Ruff: passed
Full Pytest: 56 passed
```

Checkpoint commit: `4cca3d91c04a0287072342345ef4b5cfc7bf18b0`.

### Goal: downgrade malformed structured events before desktop delivery

Acceptance criterion: JSON-valid worker output with a non-object top level, missing or invalid event type, or non-object payload must never terminate the reader thread or reach Tk state/frame handlers. Invalid events should become bounded diagnostic log entries while valid live state still coalesces normally.

Observed baseline:

- Passing a JSON list into `DesktopEventBuffer.put()` raised `AttributeError` because the buffer called `.get()` unconditionally.
- A checkpoint event with a string payload reached `_handle_event()` and raised the same error on `payload.get()`.
- One schema-invalid structured line could therefore kill the live stdout path or the Tk refresh callback despite being valid JSON.

Implemented:

- Add one event normalization boundary inside `DesktopEventBuffer`.
- Accept only non-empty string event types with object payloads.
- Convert malformed values into bounded controller diagnostic log events with a capped 500-character preview.
- Preserve existing latest-value coalescing for metadata, status, metrics, frame, and checkpoint events.

Verification:

```text
Desktop controller tests: 13 passed
Malformed list before fix: AttributeError
Malformed scalar payload before fix: AttributeError
Valid status after fix: running
Malformed events after fix: 2 diagnostic logs
Full Ruff: passed
Full Pytest: 57 passed
```

Checkpoint commit: `3c85f85bf5c7363d7db49832155ccd8bd59f3a70`.

### Goal: normalize persisted controller JSON objects

Acceptance criterion: JSON-valid but non-object request, status, control, metrics, frame, metadata, or last-save files must not raise from controller snapshot, process finalization, state waiting, or control paths. Wrong-shaped values should downgrade to safe defaults while active controls remain writable.

Observed baseline:

- A list-valued `frame.json` caused `snapshot()` to unpack a non-mapping and raise `TypeError`.
- A string-valued `status.json` caused `control()` to call `.get()` and raise `AttributeError`.
- A single legacy or corrupted run file could therefore break desktop refresh, control buttons, or exit finalization.

Implemented:

- Add one `read_json_object()` boundary on top of the existing lock/JSON retry reader.
- Use object-only reads for process exit status, controls, request/status/metrics snapshots, frame/metadata merge, and last-save data.
- Downgrade invalid optional objects to `None` and required objects to fresh default dictionaries.
- Preserve explicit inactive-run rejection and recover list-valued control files to the standard control object.

Verification:

```text
Desktop controller tests: 14 passed
Non-object snapshot baseline: TypeError
Non-object control baseline: AttributeError
Normalized request/status: {}
Normalized metrics: empty episode/update lists
Normalized frame/last-save: None
Recovered active control: paused=true
Full Ruff: passed
Full Pytest: 58 passed
```

Checkpoint commit: `0b83af4970b9857d7ad7f646897b6ebb155464a1`.

### Goal: finish close cleanup when control writes fail

Acceptance criterion: closing the desktop app must still execute the existing bounded wait, terminate, and kill sequence if writing the stop command raises an `OSError`. The control failure should be diagnostic, not an escape path that leaves the trainer alive.

Observed baseline:

- A real long-lived fake worker was started through `DesktopTrainingController`.
- Injecting a persistent `PermissionError` for `control.json` made `stop_and_wait()` raise immediately.
- The child process remained active because cleanup never reached its timeout and termination fallback.

Implemented:

- Catch both `RuntimeError` state races and `OSError` control-write failures around the stop request.
- Record a general cleanup diagnostic rather than assuming the process is already exiting.
- Continue through the unchanged bounded wait, terminate, and kill sequence.
- Add a real child-process regression test with a persistently locked control write.

Verification:

```text
Desktop controller tests: 15 passed
Locked control baseline: PermissionError escaped
Process active after baseline error: true
stop_and_wait after fix: exit code 1
Process active after cleanup: false
Control failure diagnostic persisted: true
Full Ruff: passed
Full Pytest: 59 passed
```

Checkpoint commit: `59c0deb70b3356e64c2fd533b3cc16093263b90b`.

### Goal: keep terminal failure visible when status persistence fails

Acceptance criterion: if an unexpected process exit cannot persist its failed status to `status.json`, the exit finalizer must not raise or lose the event. The current UI snapshot must continue exposing the terminal failure instead of being overwritten by the stale disk state.

Observed baseline:

- A temporary current run had durable `state=running` and 32 timesteps.
- Injecting a persistent `PermissionError` for terminal status writes made `_handle_process_exit()` raise.
- No event was queued, the process was already marked handled, and disk remained `running`, so the UI could stay logically active forever.

Implemented:

- Add a per-current-run in-memory terminal status fallback and clear it when a new run starts.
- Cache existing completed/stopped/failed states observed during process finalization.
- Treat terminal status write failures as diagnostic-only, attach the persistence error, and always publish the failed event.
- Make `snapshot()` prefer the in-memory terminal state for the matching current run, preventing stale disk state from overwriting UI failure.

Verification:

```text
Desktop controller tests: 16 passed
Terminal write baseline: PermissionError escaped
Baseline queued events: 0
Failed event after fix: 1
Event state: failed
Snapshot state: failed
Stale disk state retained for reproduction: running
Persistence diagnostic recorded: true
Full Ruff: passed
Full Pytest: 60 passed
```

Checkpoint commit: `1e99836b9bf6c5f305034fc8e0b1013ce1d52d95`.

### Goal: normalize nested live frame structures before rendering

Acceptance criterion: JSON-valid frame events with non-object `training`, non-object `frame`/`info`, or non-list action/body arrays must not raise from `PhysicsCanvas.set_frame()`, redraw, or the desktop action panel. Valid nested data must retain its order and values.

Observed baseline:

- Calling `PhysicsCanvas.set_frame()` with string-valued `training` raised `AttributeError` on `.get()`.
- List-valued `frame` would reach redraw and fail when mapping fields were accessed.
- Top-level event validation therefore did not protect the renderer from nested schema drift or damaged persisted frames.

Implemented:

- Add one tested `normalize_frame_payload()` boundary.
- Normalize `training`, `frame`, and nested `info` to objects.
- Normalize action, body-position, and body-angle values to lists.
- Apply the same normalizer in the desktop event handler and the `PhysicsCanvas` entry point so app state never retains the malformed envelope.
- Preserve all valid nested fields, including episode, target position, actions, body positions, angles, and success info.

Verification:

```text
Desktop app tests: 9 passed
String-valued training baseline: AttributeError
Normalized training after fix: {action: []}
Normalized frame after fix: empty info/body arrays
Valid nested values retained: true
Full Ruff: passed
Full Pytest: 61 passed
```

Checkpoint commit: `e819a6b9a790b5fd8e4673450a5472a53b0c990b`.

### Goal: sanitize live action values before desktop refresh

Acceptance criterion: object, non-numeric, NaN, or infinite action elements must not raise from the desktop action panel. Action indices must remain aligned with joint names, convertible values should be retained, and both live-event and disk-snapshot frame paths must use the same normalization.

Observed baseline:

- A minimal real `_refresh_ui()` call with an object-valued action raised `TypeError` during `float(value)`.
- The live event path normalized the frame envelope, but the disk snapshot fallback stored an unnormalized frame in `DesktopLabApp.self.frame`.
- One damaged action element could therefore stop all subsequent UI refreshes, especially after event-stream recovery from disk.

Implemented:

- Convert each action element to a finite float inside `normalize_frame_payload()`.
- Preserve list length and action-to-joint alignment by replacing invalid or non-finite values with `0.0`.
- Retain convertible numeric strings as floats.
- Normalize snapshot fallback frames before storing them in app state and sending them to `PhysicsCanvas`.

Verification:

```text
Desktop app tests: 9 passed
Object-valued action baseline: TypeError
Normalized actions after fix: [0.0, 0.5]
Full _refresh_ui() after fix: completed
Rendered zero action: true
Rendered 0.5 action: true
Full Ruff: passed
Full Pytest: 61 passed
```

Checkpoint commit: `bf916d97d4a611737b660b8b3fcd5c0115c6dbae`.

### Goal: sanitize body state elements before physics redraw

Acceptance criterion: malformed, missing, or non-finite body position and angle elements must not raise from `PhysicsCanvas.redraw()`. Array length and body-index alignment must remain stable, while valid numeric and convertible string values are retained.

Observed baseline:

- A real `PhysicsCanvas.redraw()` call with an object-valued body angle raised `TypeError` during `float(angle)`.
- Object, short, or non-finite body positions would subsequently break coordinate rotation or indexing.
- One damaged rigid-body sample could therefore stop the entire live physics canvas.

Implemented:

- Normalize every body position to exactly two finite floats.
- Replace invalid or missing x/y coordinates with `0.0` while preserving one output row per input row.
- Normalize every body angle to a finite float and replace invalid/non-finite values with `0.0`.
- Preserve valid numeric values and convertible numeric strings.

Verification:

```text
Desktop app tests: 9 passed
Object-valued angle baseline: TypeError
Normalized invalid position: [0.0, 0.0]
Normalized invalid angle: 0.0
Mixed valid/string/NaN/short inputs: verified
Real PhysicsCanvas.redraw after fix: completed
Full Ruff: passed
Full Pytest: 61 passed
```

Checkpoint commit: `61575e2a0ffaf97a04d76c48b34eb110fb562fa1`.

### Goal: fall back safely from malformed dynamic target coordinates

Acceptance criterion: malformed, short, or non-finite dynamic `target_position` values must not raise from physics redraw. Invalid dynamic coordinates should be removed so the renderer uses the known static target, while valid convertible coordinates remain available.

Observed baseline:

- A real `PhysicsCanvas.redraw()` call with object-valued `target_position` raised `KeyError: 0` during coordinate transformation.
- The body-position parser contained similar finite two-dimensional conversion logic, creating two fragile coordinate paths.

Implemented:

- Add one reusable `finite_point()` conversion boundary.
- Reuse it for body-position parsing, retaining partial body-coordinate recovery through explicit zero fill.
- Require both dynamic target coordinates to be finite; remove invalid values to activate the existing static-target fallback.
- Preserve valid numeric and convertible string target coordinates.

Verification:

```text
Desktop app tests: 10 passed
Object-valued dynamic target baseline: KeyError 0
Invalid dynamic target retained after fix: false
Static fallback target: [8.0, 1.0]
Real PhysicsCanvas.redraw after fix: completed
Valid string/numeric target conversion: verified
Full Ruff: passed
Full Pytest: 62 passed
```

Checkpoint commit: `46df2fac7045e1fadab4cbd6ee0465d5fce57167`.

### Goal: normalize run state names across controller and desktop UI

Acceptance criterion: non-string or blank `state` values must not raise during controller control/finalization/wait checks, desktop status rendering, history labels, or smoke completion checks. Invalid values should use an explicit context-appropriate fallback.

Observed baseline:

- Object-valued state made `DesktopTrainingController.control()` raise `TypeError: unhashable type: 'dict'` during active-state membership testing.
- The same value made `_refresh_ui()` raise during `STATUS_LABELS.get()`.
- One damaged state field could therefore break both controls and the 16 ms Tk refresh loop.

Implemented:

- Add one shared `state_name()` boundary that accepts trimmed non-empty strings only.
- Use `inactive` as the controller fallback and `idle` or `?` for desktop presentation contexts.
- Apply the boundary to process finalization, controls, state waiting, live refresh, history labels, and smoke terminal detection.
- Keep unknown but valid string states visible rather than silently remapping them.

Verification:

```text
Controller + desktop app tests: 28 passed
Object-valued state baseline: controller TypeError
Object-valued state baseline: UI TypeError
Controller after fix: RuntimeError, already inactive
UI after fix: waiting/idle state
Start button after fix: normal
Full refresh after fix: completed
Full Ruff: passed
Full Pytest: 64 passed
```

Checkpoint commit: `0006c662a393010bcf18cbaa7b9c4ab5a52b0518`.

### Goal: normalize metadata name arrays before desktop use

Acceptance criterion: non-string or blank action/body names must not raise from the action panel or physics renderer. Name-array lengths and indices must remain aligned with action/body state arrays, and invalid body geometry containers must downgrade safely.

Observed baseline:

- An object-valued action name made `_refresh_ui()` raise `TypeError: unhashable type: 'slice'` during name truncation.
- An object-valued body name made `PhysicsCanvas.redraw()` raise `TypeError: unhashable type: 'dict'` during `body_geometry.get(name)`.
- One damaged metadata name could therefore stop either the status UI or the physical visualization.

Implemented:

- Add one `normalize_metadata_payload()` boundary.
- Trim valid action/body names and replace invalid or blank entries with stable indexed placeholders such as `action_0` and `body_0`.
- Preserve input list lengths and indices so names remain aligned with live arrays.
- Normalize invalid `body_geometry` values to an empty object.
- Apply the normalizer to live metadata events, disk snapshot recovery, and `PhysicsCanvas.set_metadata()`.

Verification:

```text
Desktop app tests: 12 passed
Object action name baseline: TypeError
Object body name baseline: TypeError
Normalized action name: action_0
Normalized body name: body_0
Full action refresh after fix: completed
Full physics redraw after fix: completed
Full Ruff: passed
Full Pytest: 65 passed
```

Checkpoint commit: `aac3e7ecfbdf3f328067cffa1994d5877ff1df4a`.


### Goal: normalize room dimensions before physics-canvas transforms

Acceptance criterion: malformed, non-finite, zero, or non-object room metadata must not raise from `PhysicsCanvas._point_transform()`. Valid positive dimensions must be preserved, while invalid values fall back to the verified base environment size.

Observed baseline:

- `width: 0` raised `ZeroDivisionError`.
- `width: "bad"` raised `TypeError` during scale calculation.
- List-valued or null room metadata raised `TypeError` before the canvas could render.

Implemented:

- Add verified desktop defaults matching `configs/base.yaml`: width 12.0 and height 7.0.
- Add one positive finite float boundary for room dimensions.
- Normalize room width and height inside the existing metadata boundary before any canvas use.
- Preserve valid numeric strings and positive numeric values.

Verification:

```text
Desktop app tests: 13 passed
Zero/string/list/null room baseline: ZeroDivisionError or TypeError
All four invalid-room transforms after fix: completed
Normalized default room: 12.0 x 7.0
Full Ruff: passed
Full Pytest: 66 passed
```

Checkpoint commit: `6e08ba737d58477aa154d4857c96c1b63b739e56`.


### Goal: normalize target metadata before physics rendering

Acceptance criterion: missing, non-object, short, non-finite, or non-positive target position/size metadata must not raise from `PhysicsCanvas.redraw()`. Valid target fields must be preserved independently, while only invalid fields fall back to the verified base configuration.

Observed baseline:

- Missing or list-valued target metadata raised `TypeError`.
- A one-coordinate position raised `IndexError`.
- Non-numeric or null target sizes raised `TypeError`.

Implemented:

- Add verified target defaults matching `configs/base.yaml`: position `[9.5, 0.55]`, size `[0.8, 0.9]`.
- Require target positions to contain two finite coordinates.
- Require target width and height to be finite and strictly positive.
- Normalize position and size independently so a valid field is not discarded when the other field is malformed.
- Keep dynamic per-frame target positions unchanged; they continue to override normalized static metadata when valid.

Verification:

```text
Desktop app tests: 18 passed
Five malformed target cases before fix: TypeError or IndexError
Five malformed target redraws after fix: completed
Missing/list target fallback: full base target
Short position fallback: position only
Bad/null size fallback: size only
Full Ruff: passed
Full Pytest: 71 passed
```

Checkpoint commit: `e4a86b85cc265232cf0bd1da00acfedee774b02c`.


### Goal: normalize obstacle metadata before physics rendering

Acceptance criterion: non-list obstacle containers, non-object entries, short positions, non-finite coordinates, and non-positive sizes must not raise from `PhysicsCanvas.redraw()`. Valid obstacles must retain order and compatible custom fields.

Observed baseline:

- String-valued obstacle containers and null entries raised `AttributeError`.
- A one-coordinate box position raised `ValueError` during unpacking.
- A non-numeric size raised `TypeError` during rectangle arithmetic.

Implemented:

- Accept only list-valued obstacle containers and object-valued entries.
- Normalize obstacle type names to trimmed lowercase strings, falling back to `box` for invalid type values.
- Require box/platform/wall entries to have two finite position coordinates and strictly positive finite dimensions.
- Skip malformed drawable obstacles instead of terminating the renderer.
- Retain unsupported/custom obstacle objects for existing renderer skip behavior and preserve valid entry order.

Verification:

```text
Desktop app tests: 19 passed
Four malformed obstacle cases before fix: AttributeError, ValueError, or TypeError
Four malformed obstacle redraws after fix: completed
Malformed obstacle result: empty list
Valid wall geometry normalized: true
Custom slope entry retained: true
Full Ruff: passed
Full Pytest: 72 passed
```

Checkpoint commit: `315386009bbdc253592b4ec868ee6281dfb2b093`.


### Goal: normalize waypoint metadata before physics rendering

Acceptance criterion: non-list waypoint containers, null entries, short arrays, and non-finite coordinates must not raise from `PhysicsCanvas.redraw()`. Valid two-coordinate waypoints must retain order.

Observed baseline:

- String-valued waypoint containers and non-numeric coordinates raised `TypeError` inside the canvas transform.
- Null entries raised `TypeError`.
- A one-coordinate waypoint raised `IndexError`.

Implemented:

- Accept only list-valued waypoint containers.
- Convert each waypoint through the existing strict finite two-coordinate boundary.
- Skip malformed entries instead of terminating the renderer.
- Preserve valid waypoint order and numeric-string conversion.

Verification:

```text
Desktop app tests: 20 passed
Four malformed waypoint cases before fix: TypeError or IndexError
Four malformed waypoint redraws after fix: completed
Malformed waypoint result: empty list
Valid numeric-string waypoint preserved: [3.0, 4.0]
Full Ruff: passed
Full Pytest: 73 passed
```

Checkpoint commit: `bf0f01bebf304f231b057bedefb509a9402ad03c`.


### Goal: normalize body geometry before physics rendering

Acceptance criterion: malformed per-body circle, segment, polygon, or non-object geometry must not terminate `PhysicsCanvas.redraw()`. Invalid geometry for one body should be skipped while valid bodies remain renderable.

Observed baseline:

- Non-object geometry raised `TypeError`.
- A circle without radius raised `KeyError`.
- A segment with a short endpoint raised `ValueError`.
- A polygon with a non-numeric vertex raised `TypeError`.

Implemented:

- Add one `normalize_body_geometry()` boundary for the renderer's supported circle, segment, and polygon kinds.
- Require circles to have a positive finite radius and a finite two-coordinate offset.
- Require segments to have two finite endpoints and a non-negative finite radius.
- Require polygons to have at least three finite two-coordinate vertices.
- Drop unsupported or malformed single-body geometry without affecting other body state arrays.
- Convert valid numeric strings into finite floats.

Verification:

```text
Desktop app tests: 21 passed
Four malformed geometry cases before fix: TypeError, KeyError, or ValueError
Four malformed geometry redraws after fix: completed
Malformed geometry result: body entry omitted
Valid circle/segment/polygon normalization: passed
Full Ruff: passed
Full Pytest: 74 passed
```

Checkpoint commit: `b591885a812f53c8757299b9ae7ed36759974138`.


### Goal: tolerate non-UTF-8 persisted JSON in desktop and worker paths

Acceptance criterion: corrupt non-UTF-8 request/status/control/metrics/frame bytes must not raise into desktop history, snapshots, or the PPO control callback. Desktop readers should return caller defaults, while the worker should retain its last valid control state.

Observed baseline:

- `read_json_file()` raised `UnicodeDecodeError` on a corrupt `status.json`.
- `run_summaries()` propagated the same error and could not build desktop history.
- `LiveTrainingCallback._control()` raised `UnicodeDecodeError`, which could terminate PPO training.

Implemented:

- Treat `UnicodeDecodeError` as unreadable persisted state in the shared desktop JSON reader and return its supplied default.
- Treat `UnicodeDecodeError` as an exhausted/invalid worker JSON read and preserve the callback's last valid control state.
- Keep existing retry behavior for sharing locks and temporarily malformed UTF-8 JSON unchanged.

Verification:

```text
Controller/worker reliability tests: 27 passed
Shared reader corrupt bytes after fix: safe default returned
History with corrupt status after fix: summary returned with empty status
Worker corrupt control after fix: last valid control retained
Full Ruff: passed
Full Pytest: 76 passed
```

Checkpoint commit: `d1db32a4f5f7eb6b83636ac830cae73582e2451b`.


### Goal: keep PPO training alive when the desktop stdout pipe closes

Acceptance criterion: if the desktop reader exits or closes the worker's stdout pipe, structured event emission must not raise into PPO callbacks, model saving, or worker shutdown. After the first failure, high-frequency stdout streaming should disable itself while disk snapshots continue.

Observed baseline:

- Writing a status event to a real OS pipe whose read end was closed raised `OSError: [Errno 22] Invalid argument` on Windows.
- The exception escaped `emit_stdout_event()` and could terminate the training process from status, metrics, metadata, frame, checkpoint, or final-status paths.

Implemented:

- Make `emit_stdout_event()` return a boolean transport result and absorb closed/invalid stream `OSError` and `ValueError` failures.
- Update callback event paths to persist the returned state in `stream_stdout`.
- Disable subsequent stdout emissions after the first failure, avoiding repeated exceptions at frame rate.
- Keep disk status, metrics, frame, and checkpoint behavior unchanged.
- Keep final and failure event sends best-effort so they cannot invalidate model persistence or cleanup.

Verification:

```text
Worker reliability tests: 11 passed
Real closed OS pipe before fix: OSError [Errno 22]
Real closed OS pipe after fix: emit_result=False
Callback switches to disk-only after first failure: true
Full Ruff: passed
Full Pytest: 78 passed
```

Checkpoint commit: `adcd5335599fdca4e6496c60d80e8f9c3187a3ac`.


### Goal: tolerate oversized legal JSON numbers in desktop numeric boundaries

Acceptance criterion: arbitrarily large integers parsed from valid JSON must not raise `OverflowError` in action/coordinate normalization, metric formatting, or history timestamp sorting. Non-representable values should use the same safe fallbacks as other non-finite inputs.

Observed baseline:

- `finite_float(10**10000)` raised `OverflowError`.
- Number and percentage formatting raised the same error.
- A huge action value terminated frame normalization.
- A 4001-digit numeric `updated_at` terminated run history sorting.

Implemented:

- Catch `OverflowError` in the shared finite-float conversion boundary.
- Route number and percentage formatting through that finite boundary, also rejecting NaN and infinity consistently.
- Let frame actions and all existing coordinate/geometry consumers inherit the safe fallback behavior.
- Route numeric history timestamps through the same finite conversion and fall back to directory mtime when not representable.

Verification:

```text
Desktop app tests: 23 passed
Five oversized-number entry points before fix: OverflowError
finite_float after fix: None
Number/percentage display after fix: —
Oversized action after fix: 0.0
Oversized history timestamp after fix: summary returned
Full Ruff: passed
Full Pytest: 80 passed
```

Checkpoint commit: `22e74008d08ff007fe6cf569191c08878d7b0da9`.


### Goal: keep PPO training alive when frame recovery snapshots fail

Acceptance criterion: failure to write the optional low-frequency `frame.json` recovery snapshot must not raise from the PPO callback. The failure should be visible in status, disk writes should back off, and a later retry should restore normal snapshot persistence.

Observed baseline:

- Injecting `PermissionError` from `atomic_json_compact()` escaped `_write_frame()` directly.
- Because `_write_frame()` runs at training start, approximately 60 Hz during stepping, and training end, one locked frame file could terminate the entire PPO run despite live stdout frames still working.

Implemented:

- Keep stdout frame emission unchanged and make only the disk frame snapshot best-effort.
- Record the exception type/message in `frame_snapshot_error` for subsequent status payloads.
- Back off periodic disk frame writes for five seconds after failure.
- Emit one structured worker diagnostic when the write fails.
- Let forced start/end writes retry immediately.
- Clear the error and backoff state after the next successful snapshot write.
- Preserve strict failure behavior for status, metrics, manual/final checkpoints, and model persistence.

Verification:

```text
Worker reliability tests: 12 passed
Injected frame write before fix: PermissionError escaped
First failed write after fix: callback continued
Recorded error: PermissionError: simulated frame lock
Retry scheduled: true
Forced retry writes: 2 total attempts
Error after successful retry: cleared
Full Ruff: passed
Full Pytest: 81 passed
```

Checkpoint commit: `55e7bfda6385e8efa1d4edc7e9c881439b6d16bc`.


### Goal: keep PPO training alive when metrics recovery snapshots fail

Acceptance criterion: failure to write the optional `metrics.json` observability snapshot must not raise from episode, rollout, training-start, or training-end callbacks. Live metrics should still reach the desktop, the failure should be visible in status, and later writes should retry.

Observed baseline:

- Injecting `PermissionError` from `atomic_json()` escaped `_write_metrics()` directly.
- Disk persistence occurred before stdout emission, so the same failure both terminated PPO and suppressed the live metrics event.

Implemented:

- Emit live metrics before attempting the disk snapshot.
- Record disk exceptions in `metrics_snapshot_error` for status payloads.
- Back off ordinary disk retries for five seconds after failure.
- Emit a structured worker diagnostic for the failed snapshot.
- Force disk attempts at training start and training end.
- Clear error/backoff state after the next successful write.
- Preserve strict status and checkpoint/model persistence behavior.

Verification:

```text
Worker reliability tests: 13 passed
Injected metrics write before fix: PermissionError escaped
First failed write after fix: metrics event emitted before log
Recorded error: PermissionError: simulated metrics lock
Retry scheduled: true
Forced retry writes: 2 total attempts
Error after successful retry: cleared
Full Ruff: passed
Full Pytest: 82 passed
```

Checkpoint commit: `25e2f0ade419407f6173e8b1bb31ee84944848b7`.


### Goal: keep PPO training alive when metadata recovery snapshots fail

Acceptance criterion: failure to write the static `metadata.json` recovery snapshot during the first live frame must not terminate training or suppress live metadata/frame events. Metadata should be captured once, cached, exposed in status on failure, and retried without querying the environment every frame.

Observed baseline:

- Injecting `PermissionError` from the first `atomic_json(metadata.json)` call escaped `_write_frame()` during training startup.
- Disk persistence happened before metadata/frame stdout events, so one locked metadata file could terminate PPO before the desktop received geometry.
- Because file existence controlled metadata collection, a failed write would otherwise request the same static metadata at frame rate.

Implemented:

- Cache static metadata after the first environment snapshot.
- Emit live metadata before attempting disk persistence.
- Record disk exceptions in `metadata_snapshot_error` for status payloads.
- Back off ordinary metadata writes for five seconds after failure.
- Reuse the cached metadata for retry instead of re-querying the environment.
- Let forced start/end frame writes retry immediately.
- Clear error/backoff state after a successful write.

Verification:

```text
Worker reliability tests: 14 passed
Injected metadata write before fix: PermissionError escaped
First failed write after fix: metadata/frame/log events emitted
Environment metadata requests after first failure: 1
Forced retry environment metadata requests: still 1
Metadata file after successful retry: present
Error after retry: cleared
Full Ruff: passed
Full Pytest: 83 passed
```

Checkpoint commit: `d2562e2bbdb8c7cde7e0d8dd26a255de7227e43b`.


### Goal: keep PPO training alive when status snapshots fail

Acceptance criterion: a persistent `status.json` write lock must not terminate the PPO callback or suppress live state events. Critical start/stop/end transitions must force a retry, and a later successful write must clear the recorded persistence error.

Observed baseline:

- `atomic_json()` already retried short Windows sharing locks for about 100 ms.
- A persistent lock still made `_write_status()` raise `PermissionError` before emitting the stdout status event.
- The callback could therefore terminate even though live transport and the training process were otherwise healthy.

Implemented:

- Track `status_snapshot_error` and a one-second retry deadline in the callback.
- Keep status persistence best-effort while always emitting the current live status payload.
- Emit one bounded diagnostic log when a disk write fails.
- Force disk retries on training start, stop requests, and training end.
- Clear the error and retry deadline after a successful persistence attempt.

Verification:

```text
Worker tests: 15 passed
Persistent status lock baseline: PermissionError
Failure after fix: status event + log event, retry scheduled
Forced retry after fix: status.json created
Recovered status_snapshot_error: None
Full Ruff: passed
Full Pytest: 84 passed
```

Checkpoint commit: `57a0405359650a2c777d11d6c8e8c2f38465baf5`.


### Goal: tolerate malformed episode info without stopping PPO

Acceptance criterion: non-object `infos[0]`, non-object `info["episode"]`, and invalid episode reward/length/distance values must not raise from `_on_step()`. The callback must continue training and record finite fallback metrics.

Observed baseline:

- A completed episode with `info["episode"] = None` raised `AttributeError` at `episode_info.get(...)`.
- A non-object info row could also fail while converting `infos[0]` to a dictionary.
- One malformed wrapper payload could therefore terminate an otherwise valid PPO run at episode completion.

Implemented:

- Accept only list/tuple info containers and object-valued first entries.
- Accept only object-valued episode summaries.
- Add finite-float and nonnegative-integer conversion boundaries.
- Fall back to the callback's accumulated episode reward, current episode step, and distance `0.0`.
- Preserve normal Monitor output unchanged.

Verification:

```text
Worker tests: 17 passed
Malformed episode baseline: AttributeError
Callback continued after fix: true
Recorded reward: 1.0
Recorded length: 1
Recorded final distance: 0.0
Full Ruff: passed
Full Pytest: 86 passed
```

Checkpoint commit: `2ee746b4272b32d7d9fdb78456b6ae8404648b0b`.


### Goal: tolerate malformed rollout reward and done vectors

Acceptance criterion: non-numeric reward entries and non-sequence or invalid done containers must not raise from `_on_step()`. Invalid rewards must fall back to `0.0`, invalid done values to `False`, and no false completed episode may be recorded.

Observed baseline:

- A reward vector containing an object raised `TypeError` during NumPy float conversion.
- An object-valued dones container became a zero-dimensional array and raised `IndexError` when indexed.
- Either malformed wrapper output could terminate the PPO callback before control handling.

Implemented:

- Read only the first item from list, tuple, or ndarray vector containers.
- Reject non-sequence containers instead of coercing them through NumPy.
- Convert reward values through the finite-float boundary.
- Convert done values only from finite numeric or boolean inputs.
- Fall back to reward `0.0` and done `False` for invalid values.

Verification:

```text
Worker tests: 19 passed
Object reward baseline: TypeError
Object dones baseline: IndexError
Bad reward after fix: callback continued, reward 0.0
Bad dones after fix: callback continued, no episode recorded
Full Ruff: passed
Full Pytest: 88 passed
```

Checkpoint commit: `4fdb7eca30835c3c7564c8381d6de4c354fdf92b`.


### Goal: ignore malformed non-scalar logger metrics

Acceptance criterion: multi-element arrays, objects, and non-finite values in the SB3 logger must not raise from `_logger_metrics()` or terminate rollout-end processing. Valid scalar and one-element numeric array values must remain available.

Observed baseline:

- A two-element NumPy logger value raised `ValueError` because its finite mask was used as a boolean.
- An object-valued logger entry raised `TypeError` inside `np.isfinite`.
- One custom or damaged logger value could therefore terminate the callback at rollout end.

Implemented:

- Add one optional finite scalar conversion boundary.
- Accept one-element ndarrays and normal numeric scalar values.
- Reject multi-element arrays, objects, and non-finite values as `None`.
- Apply the same conversion to every recorded PPO logger metric.

Verification:

```text
Worker tests: 20 passed
Array logger baseline: ValueError
Object logger baseline: TypeError
One-element array after fix: 1.25
Multi-element/object/infinite after fix: None
Full Ruff: passed
Full Pytest: 89 passed
```

Checkpoint commit: `969d15f88b337644b1bd873eb8f9d784e3e97d93`.


### Goal: sanitize NumPy non-finite values recursively before JSON encoding

Acceptance criterion: NumPy scalar and array `NaN/Infinity` values must not bypass `json_safe()` or produce non-standard JSON tokens. Nested converted values must use the same finite-value cleanup as native Python floats.

Observed baseline:

- `json_safe(np.float32(np.inf))` returned Python `inf`.
- `json_safe(np.array([1.0, np.nan, np.inf]))` returned `[1.0, nan, inf]`.
- Strict JSON encoding with `allow_nan=False` raised `ValueError` for both payloads.

Implemented:

- Recursively pass NumPy scalar `.item()` results back through `json_safe()`.
- Recursively pass NumPy array `.tolist()` results back through `json_safe()`.
- Preserve finite numeric values while converting all nested non-finite values to `None`.

Verification:

```text
Worker tests: 21 passed
Scalar infinity baseline strict JSON: ValueError
Array NaN/Infinity baseline strict JSON: ValueError
Scalar after fix: null
Array after fix: [1.0, null, null]
Full Ruff: passed
Full Pytest: 90 passed
```

Checkpoint commit: `b6d54cc85b097a76cb0d3e0b8fd2cb74f63a8612`.


### Goal: make json_safe total for unsupported Python objects

Acceptance criterion: `Path`, bytes, and arbitrary third-party objects in environment info or logger payloads must not escape `json_safe()` and cause `json.dumps` to raise. The fallback representation must remain deterministic enough for diagnostics.

Observed baseline:

- `Path` values raised `TypeError: Object of type WindowsPath is not JSON serializable`.
- bytes values raised the same serialization error.
- A custom info object also raised `TypeError` and could terminate status/frame emission.

Implemented:

- Convert `Path` values to path strings.
- Decode bytes as UTF-8 with replacement for invalid byte sequences.
- Convert all other unsupported objects to stable `<module.qualname>` type markers.
- Preserve existing handling for mappings, sequences, native scalars, and NumPy values.

Verification:

```text
Worker tests: 22 passed
Path/bytes/custom baseline: TypeError
Strict JSON after fix: passed
Path result: artifact.bin
Bytes result: abc\ufffd
Custom result: <__main__.CustomInfo>
Full Ruff: passed
Full Pytest: 91 passed
```

Checkpoint commit: `5c4caf9886b563ea8b9070d724c54ef8ca33436f`.


### Goal: retry transient atomic temporary-file write locks

Acceptance criterion: a transient Windows sharing lock on `*.json.tmp` must not fail controller or worker atomic JSON writes before the existing replace retry can run. Both phases must retain the same bounded retry limit.

Observed baseline:

- Worker `atomic_json()` raised `PermissionError` on the first locked `.tmp.write_text()` attempt.
- Controller `_atomic_json()` failed identically.
- Only the later temporary-file rename had retry protection, leaving the first atomic-write phase brittle.

Implemented:

- Add bounded temporary text-write retry helpers to worker and controller.
- Use the same 20 attempts and 5 ms delay as the existing replace retry.
- Apply worker retry to both formatted and compact JSON snapshots.
- Preserve existing JSON formatting and final rename behavior.

Verification:

```text
Affected tests: 42 passed
Worker baseline attempts before failure: 1
Controller baseline attempts before failure: 1
Worker after fix: transient lock recovered, JSON persisted
Controller after fix: transient lock recovered, JSON persisted
Full Ruff: passed
Full Pytest: 93 passed
```

Checkpoint commit: `3825fae804d0ce6b0db8b9a1a9c9c3d96313caeb`.


### Goal: stop recursive info containers from overflowing json_safe

Acceptance criterion: self-referential dict/list payloads must not raise `RecursionError` during status, frame, or event serialization. Repeated references that are not cyclic must still serialize normally.

Observed baseline:

- A dict containing itself raised `RecursionError` in `json_safe()`.
- A list containing itself failed identically.
- A third-party wrapper could therefore terminate training by placing a cycle inside environment info.

Implemented:

- Track container identities only along the current recursion path.
- Replace a detected cycle with the stable marker `<recursive-reference>`.
- Remove identities after each branch so repeated non-cyclic shared objects remain fully serialized.
- Apply the same handling to dict, list, and tuple containers.

Verification:

```text
Worker tests: 24 passed
Recursive dict baseline: RecursionError
Recursive list baseline: RecursionError
Recursive dict/list after fix: <recursive-reference>
Repeated non-cyclic object after fix: serialized twice
Strict JSON after fix: passed
Full Ruff: passed
Full Pytest: 94 passed
```

Checkpoint commit: `84eea3208907fa7a88367dc914abd67b8bd8a169`.


### Goal: parse persisted control booleans without Python truthiness

Acceptance criterion: string-valued `paused/stop` fields such as `"false"` and `"0"` must not become true merely because the strings are non-empty. Recognized values must parse predictably, and malformed values must not flip the last valid control state.

Observed baseline:

- A control file containing `{"paused": "false", "stop": "false"}` produced `paused=True, stop=True`.
- The worker could therefore pause or terminate from a manually edited, migrated, or damaged control file that visually requested false.

Implemented:

- Recognize common true strings: `true`, `1`, `yes`, `on`.
- Recognize common false strings: `false`, `0`, `no`, `off`.
- Continue accepting booleans and finite numeric values.
- Preserve the previous valid control state for unknown strings or object values.
- Use the strict parser for both paused and stop fields.

Verification:

```text
Worker tests: 25 passed
String false baseline: paused=True, stop=True
String false after fix: paused=False, stop=False
String yes/on after fix: paused=True, stop=True
Unknown values after fix: last valid state retained
Full Ruff: passed
Full Pytest: 95 passed
```

Checkpoint commit: `0ea58f9f09550e61927e64a6a8692594cbfc50af`.


### Goal: parse episode success flags without Python truthiness

Acceptance criterion: string or malformed `is_success` values must not be classified through Python truthiness. Recognized true/false representations must update success metrics correctly, and unknown values must default to failure.

Observed baseline:

- An episode with `is_success="false"` was recorded as `success=True`.
- Rolling success rate and any downstream checkpoint selection using it could therefore be corrupted by a wrapper or migrated payload.

Implemented:

- Reuse the strict finite boolean parser for `is_success`.
- Recognize standard true/false strings and finite numeric booleans.
- Default unknown strings and object values to `False`.
- Preserve normal boolean environment output unchanged.

Verification:

```text
Worker tests: 29 passed
String false baseline: success=True
String false after fix: success=False
String true after fix: success=True
Unknown/object values after fix: success=False
Full Ruff: passed
Full Pytest: 99 passed
```

Checkpoint commit: `411029ee8a353cb927a7740a1b09186dd9352c86`.


### Goal: clear stale last_info when the current step has no info

Acceptance criterion: `last_info` must describe the current rollout step. An empty or invalid current info payload must clear previous success, distance, and diagnostic fields instead of leaving stale values visible in status and the desktop UI.

Observed baseline:

- The callback updated `last_info` only when the current info dictionary was non-empty.
- After a previous `is_success=True, final_distance=0.1`, a later empty info step retained those values.
- The desktop could therefore display a previous episode's result as if it belonged to the current step.

Implemented:

- Replace `last_info` on every step with the normalized current info object.
- Empty or malformed info now produces `{}` and clears stale fields.
- Preserve normal non-empty environment info unchanged.

Verification:

```text
Worker tests: 30 passed
Stale info baseline: {'is_success': True, 'final_distance': 0.1}
Empty info after fix: {}
Full Ruff: passed
Full Pytest: 100 passed
```

Checkpoint commit: `a45ebf25bdc7a8cf4969de33d03986ed430e8b1f`.


### Goal: reject non-integer TrainingRequest fields before creating a run

Acceptance criterion: `stage`, `timesteps`, and `seed` must be genuine integer values. Floats and booleans must fail before the controller creates a run directory, while valid `numbers.Integral` inputs must normalize to built-in `int` values before persistence.

Observed baseline:

- `stage=1.5`, `timesteps=64.5`, and `seed=0.5` passed range validation.
- `stage=True` also passed because Python booleans participate in integer comparisons.
- Invalid values could therefore create a run directory and fail only when the worker parsed command-line arguments.

Implemented:

- Require each numeric request field to implement `numbers.Integral` and explicitly reject `bool`.
- Normalize accepted integral values to built-in `int` before range checks and request persistence.
- Preserve the existing stage, timestep, seed, and config constraints.

Verification:

```text
Controller tests: 26 passed
Rejected before run creation: stage=1.5, stage=True, timesteps=64.5,
  timesteps=False, seed=0.5, seed=True
Run directories after every rejected request: 0
numpy.int64 normalization: types int/int/int, values 2/128/7
Full Ruff: passed
Full Pytest: 107 passed
```

Checkpoint commit: `1cf9ed14ed352cdabe4fa2b92558e40460b96f2d`.


### Goal: reject non-string config values before creating a run

Acceptance criterion: `train_config` and `env_config` must be `str` or `None`. Other runtime values must produce a clear validation error before the controller creates a run directory or starts a worker.

Observed baseline:

- `train_config=True` and `train_config=Path(...)` raised a raw `AttributeError` from `.strip()`.
- `env_config=1` and `env_config=[]` failed through the same accidental exception path.
- The type annotation alone did not protect programmatic callers.

Implemented:

- Return `None` for absent or blank string config values as before.
- Reject every non-string, non-`None` config value with `ValueError: Config path must be a string`.
- Preserve path containment, file extension, existence, and training-config naming checks.

Verification:

```text
Controller tests: 30 passed
Rejected as ValueError: train_config bool/Path, env_config int/list
Run directories after every rejected request: 0
Full Ruff: passed
Full Pytest: 111 passed
```

Checkpoint commit: `9b7a05e6c52396152631464e4ccfb81a05bafdfd`.


### Goal: isolate stdout telemetry payload sanitization failures

Acceptance criterion: stdout telemetry is auxiliary. Exceptions raised while sanitizing, encoding, or printing a structured event must disable streaming and return control to PPO instead of terminating training.

Observed baseline:

- A dictionary key whose `__str__` raised escaped from `json_safe()` as `RuntimeError`.
- A 2,500-level nested container escaped as `RecursionError`.
- Both failures happened before the existing stdout write exception handler.

Implemented:

- Move payload sanitization and JSON encoding inside the telemetry isolation boundary.
- Catch ordinary exceptions across sanitization, encoding, and output, returning `False`.
- Preserve `BaseException` behavior so process termination and interrupts are not swallowed.
- Existing callback callers continue switching to disk-only after the first `False` result.

Verification:

```text
Worker tests: 31 passed
Bad dictionary key after fix: False
2,500-level container after fix: False
Full Ruff: passed
Full Pytest: 112 passed
```

Checkpoint commit: `f34f57dc851c59601eaebf9d6283f5974b20382c`.


### Goal: make status serialization resilient to unprintable keys and excessive nesting

Acceptance criterion: malformed third-party info must not terminate status persistence. Dictionary keys that cannot be stringified need stable placeholders, and containers deeper than the supported serialization depth need deterministic truncation while ordinary, shared, and cyclic structures retain their existing behavior.

Observed baseline:

- A custom info key whose `__str__` raised caused `_status_payload()` to fail with `RuntimeError`.
- A 2,500-level info container caused `_status_payload()` to fail with `RecursionError`.
- Both failures occurred before status JSON could be written or streamed.

Implemented:

- Add a 128-level container depth limit with the marker `<max-depth>`.
- Convert unprintable mapping keys to stable type-and-index placeholders.
- Preserve recursive-reference detection and repeated non-cyclic object expansion.
- Keep all resulting status payloads compatible with strict JSON encoding.

Verification:

```text
Worker tests: 32 passed
Unprintable-key status payload: strict JSON passed
2,500-level status payload: strict JSON passed
Full Ruff: passed
Full Pytest: 113 passed
```

Checkpoint commit: `795931c350587961700de2a3a1809ce901b6ef3b`.


### Goal: parse desktop success flags without Python truthiness

Acceptance criterion: persisted success flags must be interpreted explicitly. String values such as `"false"` must not count as successful episodes or render a `GOAL REACHED` overlay, while recognized true values continue to work and unknown values are ignored.

Observed baseline:

- Episode metrics `["false", "true", 0, 1]` produced the incorrect rolling series `[1.0, 1.0, 0.667, 0.75]`.
- A frame with `is_success="false"` still rendered `GOAL REACHED`.
- Both paths relied on Python truthiness rather than persisted boolean semantics.

Implemented:

- Add a shared desktop `boolean_value()` parser for booleans, recognized strings, and finite numeric values.
- Skip unknown success values instead of treating them as true.
- Reuse the same boundary in the rolling success chart and physics success overlay.

Verification:

```text
Desktop app tests: 24 passed
Rolling success after fix: [0.0, 0.5, 0.3333333333333333, 0.5]
String false overlay after fix: absent
String true overlay after fix: present
Full Ruff: passed
Full Pytest: 114 passed
```

Checkpoint commit: `7d458905dc0012c351ada240a639ad5edc08bd13`.


### Goal: keep live frame serialization alive with malformed action values

Acceptance criterion: the action vector is observational metadata. Invalid callback values must not terminate PPO while valid scalar, vector, and batched actions remain finite numeric lists.

Observed baseline:

- A dictionary action caused `_write_frame()` to raise `TypeError`.
- A ragged nested action caused NumPy conversion to raise `ValueError`.
- Both failures occurred in the live observation path after the environment step had already completed.

Implemented:

- Add a dedicated `action_vector()` boundary.
- Accept scalar, one-dimensional, and batched actions, selecting the first environment row.
- Convert non-finite elements to `0.0`.
- Degrade non-numeric and ragged values to an empty action list.

Verification:

```text
Worker tests: 34 passed
Dictionary action after fix: []
Scalar action after fix: [0.5]
Ragged action after fix: []
Full Ruff: passed
Full Pytest: 116 passed
```

Checkpoint commit: `92b1c9c0cc61e5cd7de3cf67fd54ad85c9a8b379`.


### Goal: isolate live snapshot capture failures from PPO training

Acceptance criterion: live physics snapshots are observational. Empty, malformed, missing-metadata, or raising snapshot providers must not terminate PPO; they must expose a diagnostic error, back off, and recover on a later successful capture.

Observed baseline:

- An empty result raised `IndexError`.
- A `None` payload raised `AttributeError`.
- A frame without required metadata raised `KeyError`.
- A snapshot provider exception escaped directly as `RuntimeError`.

Implemented:

- Validate the environment result container, first payload object, frame object, and initial metadata.
- Record `frame_capture_error` in status and retry after one second.
- Emit a worker diagnostic log and return without creating a fake frame.
- Allow forced start/end calls to retry immediately; clear the error after recovery.

Verification:

```text
Worker tests: 36 passed
Empty/None/missing-metadata/raising providers: all isolated
Failed captures: no frame.json created
Forced retry recovery: metadata.json and frame.json created, error cleared
Full Ruff: passed
Full Pytest: 118 passed
```

Checkpoint commit: `417a760012bcfc71b1a6f56156ddc2ef0f75f4ee`.


### Goal: validate PPO integer configuration before environment creation

Acceptance criterion: integer PPO settings must reject booleans, fractions, missing fields, non-positive values, and malformed policy layers with clear errors before allocating the environment or constructing PPO.

Observed baseline:

- YAML `n_steps: true` was converted by `int(True)` into `1`.
- The real PPO constructor then failed with the low-level assertion ``n_steps * n_envs` must be greater than 1`.
- The environment had already been created before that error surfaced.

Implemented:

- Validate `n_steps`, `batch_size`, `n_epochs`, and `checkpoint_freq` as genuine integers with explicit minimums.
- Validate `policy_layers` as a non-empty list of positive integers.
- Normalize valid `numbers.Integral` values to built-in `int`.
- Run integer validation before `make_vec_env()` and reuse normalized values directly.

Verification:

```text
Worker tests: 47 passed
n_steps: true after fix: ValueError n_steps must be an integer
Environment created for invalid config: False
Integral normalization and bool/fraction/missing/non-positive cases: passed
Full Ruff: passed
Full Pytest: 129 passed
```

Checkpoint commit: `b7f37b00735becd5502a9e12cb233d4b544e794e`.


### Goal: validate PPO floating-point configuration before environment creation

Acceptance criterion: PPO float settings must reject booleans, missing values, NaN/Infinity, and unsupported ranges with clear errors before allocating the environment or constructing the optimizer.

Observed baseline:

- YAML `learning_rate: .nan` reached the real PPO constructor.
- PPO failed with `ValueError: Invalid learning rate: nan` only after the environment had been created.
- Boolean and out-of-range values were also silently accepted by direct `float(...)` conversion.

Implemented:

- Validate learning rate, gamma, GAE lambda, clip range, entropy/value coefficients, max gradient norm, initial log standard deviation, and optional target KL.
- Reject booleans and require finite real values.
- Enforce positive or probability-range constraints appropriate to each field.
- Normalize valid real values to built-in `float` and validate before `make_vec_env()`.

Verification:

```text
Worker tests: 63 passed
learning_rate: .nan after fix: ValueError learning_rate must be finite
Environment created for invalid float config: False
Boolean/non-finite/missing/out-of-range cases: passed
Full Ruff: passed
Full Pytest: 145 passed
```

Checkpoint commit: `73317979d18ab8070bdb6ad3f776e9c91d0798cf`.


### Goal: reject unsupported live worker mode configuration

Acceptance criterion: the desktop live worker is intentionally PPO with one environment. Configurations declaring another algorithm or environment count must fail clearly before environment allocation instead of being silently ignored.

Observed baseline:

- A config declaring `algorithm: SAC` still constructed PPO.
- A config declaring `n_envs: 8` still called `make_vec_env(..., n_envs=1)`.
- Persisted configuration could therefore disagree with the training process that actually ran.

Implemented:

- Accept algorithm values only when they normalize case-insensitively to `PPO`.
- Require `n_envs` to be a genuine integer exactly equal to `1`.
- Reject booleans, fractional counts, unsupported algorithms, and multi-environment values.
- Normalize supported values to `PPO` and `1` before all other config validation and before `make_vec_env()`.

Verification:

```text
Worker tests: 70 passed
SAC / 8-env config after fix: ValueError algorithm must be PPO for live training
Environment created for unsupported mode: False
Supported normalization: {'algorithm': 'PPO', 'n_envs': 1}
Full Ruff: passed
Full Pytest: 152 passed
```

Checkpoint commit: `554399a11702d4e80570b2619b57efe6a19ece44`.


### Goal: normalize invalid failure timestep progress

Acceptance criterion: persisted `num_timesteps` values used while finalizing a failed worker must not invent progress or raise a second exception. Genuine non-negative integer-like values should remain available, while booleans, negative values, and non-finite values must fall back to zero.

Observed baseline:

- `num_timesteps: true` was converted by `int(True)` into one completed step.
- Negative persisted values remained negative in the final failed status.
- A non-finite value raised `OverflowError` inside the top-level failure handler, potentially masking the original trainer exception.

Implemented:

- Reject booleans in the shared worker `nonnegative_int()` boundary.
- Reuse that boundary when building final failure status instead of performing a separate direct `int()` conversion.
- Preserve valid durable progress while clamping negative values and safely defaulting conversion failures to zero.
- Add focused regressions for boolean, negative, and non-finite persisted progress.

Verification:

```text
TrainingRequest integer-boundary recheck: 7 passed
Worker failure-status regressions: 5 passed
Invalid request run directories after rejection: 0
Sensitive information scan: no findings
Full Ruff: passed
Full Pytest: 155 passed
```

Checkpoint commit: `22eddefd9920fe1a909806250f040c5c3b71345b`.


### Goal: preserve the original worker failure when terminal status persistence fails

Acceptance criterion: a persistent `status.json` write failure during top-level worker exception handling must not replace the original trainer exception or suppress the live failed-status event. The event should expose the persistence diagnostic while the original exception continues to determine process failure.

Observed baseline:

- An injected trainer `RuntimeError` entered the top-level failure handler correctly.
- A subsequent persistent `PermissionError` from `atomic_json(status.json)` escaped instead.
- The original trainer exception was masked and no failed-status stdout event was emitted.

Implemented:

- Make terminal failed-status persistence best-effort inside the top-level exception handler.
- Attach `status_persist_error` to the in-memory failure payload when the write raises `OSError`.
- Emit the failed-status event after the persistence attempt regardless of disk success.
- Preserve the original exception with the existing bare re-raise.
- Add an end-to-end `main()` regression using a locked status write and captured stdout event.

Verification:

```text
Reproduction before fix: PermissionError masked RuntimeError
Focused failure-finalization regression: 1 passed
Sensitive information scan: no findings
Full Ruff: passed
Full Pytest: 156 passed
```

Checkpoint commit: `59618e9e1b5c905d2a4f2210035fe61d6b8948df`.


### Goal: preserve accumulated episode length for invalid negative summaries

Acceptance criterion: a malformed negative episode length from wrapper metadata must not be recorded as a zero-length episode. The callback should use its already accumulated current episode step as the fallback, while valid non-negative episode lengths remain unchanged.

Observed baseline:

- A completed episode had five callback steps already accumulated.
- Injected Monitor metadata reported `l=-7`.
- The shared `nonnegative_int(value, default)` helper returned zero instead of its supplied fallback, so the persisted episode length became impossible.

Implemented:

- Normalize the helper fallback itself to a non-negative value.
- Return that fallback for booleans, conversion failures, and parsed negative integers.
- Preserve any successfully parsed non-negative integer without treating the fallback as a minimum.
- Add a callback regression proving a negative metadata length falls back to the five accumulated steps.

Verification:

```text
Reproduction before fix: recorded episode length 0 instead of 5
Focused fallback regressions: 4 passed
Sensitive information scan: no findings
Full Ruff: passed
Full Pytest: 157 passed
```

Checkpoint commit: `f790131e5351e8a16f600fd32bd844952d6358dd`.


### Goal: reject boolean values at worker floating-point metric boundaries

Acceptance criterion: boolean values in numeric reward, distance, or logger fields must not be interpreted through Python's `bool`-as-`int` behavior. Invalid booleans should use the caller's numeric fallback, while explicit boolean fields continue through the dedicated strict parser.

Observed baseline:

- A rollout reward vector containing `True` passed `float(True)` conversion.
- The callback added `1.0` to the current episode reward even though the reward payload was malformed.
- This could silently corrupt rolling reward and persisted episode metrics without stopping training.

Implemented:

- Reject both native `bool` and NumPy `bool_` values in the shared `finite_float()` boundary.
- Return the caller-provided numeric fallback before attempting float conversion.
- Preserve the dedicated `finite_bool()` path for success, done, paused, and stop values.
- Extend malformed reward-vector regressions with a boolean reward case.

Verification:

```text
Reproduction before fix: True reward accumulated as 1.0
Focused numeric/boolean regressions: 7 passed
Sensitive information scan: no findings
Full Ruff: passed
Full Pytest: 158 passed
```

Checkpoint commit: `27d829081d5bf3c1ddf48ec00b9b400922a7fecf`.

### Goal: isolate concurrent controller and worker atomic JSON temporary files

Acceptance criterion: controller and worker writes targeting the same run JSON file must not share one fixed temporary path. Concurrent atomic writes should both complete without one writer deleting the other writer's pending temporary file, while the destination remains valid complete JSON.

Observed baseline:

- Both `DesktopTrainingController._atomic_json()` and the live worker atomic writers used the fixed path `status.json.tmp` for `status.json`.
- A deterministic two-writer regression synchronized both temporary writes, let one writer replace the shared file, and then made the second writer fail with `FileNotFoundError` because its temporary file had already been moved.
- This race was relevant to controller process-exit finalization overlapping a worker terminal status write.

Implemented:

- Give controller atomic writes a temporary path scoped by process ID and thread ID.
- Give worker formatted and compact atomic JSON writes the same per-writer isolation.
- Preserve the existing bounded write/replace retry behavior and final JSON filenames.
- Update temporary-file lock regressions for the isolated name format.
- Add a deterministic cross-module concurrency regression covering controller and worker writes to one destination.

Verification:

```text
Clean baseline: Ruff passed, Pytest 158 passed
Reproduction before fix: FileNotFoundError replacing shared status.json.tmp
Focused atomic-write regressions: 3 passed
Collision regression repeated independently: 5/5 passed
Full Ruff: passed
Full Pytest: 159 passed
Sensitive information scan: no findings
```

Checkpoint commit: `4d7ad2db6b08f30cd5a3841d52d117a8e368dd08`.


### Goal: stop unattended review before training when preflight validation fails

Acceptance criterion: the long-running autonomous review orchestrator must not launch any training experiment after Ruff, Pytest, or the environment smoke check returns a non-zero exit code. It should stop at the first failed preflight command and propagate that exit code.

Observed baseline:

- The orchestrator invoked Ruff, Pytest, and the Stage-3 environment check but discarded every return code.
- A deterministic regression made the Pytest preflight return exit code 7.
- The script still launched all configured training branches, ran final checks, printed a successful completion message, and returned zero.

Implemented:

- Capture each preflight command exit code.
- Return immediately on the first non-zero result before constructing or launching any training experiment.
- Preserve successful preflight behavior and the existing command logging path.
- Add an isolated main-orchestrator regression with mocked commands and a temporary project root.

Verification:

```text
Reproduction before fix: pytest exit 7 ignored; orchestrator returned 0
Focused autonomous-review regression: 1 passed
Full Ruff: passed
Full Pytest: 160 passed
Sensitive information scan: no findings
```

Checkpoint commit: `PENDING`.


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
- Desktop-controlled live runs and checkpoints: `lab/runs/*`

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
