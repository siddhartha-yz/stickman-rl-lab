# Stickman RL Lab

A modular two-dimensional reinforcement-learning laboratory built with **Gymnasium**, **PyMunk**, **Pygame**, **Stable-Baselines3 PPO**, and **PyTorch**. An agent controls eight angle-limited rotary joints on a multi-rigid-body stickman and learns to move toward a red target inside a closed room.

## Verified capabilities

- Closed room with floor, side walls, and ceiling.
- Ten dynamic rigid bodies: head, torso, paired upper arms, forearms, thighs, and shins.
- Passive limited neck plus actuated shoulders, elbows, hips, and knees.
- Continuous eight-value action space controlling target joint angular velocities.
- Normalized 49-value default observation vector with torso motion, joint state, relative body positions, target displacement/distance, contact flags, and posture.
- Optional world-fixed obstacle-ray proximities append nine values for a versioned 58-value Stage-3 observation without changing existing checkpoints.
- Decomposed reward terms logged separately to TensorBoard.
- Headless training, real-time rendering, RGB-array rendering, random debug mode, evaluation, trajectory export, and GIF recording.
- Browser-based live training console that launches PPO from random weights, streams the current PyMunk rigid-body state and metrics, and controls pause, resume, stop, and checkpoint saving.
- Configuration-driven fixed targets, random targets, obstacles, upright shaping, and walking-oriented shaping.
- Boxes, walls, platforms, slopes, and trench floor gaps.
- Runtime reward annealing for gradual curriculum transitions.
- Latest, periodic, and best checkpoints; final-vs-best evaluation summaries; reproducibility metadata and config snapshots.
- Ruff static checks and a 29-test suite covering PPO train/save/reload, obstacle-ray geometry, strict route transitions, actor interpolation, phase-balanced distillation, live physics snapshots, trajectory serialization, and training-control APIs.

## Measured results

The first curriculum stage deliberately allows crawling, rolling, and other non-human movement.

- Verified stage-1 deterministic model: `checkpoints/repeat4-stage1/best/best_model.zip`
- Independent stage-1 evaluation: **20/20 successes**, mean 337 decision steps, mean final distance 0.346.
- Stage-2 randomized-target best model: `checkpoints/stage2-random-targets/best/best_model.zip`
- Independent stage-2 evaluation: **56% success over 50 random targets**, mean final distance 0.990.
- Verified medium single-box route policy: `checkpoints/agent-medium-lidar-clearance-seed17-65k/best/best_model.zip`
- Independent medium-course evaluation: **30/30 deterministic successes**, mean 333.5 decision steps, mean final distance 0.436, 100% strict route completion.
- Recommended full box-plus-platform policy: `checkpoints/stage3-full-recommended/model.zip`
- Independent full-course deterministic evaluation: **54/80 successes (67.5%)** across two disjoint 40-target sets, with **100% strict two-waypoint route completion**.
- The remaining failures concentrate in the farthest target interval (`x >= 10.1`); this is a majority-success result, not a claim of perfect robustness. See `PROGRESS.md` for failed PPO branches, distillation artifacts, hashes, and target-bin diagnostics.

## Installation

Python 3.10 or newer is required.

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e .
```

On Windows, a deeply nested repository may exceed the legacy path limit while installing PyTorch. The verified workaround for this workspace is:

```bat
python -m venv C:\rlv
C:\rlv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
C:\rlv\Scripts\python.exe -m pip install -r requirements.txt
C:\rlv\Scripts\python.exe -m pip install -e .
mklink /J .venv C:\rlv
```

## Native desktop training console

The preferred interface is now a native Tk desktop application. It launches the PPO worker directly, consumes structured physics/status/metric events through a local standard-output pipe, and does not require a browser, FastAPI, Vite, or a localhost HTTP port.

Launch it with:

```bat
C:\rlv\Scripts\python.exe scripts\run_desktop.py
```

The desktop console can start a new random-weight PPO run, render the current ten-body PyMunk state, display live reward/success/distance/loss metrics, inspect joint actions, browse local run records, and pause, resume, save, or stop the trainer. Runs remain reproducible under `lab/runs/<run-id>/` with request, control, status, metrics, snapshots, TensorBoard data, and checkpoints.

A real native-window smoke test is available:

```bat
C:\rlv\Scripts\python.exe scripts\run_desktop.py --smoke
```

It performs a 64-step PPO run, captures `reports/desktop-training-console.png`, writes `reports/desktop-smoke.json`, and exits. The verified smoke completed 64/64 steps, received live frames containing all ten rigid bodies, and exited with code 0.

## Legacy web training console (temporary during migration)

The browser interface remains available only until the native migration is independently committed and the web-specific files are removed. It is a real training control plane, not a prerecorded demo. Creating a session starts a separate Python process, initializes a new Stable-Baselines3 PPO policy from random weights, and trains it inside the PyMunk environment. Dynamic physics frames travel directly from the trainer to the FastAPI process over a localhost WebSocket and are pushed to the browser; low-frequency JSON snapshots remain only as a recovery fallback.

Build and launch the legacy interface:

```bat
C:\rlv\Scripts\python.exe -m pip install -r requirements.txt
cd frontend
npm install
npm run build
cd ..
C:\rlv\Scripts\python.exe scripts\run_lab.py
```

Open `http://127.0.0.1:8000`, choose a stage, number of timesteps, seed, PPO config, and optional environment override, then press **创建并启动真实训练**. Stage 1 with `configs/train_live.yaml` is the recommended first from-scratch run.

The live console provides:

- a newly initialized PPO process for every run;
- current episode rigid-body positions, angles, target, obstacles, waypoints, and action outputs;
- total timesteps, FPS, episode reward, rolling success rate, final distance, and PPO losses;
- pause, resume, stop, manual checkpoint save, periodic checkpoints, and final/stopped model saving;
- persistent run records under `lab/runs/<run-id>/`, including request, control, status, metrics, frame snapshots, worker log, TensorBoard data, and checkpoints.

The browser does not calculate physics and does not replay a pre-exported trajectory during live training. It visualizes states emitted by the currently executing Python environment. Incoming source states are interpolated with `requestAnimationFrame` so display rendering follows the monitor refresh rate while preserving the real trainer states as endpoints. A measured Stage-1 smoke run delivered 148 states in 5.018 seconds (29.5 source FPS) while PPO trained at about 201 steps/s. Historical version-2 trajectory export and the experiment-manifest API remain available for offline analysis, but they are no longer the main UI.

## Core commands

Validate the environment and run tests:

```bat
.venv\Scripts\python.exe scripts\check_env.py --stage 0 --steps 1000
.venv\Scripts\python.exe scripts\check_env.py --stage 3 --steps 1000
.venv\Scripts\python.exe -m ruff check src scripts tests
.venv\Scripts\python.exe -m pytest -q
```

Watch verified policies:

```bat
.venv\Scripts\python.exe scripts\demo.py --model checkpoints\repeat4-stage1\best\best_model.zip --stage 1
.venv\Scripts\python.exe scripts\demo.py --model checkpoints\stage3-full-recommended\model.zip --stage 3 --env-config configs\stage3_lidar_waypoints_goalboost.yaml
```

Run random-action physics debugging:

```bat
.venv\Scripts\python.exe scripts\random_debug.py --stage 0 --steps 3000
```

Train from scratch or resume:

```bat
.venv\Scripts\python.exe scripts\train.py --stage 1 --timesteps 100000 --run-name stage1-production --train-config configs\train_tuned.yaml
.venv\Scripts\python.exe scripts\train.py --stage 2 --timesteps 100000 --resume checkpoints\stage1-production\best\best_model.zip --run-name stage2-production --train-config configs\train_tuned.yaml
```

Use an environment override for a sub-curriculum:

```bat
.venv\Scripts\python.exe scripts\train.py --stage 3 --env-config configs\stage3_medium.yaml --timesteps 100000 --resume checkpoints\stage2-random-targets\best\best_model.zip --run-name stage3-medium --train-config configs\train_deterministic.yaml
```

Train the versioned obstacle-ray observation from scratch:

```bat
.venv\Scripts\python.exe scripts\check_env.py --stage 3 --env-config configs\stage3_medium_lidar.yaml --steps 1000
.venv\Scripts\python.exe scripts\train.py --stage 3 --env-config configs\stage3_medium_lidar.yaml --timesteps 100000 --run-name stage3-medium-lidar --train-config configs\train_tuned.yaml
```

The ray-enabled policy observes 58 values instead of 49. Existing 49-value checkpoints remain valid under default configs, but they cannot be resumed directly into the 58-value policy without an explicit weight-migration step.

Test route-aware progress over the medium box while retaining the 58-value schema:

```bat
.venv\Scripts\python.exe scripts\train.py --stage 3 --env-config configs\stage3_medium_lidar_waypoint.yaml --timesteps 100000 --resume checkpoints\stage3-medium-lidar\best\best_model.zip --run-name stage3-medium-lidar-waypoint --train-config configs\train_deterministic.yaml
```

In this override, the observation target and dense progress reward initially point to a box-top waypoint. They switch back to the red target after the torso reaches the waypoint or crosses the configured obstacle boundary. Final success and reported final distance always remain tied to the red target.

Gradually transition reward weights while resuming:

```bat
.venv\Scripts\python.exe scripts\train.py --stage 4 --resume checkpoints\stage3-run\best\best_model.zip --timesteps 100000 --anneal-from-stage 3 --anneal-timesteps 50000 --train-config configs\train_tuned.yaml
```

Evaluate deterministic or stochastic behavior:

```bat
.venv\Scripts\python.exe scripts\evaluate.py checkpoints\repeat4-stage1\best\best_model.zip --stage 1 --episodes 20
.venv\Scripts\python.exe scripts\evaluate.py checkpoints\stage2-random-targets\best\best_model.zip --stage 2 --episodes 50
.venv\Scripts\python.exe scripts\evaluate.py checkpoints\some-run\final_model.zip --stage 2 --episodes 20 --stochastic
.venv\Scripts\python.exe scripts\evaluate.py checkpoints\stage3-full-recommended\model.zip --stage 3 --episodes 40 --seed 1000 --env-config configs\stage3_lidar_waypoints_goalboost.yaml --output reports\stage3-full-seed1000.json
.venv\Scripts\python.exe scripts\evaluate_detailed.py checkpoints\stage3-full-recommended\model.zip --stage 3 --episodes 40 --seed 1000 --env-config configs\stage3_lidar_waypoints_goalboost.yaml --json reports\stage3-full-detailed.json --csv reports\stage3-full-detailed.csv
.venv\Scripts\python.exe scripts\evaluate_random.py --stage 1 --episodes 20
```

Record trajectories and GIFs:

```bat
.venv\Scripts\python.exe scripts\record_trajectory.py --model checkpoints\repeat4-stage1\best\best_model.zip --stage 1 --output trajectories\stage1.npz --gif videos\stage1.gif
.venv\Scripts\python.exe scripts\record_trajectory.py --model checkpoints\agent-medium-lidar-clearance-seed17-65k\best\best_model.zip --stage 3 --env-config configs\stage3_medium_lidar_waypoint.yaml --output trajectories\stage3-medium.npz --gif videos\stage3-medium.gif
.venv\Scripts\python.exe scripts\record_trajectory.py --model checkpoints\stage3-full-recommended\model.zip --stage 3 --env-config configs\stage3_lidar_waypoints_goalboost.yaml --seed 1000 --output trajectories\stage3-full.npz --gif videos\stage3-full.gif
```

Plot evaluation curves:

```bat
.venv\Scripts\python.exe scripts\plot_evaluations.py logs\repeat4-stage1\eval\evaluations.npz --output-dir reports --prefix stage1
```

Open TensorBoard:

```bat
.venv\Scripts\python.exe -m tensorboard.main --logdir logs
```

## Configuration and curriculum

- `configs/base.yaml`: room, gravity, solver, action repeat, body dimensions, joints, target, episode, rendering, and obstacles.
- `configs/rewards.yaml`: decomposed reward weights.
- `configs/stage0.yaml` to `stage5.yaml`: main curriculum stages.
- `configs/stage3_easy.yaml` and `stage3_medium.yaml`: obstacle sub-curricula using the legacy 49-value observation.
- `configs/stage3_lidar.yaml` and `stage3_medium_lidar.yaml`: full and medium obstacle curricula with nine obstacle-ray proximity values.
- `configs/stage3_medium_lidar_waypoint.yaml`: the 58-value medium course with strict route-aware progress.
- `configs/stage3_lidar_waypoints_goalboost.yaml`: the verified full box-plus-platform course with two strict waypoints and amplified post-route target progress.
- `configs/stage3_lidar_waypoints_goalboost_far.yaml` and `stage3_lidar_waypoints_goalboost_ultrafar.yaml`: focused far-target diagnostic/data-collection curricula.
- `configs/train.yaml`: default PPO parameters.
- `configs/train_tuned.yaml`: low-entropy training used for the successful stage-1 run.
- `configs/train_deterministic.yaml`: lower-variance, lower-learning-rate fine-tuning.
- `configs/train_smoke.yaml`: very small orchestration validation.
- `configs/train_live.yaml`: responsive one-environment PPO settings used by the live training console.

Random target sampling rejects positions overlapping obstacles or trench gaps. Stage 3 places targets beyond the obstacle course so episodes are physically meaningful.

## Project layout

```text
configs/                 Environment, reward, curriculum, and PPO settings
src/stickman_rl/         Physics, articulated body, Gym environment, rewards, rendering, training, evaluation
scripts/                 Training, evaluation, live worker, training API, trajectory export, and launcher
frontend/                React/Vite live training console and production build
lab/runs/                 UI-created run state, live frames, metrics, logs, and checkpoints
lab/experiments.json      Optional reproducible offline trajectory manifest
tests/                   Environment, curriculum, rendering, PPO, live snapshot, trajectory, and API tests
checkpoints/             Periodic, best, and final models with summaries/config snapshots
logs/                    TensorBoard and EvalCallback logs
trajectories/            Versioned compressed physics/reward/action traces
videos/                  GIF demonstrations
reports/                 Evaluation plots, JSON/CSV evidence, and observer screenshots
```

## Engineering notes

Physics, rewards, rendering, training, and evaluation are separate modules. Configuration values are not scattered through the Pygame loop. Self-collision is disabled by default for initial stability but remains configurable. Four-frame action repeat was added after experiments showed one-frame commands produced policies that relied on exploration noise; it enabled the first verified deterministic stage-1 solution.

The full-course recommended checkpoint was produced by collecting real stochastic successes, actor-only anchored distillation, and conservative actor interpolation. The parameter basin is unusually sensitive: adjacent interpolation coefficients can collapse from majority success to one-waypoint failure, so exact checkpoint hashes and large independent evaluation sets are retained.

The live console uses an independent worker process and atomic JSON snapshots so the FastAPI server remains responsive during training. The current transport is frequent HTTP polling rather than WebSocket; the data is still generated by the active trainer, and control commands are consumed by the PPO callback at environment-step boundaries.

Stages 4-5 are architectural and experimental targets. The repository does **not** claim that natural upright walking has been solved.
