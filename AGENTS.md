# AGENTS.md

## Cursor Cloud specific instructions

This repo is a Python research monorepo for vision-based spacecraft docking. Three components:

- `src/` — standalone PyTorch vision library (`SplitPosePredictor`) for 6D pose estimation.
- `spacecraft_docking_controller/sdc_core/` — pure NumPy/SciPy guidance & control library (Clohessy-Wiltshire dynamics, PID/LQR/MPC controllers, EKF, trajectory generation).
- ROS 2 packages (`ros2_isaac_inference/`, `spacecraft_docking_controller/` nodes + launch files) that bridge the above to NVIDIA Isaac Sim.

### Environment

- Python deps live in the venv at `/workspace/.venv` (created by the update script from `requirements.txt`). Run everything with `/workspace/.venv/bin/python` (or activate it). CPU-only here: `torch.cuda.is_available()` is `False`, and the vision code auto-falls back to CPU.
- No linter, formatter, or automated test suite is configured (no `pytest`/`ruff`/`flake8`/pre-commit). Use `python -m py_compile <files>` as a smoke/lint check.

### Import-path gotcha (non-obvious)

The vision modules mix package and flat imports: `src/pose_inference.py` uses `from src.pose_model import ...`, but `src/orientation_model_alt.py` uses `from orientation_model import ...`. To import the vision library you must put BOTH the repo root and `src/` on the path, e.g. run from `/workspace` with `PYTHONPATH=/workspace:/workspace/src`. Do not "fix" these imports unless asked.

For `sdc_core`, run with `PYTHONPATH=/workspace/spacecraft_docking_controller` so `import sdc_core...` resolves (it is a colcon package, not pip-installed).

### Running the runnable products (no ROS / Isaac Sim needed)

- Vision 6D pose inference: instantiate `SplitPosePredictor("models/pose_net_baseline.pt", "models/orientation_net_efficientnet_b4.pt", backbone="resnet18")` and call `predict(image)` on an `(H,W,3)` uint8 array. Model checkpoints are committed under `models/`.
- Docking control: use `sdc_core.dynamics.ClohessyWiltshireDynamics` + `sdc_core.controllers.LQRController` (`compute_control(state, target)` where state is `[x,y,z,vx,vy,vz]`, control is `[ax,ay,az]`) and `propagate(...)` for a closed-loop simulation. `LQRController` reads `self.dt` (default `0.1`); after changing `dt`, call `_compute_gain()`.
- Offline analysis notebook: `spacecraft_docking_controller/notebooks/docking_analysis.ipynb` (Jupyter is installed).

### Out of scope in this VM

The full closed-loop ROS 2 + NVIDIA Isaac Sim pipeline cannot run here: ROS 2 is not installed and Isaac Sim needs a GPU + external simulator. `set VISION_BENCHMARK_ROOT=/workspace` and follow `README.md` / `spacecraft_docking_controller/ISAAC_SIM_SETUP.md` on a ROS 2 + GPU machine for that path.
