# RoboStep benchmark release

This folder is an independent release of the RoboStep method used for the
75-task main comparison: 50 Meta-World tasks, 19 ManiSkill tasks, and 6
SoftGym tasks. It contains the complete task-level Frame-VLM reward programs,
the active-stage reward machine, GateRule/GateVLM interfaces, matched PPO
defaults, environment setup instructions, and one-task training launchers.

The release does not import a sibling checkout, a paper workspace, or a local
deployment project. The only external components are the benchmark packages
and their own simulator assets, plus an optional VLM checkpoint.
UR3/ROS/real-robot deployment code and anonymous paper links are intentionally
outside this release.

## Layout

```text
robostep/                  core reward machine, gates, VLM protocol, PPO
task_registry/reward_machines/
                           75 complete Frame-VLM task programs
main_method/               release-local task-specific reward/feature code
experiments/compositional/
                           measured Meta-World M10 runner and evaluator
experiments/recovery/      ManiSkill disturbance-to-GateVLM runner
artifacts/checkpoints/     selected RL checkpoints used by these reports
task_registry/reward_machine.py
                           loader + callable task-level machine
task_registry/manifest.json
                           machine-readable 50/19/6 index
task_registry/task_manifest.csv
                           flat 75-row inventory for scripts and audits
examples/train_task.py     one real native-environment PPO run
scripts/                   one-command launchers for each benchmark family
configs/                   Frame-VLM request examples
tests/                     reward/registry/protocol tests
```

## Install the release

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[torch,benchmarks]"
python tools/verify_task_registry.py
python -m unittest discover -s tests -v
```

The package itself only requires Python, NumPy, and the optional PyTorch
extra. Benchmark packages are installed separately because they manage their
own native physics and rendering dependencies.

## Configure the benchmark environments

### Meta-World MT50

```bash
python -m pip install metaworld
```

The native state/proprioceptive observation and action space are preserved.
The release appends the active-stage one-hot only for the staged policy. RGB is
requested only by the gate adapter. The official task success signal is used
for evaluation and terminal-only reward, never as a dense feature.

### ManiSkill 3

```bash
python -m pip install "mani-skill==3.0.1"
```

Use the ManiSkill installation guide to configure SAPIEN/Vulkan and download
the assets required by the selected task. The matched state-policy setup is:

```python
import gymnasium as gym
env = gym.make(
    "PickCube-v1",
    obs_mode="state",
    control_mode="pd_joint_delta_pos",
    render_mode=None,             # use "rgb_array" for GateVLM packets
)
```

The actor and critic use the benchmark's three-layer 256-wide Tanh block. The
native state API is converted to the named features in the corresponding
reward machine. The supplied PickCube adapter is executable; other tasks use
the same adapter contract and expose their task-specific state features.

### SoftGym

SoftGym/PyFlex is normally installed from its own checkout because the FleX
build and assets are not distributed as a normal wheel. Follow the upstream
SoftGym build instructions, activate that environment, and make its Python
package importable before launching `scripts/train_softgym_task.sh`.

The native particle/cup/rope state and action interface are preserved. The six
release tasks are `ClothDrop`, `ClothFlatten`, `ClothFold`, `RopeFlatten`,
`PassWater`, and `PourWater`. The task metric used for reporting is not fed
back into the dense reward machine.

More adapter details are in [docs_BENCHMARK_ADAPTERS.md](docs_BENCHMARK_ADAPTERS.md).

The benchmark packages and simulator assets are environment dependencies, not
reward code. The task-specific feature extraction and progress functions are
code and are included under `main_method/`: Meta-World keeps its task-family
cores, ManiSkill keeps its staged state/recovery API, and SoftGym keeps
`stage_env.py` with the cloth, rope, and fluid recipes. The environment
adapter only supplies native simulator state and RGB packets; it does not
replace these functions with official success or a generic distance.

## Load and call a task reward machine

Every file in `task_registry/reward_machines/` contains a complete frozen
Frame-VLM program. It is not a task-name summary: it contains the ordered
stages, named dense terms, term direction, completion predicate, VLM evidence
contract, and the shared reward configuration.

```python
from task_registry import load_reward_machine

machine = load_reward_machine("PickCube-v1", "ManiSkill")
machine.reset({"tcp_obj": 0.8, "grasped": 0.0, "obj_goal": 1.0})
step = machine.step(
    {"tcp_obj": 0.6, "grasped": 1.0, "obj_goal": 1.0},
    candidates=[True, False, False],
)
print(step.reward, step.stage_after, step.transition)
```

An adapter returns normalized feature values in `[0, 1]`. Terms marked
`decrease` are normalized costs and are inverted by `TaskRewardMachine`; terms
marked `increase` are already progress values. The adapter supplies one
candidate boolean per stage. A GateVLM response can be passed as
`transition_authorization`; an abstention blocks the transition.

## One-command real task runs

The launchers call a native benchmark environment and the release PPO trainer;
they do not use a synthetic environment.

```bash
# ManiSkill: fully wired release example
bash scripts/train_maniskill_pickcube.sh

# Meta-World: select a task and seed
TASK_ID=reach-v3 SEED=42 bash scripts/train_metaworld_task.sh

# SoftGym: select a task after installing SoftGym/PyFlex
TASK_ID=PassWater SEED=42 bash scripts/train_softgym_task.sh
```

The ManiSkill `PickCube-v1` command is the turnkey native example included in
this release. The Meta-World and SoftGym launchers select the native task and
then use the same `examples/train_task.py` path; their task wrapper must expose
the adapter hooks described below before training can start. This keeps an
incomplete feature mapping from silently substituting the benchmark's official
success signal as dense reward.

The direct Python form is:

```bash
python examples/train_task.py \
  --benchmark ManiSkill \
  --task PickCube-v1 \
  --seed 42 \
  --total-timesteps 4096 \
  --output runs/pickcube_seed42.json
```

For a task not yet covered by a native adapter, add
`get_robostep_features()` and `get_robostep_candidates()` to the benchmark
wrapper. The reward machine and PPO command do not change. This keeps the
task-specific privileged state extraction explicit rather than silently using
the official success API.

## Reproducing the main-table protocol

For each task and seed:

1. load its package-local reward machine;
2. keep the benchmark-native action/controller and base state observation;
3. append the active-stage code for actor and critic;
4. choose `GateRule` or a frozen GateVLM cache;
5. train with the benchmark row in `PPOConfig.for_benchmark(...)`;
6. select a checkpoint on the validation episodes only;
7. evaluate held out episodes with the official task metric;
8. write one JSON summary and aggregate the summaries afterwards.

The release does not silently copy a benchmark's native dense reward into the
RoboStep machine. Native dense, terminal-only, and per-subtask controls are
separate reward modes in `ActiveStageReward`.

## Main-method fidelity and measured extensions

`robostep/reward.py::ActiveStageReward` is the release implementation of the
shared stage-reward engine used by the main runs. Its active-stage potential
difference, clipping, dwell counter, candidate mask, transition authority,
transition bonus, stage-entry baseline, and maintenance/safety accounting are
kept in sync with the release-local ManiSkill copy at
`main_method/maniskill/stage_reward/core.py::BatchedStageReward`.

The benchmark-specific code is not an approximation: the Meta-World
`*_core.py` modules, the SoftGym `stage_env.py` and recipe classes, and the
ManiSkill staged/recovery modules are included under `main_method/`. The JSON
files in `task_registry/reward_machines/` are the frozen Frame-VLM programs;
the native benchmark adapters turn simulator state into the named features
those programs consume. Official success remains a terminal/evaluation signal
and is not substituted for dense progress.

The compositional forward-chain and disturbance-recovery experiments are also
included under `experiments/`, with the reportable RL checkpoints under
`artifacts/checkpoints/`. VLM weights are intentionally external.

### Reproduce the measured Meta-World M10 composition

After installing Meta-World, MuJoCo, TensorDict, and the PPO runtime:

```bash
python -m pip install -e ".[torch,m10]"
```

```bash
bash experiments/compositional/run_m10.sh
bash experiments/compositional/eval_m10.sh \
  artifacts/checkpoints/compositional_m10/model_6999.pt
```

The packaged M10 checkpoint includes its frozen `config.json` task definition:
the ten destination chain, three atomic stages per macro, stage-code width,
reward constants, and PPO configuration. To run the frozen actor and record one
directly viewable policy rollout as MP4:

```bash
bash experiments/compositional/record_m10.sh
```

The video is written to `runs/compositional/m10_video/m10_rule_policy.mp4`.
This is inference only; it does not retrain the checkpoint. `ffmpeg` and the
Meta-World installation are required for rendering.

The runner fixes `pick-place-v3`, ten macros, three atomic stages per macro,
31-way stage code, `tabletop_unique_chain_v8`, seed 42, a 128-env/64-step
rollout, 700 iterations per macro, and no reset between macro handoffs.

### Reproduce disturbance-to-GateVLM recovery

Place `Qwen/Qwen3.5-9B` at `models/Qwen3.5-9B`, or set
`ROBOSTEP_QWEN_MODEL`, after installing ManiSkill/SAPIEN:

```bash
bash experiments/recovery/run_gatevlm_recovery.sh
```

Set `ROBOSTEP_RECOVERY_TASK=PickSingleYCB-v1` for the second packaged policy.
The actor checkpoint is frozen; GateVLM receives post-disturbance RGB evidence
and predicts the earlier stage index. Invalid or abstaining responses fail
closed and never fall back to an official-success label.

## Frame-VLM and GateVLM

`FrameVLMCompiler` produces/validates the task reward program outside PPO.
`GateVLM` receives low-frequency RGB frame packets only at candidate
transitions and returns `accept`, `reject`, or `abstain`. The worker protocol,
cache format, and freeze step are documented in
[docs_VLM_PROTOCOL.md](docs_VLM_PROTOCOL.md). Model weights are intentionally
not bundled.

To inspect a frozen task program:

```bash
python examples/inspect_task_registry.py StackPyramid-v1 --benchmark ManiSkill
python examples/inspect_task_registry.py PassWater --benchmark SoftGym
```
