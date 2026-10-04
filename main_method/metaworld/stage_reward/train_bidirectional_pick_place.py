"""Train the same-object multi-destination MetaWorld composition experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random

import numpy as np
import torch
try:
    from rsl_rl.networks import EmpiricalNormalization
except ImportError:  # rsl-rl-lib >= 5 exposes it from modules.
    from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.runners import OnPolicyRunner

from .grasp_insert_vec_env import MetaWorldGraspInsertVecEnv
from .suite_qwen_gate import MandatoryQwenStageGate
from .train_drawer import ISAAC_PYTHON, METAWORLD_ROOT, QWEN_MODEL, QWEN_WORKER
from .train_grasp_insert import QWEN_PROMPT, task_reward_config
from .train_target_reach import runner_config

TASK = "pick-place-v3"
ATOMIC_STAGE_NAMES = ("pregrasp", "capture_and_lift", "place_and_release")
ATOMIC_STAGE_INSTRUCTIONS = (
    "The active macro destination is the visible goal marker. Audit only whether the TCP is at the safe pregrasp pose above the same movable object; do not infer centimeter-level geometry from pixels.",
    "The active macro destination is the visible goal marker. Audit only whether the same object has been captured and visibly lifted/displaced by the robot; reject empty-space motion or a wrong object.",
    "The active macro destination is the visible goal marker. Audit only whether the same object is visibly at the current destination region; reject a wrong target or robot-only motion.",
)
MAX_MACROS = 15
STAGE_ID_SIZE = 3 * MAX_MACROS + 1
STATE_OBSERVATION_SIZE = 39
RAW_STAGE_CONTEXT_PROTOCOL = "state39_running_norm_plus_raw_global_stage_onehot"
APPEND_STAGE_CONTEXT_PROTOCOL = "state39_running_norm_plus_raw_global_stage_onehot_tiled_by_local_stage"


class StateOnlyEmpiricalNormalization(EmpiricalNormalization):
    """Normalize MetaWorld state while leaving the discrete stage one-hot unchanged."""

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = super().forward(values)
        return torch.cat(
            (
                normalized[..., :STATE_OBSERVATION_SIZE],
                values[..., STATE_OBSERVATION_SIZE:],
            ),
            dim=-1,
        )


def _install_state_only_normalizers(policy: torch.nn.Module) -> None:
    for name in ("actor_obs_normalizer", "critic_obs_normalizer"):
        old = getattr(policy, name)
        replacement = StateOnlyEmpiricalNormalization(
            int(old._mean.shape[-1]), eps=float(old.eps), until=old.until
        ).to(old._mean.device)
        replacement.load_state_dict(old.state_dict())
        replacement.train(old.training)
        setattr(policy, name, replacement)


def _convert_legacy_stage_normalization_and_tile(policy: torch.nn.Module) -> None:
    """Preserve the learned A mapping, then give appended stages a sane prior."""
    with torch.no_grad():
        for network_name, normalizer_name in (
            ("actor", "actor_obs_normalizer"),
            ("critic", "critic_obs_normalizer"),
        ):
            network = getattr(policy, network_name)
            normalizer = getattr(policy, normalizer_name)
            first_layer = network[0]
            stage_slice = slice(
                STATE_OBSERVATION_SIZE,
                STATE_OBSERVATION_SIZE + STAGE_ID_SIZE,
            )
            old_stage_weights = first_layer.weight[:, stage_slice].clone()
            stage_mean = normalizer._mean[0, stage_slice]
            stage_denom = normalizer._std[0, stage_slice] + float(normalizer.eps)
            first_layer.weight[:, stage_slice].copy_(
                old_stage_weights / stage_denom.unsqueeze(0)
            )
            first_layer.bias.sub_(
                torch.sum(
                    old_stage_weights
                    * (stage_mean / stage_denom).unsqueeze(0),
                    dim=1,
                )
            )
            for stage_index in range(3, STAGE_ID_SIZE):
                source_index = stage_index % 3
                first_layer.weight[:, STATE_OBSERVATION_SIZE + stage_index].copy_(
                    first_layer.weight[
                        :, STATE_OBSERVATION_SIZE + source_index
                    ]
                )


def _checkpoint_stage_context_protocol(checkpoint: Path) -> str | None:
    config_path = checkpoint.expanduser().resolve().parent / "config.json"
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    return saved.get("task_definition", {}).get("stage_context_normalization")


def stage_semantics(macro_count: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    names: list[str] = []
    instructions: list[str] = []
    for macro in range(macro_count):
        for atomic_name, instruction in zip(
            ATOMIC_STAGE_NAMES, ATOMIC_STAGE_INSTRUCTIONS, strict=True
        ):
            names.append(f"macro{macro + 1}_{atomic_name}")
            instructions.append(
                f"This is macro-subtask {macro + 1} of {macro_count}. "
                f"{instruction} The next macro, if any, begins only after this "
                "macro reaches its destination."
            )
    return tuple(names), tuple(instructions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--macro-count", type=int, choices=range(1, MAX_MACROS + 1), required=True
    )
    parser.add_argument(
        "--stage-id-size",
        type=int,
        default=STAGE_ID_SIZE,
        help="fixed global-stage one-hot width, including the terminal slot",
    )
    parser.add_argument("--gate", choices=("rule", "qwen", "synthetic"), default="rule")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--base-iterations", type=int, default=250)
    parser.add_argument("--steps-per-env", type=int, default=64)
    parser.add_argument("--max-episode-length", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-device", default="cpu")
    parser.add_argument("--qwen-gpu", type=int, default=0)
    parser.add_argument("--qwen-camera", default="corner3")
    parser.add_argument("--source-macro-index", type=int, default=0, help="independent suffix task index; 0 is the native A target")
    parser.add_argument(
        "--geometry-protocol",
        choices=(
            "native_forward_v0",
            "tabletop_supported_chain_v1",
            "tabletop_bidirectional_v2",
            "tabletop_balanced_chain_v2",
            "tabletop_square_chain_v3",
            "tabletop_square_chain_v4",
            "tabletop_square_chain_v5",
            "tabletop_unique_chain_v6",
            "tabletop_unique_chain_v8",
            "tabletop_unique_chain_v9",
            "tabletop_unique_chain_v10",
            "tabletop_reset_v1",
        ),
        default="tabletop_supported_chain_v1",
    )
    parser.add_argument("--reset-between-macros", action="store_true", help="full environment/object reset between macros")
    parser.add_argument("--reset-arm-between-macros", action="store_true", help="reset only Sawyer arm/gripper; keep object at previous macro destination")
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--init-noise-std", type=float, default=0.35)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--raw-stage-onehot",
        action="store_true",
        help="normalize only the 39 continuous state values; keep stage ID at 0/1",
    )
    parser.add_argument(
        "--append-curriculum",
        action="store_true",
        help=(
            "append one macro from a trained prefix: preserve state normalization, "
            "leave the fixed stage one-hot unnormalized, tile local-stage input "
            "weights into new global stage slots, and reset exploration std"
        ),
    )
    parser.add_argument("--learn-iterations", type=int, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/compositional"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.macro_count < 1 or args.macro_count > MAX_MACROS:
        raise SystemExit(f"macro-count must be in 1..{MAX_MACROS}")
    minimum_stage_id_size = 3 * args.macro_count + 1
    if args.stage_id_size < minimum_stage_id_size:
        raise SystemExit(
            f"stage-id-size must be at least {minimum_stage_id_size} for "
            f"macro-count={args.macro_count}"
        )
    if args.append_curriculum and args.init_checkpoint is None:
        raise SystemExit("--append-curriculum requires --init-checkpoint")
    if args.append_curriculum:
        args.raw_stage_onehot = True
    if args.append_curriculum and (
        args.reset_between_macros or args.reset_arm_between_macros
    ):
        raise SystemExit("the main append-curriculum protocol is physically continuous")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    stage_names, stage_instructions = stage_semantics(args.macro_count)
    args.iterations = (
        int(args.learn_iterations)
        if args.learn_iterations is not None
        else args.base_iterations * args.macro_count
    )
    if args.iterations <= 0:
        raise SystemExit("learn-iterations must be positive")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or (
        f"pick_place_M{args.macro_count}_{args.gate}_scratch_seed"
        f"{args.seed}_{timestamp}"
    )
    run_dir = (
        args.output_root.expanduser().resolve()
        / TASK
        / f"M{args.macro_count}"
        / args.gate
        / run_name
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    gate = None
    if args.gate == "qwen":
        gate = MandatoryQwenStageGate(
            output_dir=run_dir / "mandatory_qwen",
            python_executable=ISAAC_PYTHON,
            worker_script=QWEN_WORKER,
            model_path=QWEN_MODEL,
            prompt_template=QWEN_PROMPT,
            stage_names=stage_names,
            stage_instructions=stage_instructions,
            qwen_gpu=args.qwen_gpu,
        )
    env = MetaWorldGraspInsertVecEnv(
        task=TASK,
        stage_instructions=stage_instructions,
        num_envs=args.num_envs,
        seed=args.seed,
        gate_mode=args.gate,
        qwen_gate=gate,
        config=task_reward_config(TASK),
        device=args.train_device,
        max_episode_length=args.max_episode_length * args.macro_count,
        max_macro_length=args.max_episode_length,
        camera_name=args.qwen_camera,
        macro_count=args.macro_count,
        stage_id_size=args.stage_id_size,
        source_macro_index=args.source_macro_index,
        geometry_protocol=args.geometry_protocol,
        reset_between_macros=args.reset_between_macros,
        reset_arm_between_macros=args.reset_arm_between_macros,
    )
    cfg = runner_config(args)
    cfg["experiment_name"] = "metaworld_bidirectional_pick_place"
    serializable_args = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    payload = {
        "args": serializable_args,
        "runner": cfg,
        "environment": env.cfg,
        "task_definition": {
            "macro_unit": "same object placed at one destination",
            "atomic_stages_per_macro": list(ATOMIC_STAGE_NAMES),
            "macro_count": args.macro_count,
            "source_macro_index": args.source_macro_index,
            "active_atomic_stage_count": 3 * args.macro_count,
            "stage_id_size": args.stage_id_size,
            "target_policy": (
                "versioned by geometry_protocol; tabletop_bidirectional_v2 uses "
                "the equal-distance O->A->O->A chain"
            ),
            "geometry_protocol": args.geometry_protocol,
            "episode_reset_between_macros": args.reset_between_macros,
            "arm_reset_between_macros": args.reset_arm_between_macros,
            "prefix_reward_frozen": True,
            "budget_scaling": (
                "one base-iteration block per appended macro; cumulative cost B(M)=base*M"
                if args.append_curriculum
                else "base_iterations * macro_count"
            ),
            "stage_context_normalization": (
                APPEND_STAGE_CONTEXT_PROTOCOL
                if args.append_curriculum
                else (
                    RAW_STAGE_CONTEXT_PROTOCOL
                    if args.raw_stage_onehot
                    else "legacy_joint_running_norm_state_and_stage_onehot"
                )
            ),
            "appended_stage_initialization": (
                "copy first-layer column from the matching local atomic stage"
                if args.append_curriculum
                else None
            ),
        },
        "stage_names": stage_names,
        "stage_instructions": stage_instructions,
        "native_reward_used_for_ppo": False,
        "initial_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "curriculum_protocol": (
            "append_one_macro_from_trained_prefix_no_physical_reset"
            if args.append_curriculum
            else ("generic_checkpoint_initialization" if args.init_checkpoint else None)
        ),
    }
    if env.cfg.get("policy_stage_information") is False:
        payload["task_definition"]["policy_stage_observation"] = False
        payload["task_definition"]["stage_context_normalization"] = "state39_running_norm_no_stage_id"
    (run_dir / "config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    try:
        runner = OnPolicyRunner(env, cfg, log_dir=str(run_dir), device=args.train_device)
        if args.init_checkpoint is not None:
            runner.load(
                str(args.init_checkpoint),
                load_optimizer=False,
                map_location=args.train_device,
            )
            if args.append_curriculum:
                source_protocol = _checkpoint_stage_context_protocol(
                    args.init_checkpoint
                )
                if source_protocol != APPEND_STAGE_CONTEXT_PROTOCOL:
                    _convert_legacy_stage_normalization_and_tile(
                        runner.alg.policy
                    )
                _install_state_only_normalizers(runner.alg.policy)
                with torch.no_grad():
                    runner.alg.policy.std.fill_(args.init_noise_std)
            # Generic checkpoint initialization otherwise preserves the source
            # exploration std and observation normalizers.
        elif args.raw_stage_onehot:
            _install_state_only_normalizers(runner.alg.policy)
        runner.learn(
            num_learning_iterations=args.iterations,
            init_at_random_ep_len=False,
        )
        summary = env.statistics() | {
            "checkpoint": str(
                run_dir / f"model_{runner.current_learning_iteration}.pt"
            ),
            "macro_count": args.macro_count,
            "base_iterations": args.base_iterations,
            "actual_iterations": args.iterations,
        }
        (run_dir / "training_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    main()
