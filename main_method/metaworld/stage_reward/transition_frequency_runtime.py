"""Runtime controls for stage-transition ablations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .transition_prompt_runtime import (
    PROMPT_MODE_HYBRID_NEUTRAL,
    PROMPT_MODE_PERIODIC_NEUTRAL,
    resolve_transition_prompt_mode,
)


ENV_QUERY_MODE = "STAGE_TRANSITION_QUERY_MODE"
ENV_INTERVAL = "STAGE_TRANSITION_QUERY_INTERVAL_STEPS"
ENV_CONSECUTIVE_ACCEPTS = "STAGE_TRANSITION_CONSECUTIVE_ACCEPTS"
ENV_MIN_GAP_STEPS = "STAGE_TRANSITION_MIN_GAP_STEPS"
ENV_SYNTHETIC_FAR = "STAGE_TRANSITION_SYNTHETIC_FAR_ALPHA"
ENV_SYNTHETIC_FRR = "STAGE_TRANSITION_SYNTHETIC_FRR_BETA"
ENV_SYNTHETIC_SEED = "STAGE_TRANSITION_SYNTHETIC_SEED"
ENV_SYNTHETIC_LOG_PATH = "STAGE_TRANSITION_SYNTHETIC_LOG_PATH"
DEFAULT_QUERY_MODE = "candidate_retry"
DEFAULT_INTERVAL = 64
DEFAULT_CONSECUTIVE_ACCEPTS = 1
DEFAULT_MIN_GAP_STEPS = 0
_NEVER_TRANSITIONED_STEP = -(10**12)


@dataclass(frozen=True)
class SyntheticGateOpportunity:
    env_id: int
    gt_complete: bool
    opportunity_type: str


@dataclass(frozen=True)
class SyntheticGateDecision:
    env_id: int
    gt_complete: bool
    opportunity_type: str
    prediction: str
    accepted: bool
    false_accept: bool
    false_reject: bool
    draw: float


def _active_stage_count(owner: object) -> int:
    instructions = getattr(owner, "stage_instructions", None)
    if instructions is not None:
        return len(instructions)
    if getattr(owner, "stage_entries", None) is not None:
        return len(getattr(owner, "stage_entries")) - 1
    if getattr(owner, "active_stage_count", None) is not None:
        return int(getattr(owner, "active_stage_count"))
    raise AttributeError(
        "owner must expose stage_instructions, stage_entries, or active_stage_count"
    )


def _stable_seed(owner: object, base_seed: int) -> int:
    task = str(getattr(owner, "task", "unknown_task"))
    digest = hashlib.blake2b(
        f"{task}:{base_seed}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def configure_transition_schedule(owner: object, cfg: dict[str, object]) -> None:
    mode = os.environ.get(ENV_QUERY_MODE, DEFAULT_QUERY_MODE).strip().lower()
    if not mode:
        mode = DEFAULT_QUERY_MODE
    if mode not in {"candidate_retry", "fixed_frequency", "hybrid_k64"}:
        raise ValueError(f"unsupported {ENV_QUERY_MODE}: {mode}")
    interval = int(os.environ.get(ENV_INTERVAL, str(DEFAULT_INTERVAL)))
    if interval < 1:
        raise ValueError(f"{ENV_INTERVAL} must be positive")
    consecutive_accepts = int(
        os.environ.get(ENV_CONSECUTIVE_ACCEPTS, str(DEFAULT_CONSECUTIVE_ACCEPTS))
    )
    if consecutive_accepts < 1:
        raise ValueError(f"{ENV_CONSECUTIVE_ACCEPTS} must be positive")
    min_gap_steps = int(os.environ.get(ENV_MIN_GAP_STEPS, str(DEFAULT_MIN_GAP_STEPS)))
    if min_gap_steps < 0:
        raise ValueError(f"{ENV_MIN_GAP_STEPS} must be non-negative")
    prompt_mode = resolve_transition_prompt_mode()
    if (
        mode == "hybrid_k64"
        and getattr(owner, "gate_mode", None) == "qwen"
        and prompt_mode != PROMPT_MODE_HYBRID_NEUTRAL
    ):
        raise ValueError("real Hybrid-K64 requires the hybrid_neutral prompt")
    num_envs = int(getattr(owner, "num_envs"))
    synthetic_enabled = getattr(owner, "gate_mode", None) == "synthetic"
    far = float(os.environ.get(ENV_SYNTHETIC_FAR, "0.0"))
    frr = float(os.environ.get(ENV_SYNTHETIC_FRR, "0.0"))
    if not 0.0 <= far <= 1.0:
        raise ValueError(f"{ENV_SYNTHETIC_FAR} must be in [0, 1]")
    if not 0.0 <= frr <= 1.0:
        raise ValueError(f"{ENV_SYNTHETIC_FRR} must be in [0, 1]")
    synthetic_seed = int(os.environ.get(ENV_SYNTHETIC_SEED, "0"))
    setattr(owner, "transition_query_mode", mode)
    setattr(owner, "transition_query_interval_steps", interval)
    setattr(owner, "transition_consecutive_accepts", consecutive_accepts)
    setattr(owner, "transition_min_gap_steps", min_gap_steps)
    setattr(owner, "synthetic_gate_far_alpha", far)
    setattr(owner, "synthetic_gate_frr_beta", frr)
    setattr(owner, "synthetic_gate_seed", synthetic_seed)
    setattr(owner, "_synthetic_gate_rng", np.random.default_rng(_stable_seed(owner, synthetic_seed)))
    setattr(owner, "_synthetic_gate_stats", {
        "opportunities": 0,
        "gt_complete": 0,
        "gt_incomplete": 0,
        "accepts": 0,
        "rejects": 0,
        "false_accepts": 0,
        "false_rejects": 0,
        "true_accepts": 0,
        "true_rejects": 0,
        "transitions": 0,
    })
    log_path_raw = os.environ.get(ENV_SYNTHETIC_LOG_PATH, "").strip()
    log_handle = None
    if synthetic_enabled and log_path_raw:
        log_path = Path(log_path_raw).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8")
        setattr(owner, "synthetic_gate_log_path", str(log_path))
    else:
        setattr(owner, "synthetic_gate_log_path", None)
    setattr(owner, "_synthetic_gate_log_handle", log_handle)
    setattr(owner, "_last_gate_query_step", np.zeros(num_envs, dtype=np.int64))
    setattr(
        owner,
        "_last_stage_transition_step",
        np.full(num_envs, _NEVER_TRANSITIONED_STEP, dtype=np.int64),
    )
    cfg.update(
        {
            "transition_query_mode": mode,
            "transition_query_interval_steps": interval,
            "transition_consecutive_accepts": consecutive_accepts,
            "transition_min_gap_steps": min_gap_steps,
            "transition_prompt_mode": prompt_mode,
            "transition_authority": (
                "qwen_fixed_frequency_visual_only_neutral_prompt"
                if mode == "fixed_frequency" and prompt_mode == PROMPT_MODE_PERIODIC_NEUTRAL
                else "qwen_hybrid_candidate_plus_periodic_neutral_prompt"
                if mode == "hybrid_k64" and getattr(owner, "gate_mode", None) == "qwen"
                else "qwen_fixed_frequency_visual_only"
                if mode == "fixed_frequency"
                else "synthetic_hybrid_candidate_plus_periodic_gt"
                if synthetic_enabled and mode == "hybrid_k64"
                else "candidate_then_gate"
            ),
            "synthetic_gate_enabled": synthetic_enabled,
            "synthetic_gate_far_alpha": far if synthetic_enabled else None,
            "synthetic_gate_frr_beta": frr if synthetic_enabled else None,
            "synthetic_gate_seed": synthetic_seed if synthetic_enabled else None,
            "synthetic_gate_log_path": getattr(owner, "synthetic_gate_log_path", None),
        }
    )


def reset_transition_schedule(owner: object, env_id: int) -> None:
    last = getattr(owner, "_last_gate_query_step", None)
    if last is None:
        return
    episode_length_buf = getattr(owner, "episode_length_buf")
    last[int(env_id)] = int(episode_length_buf[int(env_id)].item())
    last_transition = getattr(owner, "_last_stage_transition_step", None)
    if last_transition is not None:
        last_transition[int(env_id)] = _NEVER_TRANSITIONED_STEP


def select_transition_due(owner: object, candidate_due: Iterable[int]) -> list[int]:
    mode = getattr(owner, "transition_query_mode", DEFAULT_QUERY_MODE)
    candidates = list(candidate_due)
    gate_mode = getattr(owner, "gate_mode", None)
    if mode == "hybrid_k64" and gate_mode == "qwen":
        # Use the exact same opportunity schedule as the synthetic surface,
        # but expose only due env IDs to the visual gate, never GT labels.
        setattr(owner, "_hybrid_qwen_candidate_due", frozenset(candidates))
        opportunities = select_synthetic_gate_opportunities(owner, candidates)
        _record_hybrid_packet_oracle_sidecar(owner, opportunities)
        return [opportunity.env_id for opportunity in opportunities]
    if mode != "fixed_frequency" or gate_mode == "rule":
        return candidates
    interval = int(getattr(owner, "transition_query_interval_steps"))
    min_gap_steps = int(getattr(owner, "transition_min_gap_steps", DEFAULT_MIN_GAP_STEPS))
    last = getattr(owner, "_last_gate_query_step")
    last_transition = getattr(owner, "_last_stage_transition_step", None)
    episode_length_buf = getattr(owner, "episode_length_buf")
    states = getattr(owner, "_states")
    active_stage_count = _active_stage_count(owner)
    due: list[int] = []
    for env_id, state in enumerate(states):
        if int(state.stage) >= active_stage_count:
            continue
        current_step = int(episode_length_buf[env_id].item())
        if min_gap_steps > 0 and last_transition is not None:
            since_transition = current_step - int(last_transition[env_id])
            if since_transition < min_gap_steps:
                continue
        elapsed = current_step - int(last[env_id])
        if elapsed >= interval:
            due.append(int(env_id))
    return due


def _jsonable_privileged(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable_privileged(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable_privileged(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_privileged(item) for item in value]
    return value


def _record_hybrid_packet_oracle_sidecar(
    owner: object, opportunities: Sequence[SyntheticGateOpportunity]
) -> None:
    """Save hidden privileged rule evidence for a separate packet-label audit.

    The operational label matches the synthetic gate's opportunity labels.
    It is not exposed to the VLM and must be audited against independent
    state/mesh completion criteria before being called a semantic GT label.
    """
    raw_path = os.environ.get("STAGE_TRANSITION_PACKET_ORACLE_LOG_PATH", "").strip()
    if not raw_path or not opportunities:
        return
    path = Path(raw_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    episode_ids = getattr(owner, "_episode_ids")
    states = getattr(owner, "_states")
    episode_length_buf = getattr(owner, "episode_length_buf")
    feature_fn = getattr(owner, "_features")
    with path.open("a", encoding="utf-8") as handle:
        for opportunity in opportunities:
            env_id = int(opportunity.env_id)
            state = states[env_id]
            payload = {
                "schema": "hybrid_packet_privileged_evidence_v1",
                "task": str(getattr(owner, "task", "")),
                "env_id": env_id,
                "episode_id": int(episode_ids[env_id]),
                "stage_id": int(state.stage),
                "candidate_epoch": int(state.candidate_epoch),
                "request_step": int(getattr(owner, "_global_step")),
                "episode_step": int(episode_length_buf[env_id].item()),
                "opportunity_type": opportunity.opportunity_type,
                "operational_gt_complete": bool(opportunity.gt_complete),
                "privileged_features": _jsonable_privileged(feature_fn(env_id)),
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def synthetic_gate_enabled(owner: object) -> bool:
    return getattr(owner, "gate_mode", None) == "synthetic"


def select_synthetic_gate_opportunities(
    owner: object,
    candidate_due: Iterable[int],
) -> list[SyntheticGateOpportunity]:
    """Return GT-labelled opportunities for the synthetic gate.

    In ``candidate_retry`` mode this degenerates to the original candidate
    locations. In ``hybrid_k64`` mode every candidate is still queried, and
    additional fixed-interval negative shadow opportunities are inserted. A
    false accept on those negative opportunities is the controlled FAR event.
    """

    candidate_set = {int(env_id) for env_id in candidate_due}
    opportunities: dict[int, SyntheticGateOpportunity] = {
        env_id: SyntheticGateOpportunity(
            env_id=env_id,
            gt_complete=True,
            opportunity_type="candidate_gt_complete",
        )
        for env_id in candidate_set
    }
    mode = getattr(owner, "transition_query_mode", DEFAULT_QUERY_MODE)
    if mode != "hybrid_k64":
        return [opportunities[env_id] for env_id in sorted(opportunities)]

    interval = int(getattr(owner, "transition_query_interval_steps"))
    min_gap_steps = int(getattr(owner, "transition_min_gap_steps", DEFAULT_MIN_GAP_STEPS))
    last = getattr(owner, "_last_gate_query_step")
    last_transition = getattr(owner, "_last_stage_transition_step", None)
    episode_length_buf = getattr(owner, "episode_length_buf")
    states = getattr(owner, "_states")
    active_stage_count = _active_stage_count(owner)
    for env_id, state in enumerate(states):
        if int(state.stage) >= active_stage_count:
            continue
        current_step = int(episode_length_buf[env_id].item())
        if min_gap_steps > 0 and last_transition is not None:
            since_transition = current_step - int(last_transition[env_id])
            if since_transition < min_gap_steps:
                continue
        elapsed = current_step - int(last[env_id])
        if elapsed < interval:
            continue
        if env_id in opportunities:
            opportunities[env_id] = SyntheticGateOpportunity(
                env_id=env_id,
                gt_complete=True,
                opportunity_type="candidate_and_periodic_gt_complete",
            )
        else:
            opportunities[env_id] = SyntheticGateOpportunity(
                env_id=env_id,
                gt_complete=False,
                opportunity_type="periodic_gt_incomplete",
            )
    return [opportunities[env_id] for env_id in sorted(opportunities)]


def mark_transition_queries(owner: object, due: Iterable[int]) -> None:
    if getattr(owner, "transition_query_mode", DEFAULT_QUERY_MODE) not in {
        "fixed_frequency",
        "hybrid_k64",
    }:
        return
    last = getattr(owner, "_last_gate_query_step", None)
    if last is None:
        return
    episode_length_buf = getattr(owner, "episode_length_buf")
    for env_id in due:
        last[int(env_id)] = int(episode_length_buf[int(env_id)].item())


def bypass_rule_candidate(owner: object, env_id: int | None = None) -> bool:
    if getattr(owner, "gate_mode", None) != "qwen":
        return False
    mode = getattr(owner, "transition_query_mode", DEFAULT_QUERY_MODE)
    if mode == "fixed_frequency":
        return True
    if mode == "hybrid_k64" and env_id is not None:
        candidates = getattr(owner, "_hybrid_qwen_candidate_due", frozenset())
        return int(env_id) not in candidates
    return False

def decide_qwen_with_repeated_accepts(
    owner: object, candidates: Sequence[object]
) -> tuple[list[str], list[bool], int]:
    """Query Qwen one or more times and require all repeats to accept.

    The repeated-query mode is intended for fixed-frequency ablations where the
    visual gate is queried without a simulator candidate filter. It reduces
    false accepts at a due event without delaying the policy by additional
    environment intervals.
    """

    if not candidates:
        return [], [], 0
    qwen_gate = getattr(owner, "qwen_gate", None)
    if qwen_gate is None:
        raise RuntimeError("Qwen gate is missing in qwen mode")
    label_hook = getattr(qwen_gate, "set_operational_labels", None)
    if callable(label_hook):
        candidate_due = getattr(owner, "_hybrid_qwen_candidate_due", frozenset())
        label_hook({int(candidate.env_id): int(candidate.env_id) in candidate_due for candidate in candidates})
    repeats = int(
        getattr(owner, "transition_consecutive_accepts", DEFAULT_CONSECUTIVE_ACCEPTS)
    )
    if repeats < 1:
        raise ValueError("transition_consecutive_accepts must be positive")
    batches = [qwen_gate.decide_batch(candidates) for _ in range(repeats)]
    predictions: list[str] = []
    decisions: list[bool] = []
    for candidate_index in range(len(candidates)):
        replies = [batch[candidate_index] for batch in batches]
        accepted = all(bool(reply.accepted) for reply in replies)
        decisions.append(accepted)
        if accepted:
            predictions.append("success")
        elif any(str(reply.prediction) == "failure" for reply in replies):
            predictions.append("failure")
        elif any(str(reply.prediction) == "unknown" for reply in replies):
            predictions.append("unknown")
        else:
            predictions.append("failure")
    return predictions, decisions, len(candidates) * repeats


def decide_synthetic_gate(
    owner: object,
    opportunities: Sequence[SyntheticGateOpportunity],
) -> list[SyntheticGateDecision]:
    if not synthetic_gate_enabled(owner):
        raise RuntimeError("synthetic gate decisions requested outside synthetic mode")
    rng = getattr(owner, "_synthetic_gate_rng", None)
    if rng is None:
        raise RuntimeError("synthetic gate RNG is not configured")
    far = float(getattr(owner, "synthetic_gate_far_alpha", 0.0))
    frr = float(getattr(owner, "synthetic_gate_frr_beta", 0.0))
    stats = getattr(owner, "_synthetic_gate_stats", None)
    decisions: list[SyntheticGateDecision] = []
    for opportunity in opportunities:
        draw = float(rng.random())
        if opportunity.gt_complete:
            accepted = draw >= frr
            false_accept = False
            false_reject = not accepted
        else:
            accepted = draw < far
            false_accept = accepted
            false_reject = False
        decision = SyntheticGateDecision(
            env_id=int(opportunity.env_id),
            gt_complete=bool(opportunity.gt_complete),
            opportunity_type=str(opportunity.opportunity_type),
            prediction="success" if accepted else "failure",
            accepted=bool(accepted),
            false_accept=bool(false_accept),
            false_reject=bool(false_reject),
            draw=draw,
        )
        decisions.append(decision)
        if isinstance(stats, dict):
            stats["opportunities"] += 1
            stats["gt_complete" if opportunity.gt_complete else "gt_incomplete"] += 1
            stats["accepts" if accepted else "rejects"] += 1
            if false_accept:
                stats["false_accepts"] += 1
            if false_reject:
                stats["false_rejects"] += 1
            if opportunity.gt_complete and accepted:
                stats["true_accepts"] += 1
            if (not opportunity.gt_complete) and (not accepted):
                stats["true_rejects"] += 1
    return decisions


def record_synthetic_gate_outcome(
    owner: object,
    decision: SyntheticGateDecision,
    *,
    old_stage: int,
    new_stage: int,
    transitioned: bool,
) -> None:
    stats = getattr(owner, "_synthetic_gate_stats", None)
    if isinstance(stats, dict) and transitioned:
        stats["transitions"] += 1
    handle = getattr(owner, "_synthetic_gate_log_handle", None)
    if handle is None:
        return
    episode_ids = getattr(owner, "_episode_ids")
    episode_length_buf = getattr(owner, "episode_length_buf")
    payload = {
        "phase": os.environ.get("STAGE_TRANSITION_SYNTHETIC_PHASE", "train"),
        "task_name": str(getattr(owner, "task", "")),
        "train_seed": os.environ.get("STAGE_TRANSITION_SYNTHETIC_TRAIN_SEED"),
        "env_id": int(decision.env_id),
        "episode_id": int(episode_ids[int(decision.env_id)]),
        "step": int(episode_length_buf[int(decision.env_id)].item()),
        "global_step": int(getattr(owner, "_global_step", -1)),
        "stage_id": int(old_stage),
        "new_stage_id": int(new_stage),
        "opportunity_type": decision.opportunity_type,
        "gt_complete": bool(decision.gt_complete),
        "alpha": float(getattr(owner, "synthetic_gate_far_alpha", 0.0)),
        "beta": float(getattr(owner, "synthetic_gate_frr_beta", 0.0)),
        "gate_output": "accepted" if decision.accepted else "rejected",
        "false_accept": bool(decision.false_accept),
        "false_reject": bool(decision.false_reject),
        "transitioned": bool(transitioned),
        "draw": float(decision.draw),
    }
    handle.write(json.dumps(payload, sort_keys=True) + "\n")


def synthetic_gate_summary(owner: object) -> dict[str, object] | None:
    if not synthetic_gate_enabled(owner):
        return None
    stats = dict(getattr(owner, "_synthetic_gate_stats", {}))
    opportunities = max(int(stats.get("opportunities", 0)), 1)
    gt_complete = max(int(stats.get("gt_complete", 0)), 1)
    gt_incomplete = max(int(stats.get("gt_incomplete", 0)), 1)
    stats.update(
        {
            "far_observed": float(stats.get("false_accepts", 0)) / gt_incomplete,
            "frr_observed": float(stats.get("false_rejects", 0)) / gt_complete,
            "accept_rate": float(stats.get("accepts", 0)) / opportunities,
        }
    )
    return {
        "alpha": float(getattr(owner, "synthetic_gate_far_alpha", 0.0)),
        "beta": float(getattr(owner, "synthetic_gate_frr_beta", 0.0)),
        "seed": int(getattr(owner, "synthetic_gate_seed", 0)),
        "query_mode": getattr(owner, "transition_query_mode", DEFAULT_QUERY_MODE),
        "query_interval_steps": int(getattr(owner, "transition_query_interval_steps", DEFAULT_INTERVAL)),
        "log_path": getattr(owner, "synthetic_gate_log_path", None),
        "stats": stats,
    }


def close_synthetic_gate_log(owner: object) -> None:
    handle = getattr(owner, "_synthetic_gate_log_handle", None)
    if handle is None:
        return
    handle.flush()
    handle.close()
    setattr(owner, "_synthetic_gate_log_handle", None)


def record_stage_transition(owner: object, env_id: int) -> None:
    last_transition = getattr(owner, "_last_stage_transition_step", None)
    if last_transition is None:
        return
    episode_length_buf = getattr(owner, "episode_length_buf")
    last_transition[int(env_id)] = int(episode_length_buf[int(env_id)].item())


def allow_immediate_stage_transition(owner: object) -> bool:
    """Return False when safe fixed-frequency mode forbids same-step jumps."""

    return int(
        getattr(owner, "transition_min_gap_steps", DEFAULT_MIN_GAP_STEPS)
    ) <= 0


