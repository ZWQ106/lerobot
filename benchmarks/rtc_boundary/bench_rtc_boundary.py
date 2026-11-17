"""Offline chunk-boundary benchmark for Multi-Task DiT real-time chunking.

Replays every episode of a LeRobot dataset through the local RTC engine's capture history,
preprocessing and `predict_chunk` (driven tick by tick, without the engine's worker thread),
once without guidance (sync) and once with guided RTC, drawing the same seed for each chunk.
A new chunk is requested every `execution_horizon` ticks from tick 0; at a hand-over the old
chunk's actions for the first `inference_delay` ticks are already committed. Metrics are in
canonical action units (degrees for the SO-101 arm joints), averaged over the arm joints:

- seam: new chunk vs old chunk at the hand-over index (tick request + delay)
- jump: last executed old action vs first executed new action
- step: mean change between consecutive actions inside a chunk (reference)
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from types import SimpleNamespace

import numpy as np
import torch

from lerobot.configs import RTCAttentionSchedule
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.inference import ObservationSnapshot, RTCInferenceEngine
from lerobot.inference.prediction import chunk_inference_context, predict_chunk
from lerobot.policies import ExecutionMode
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.multi_task_dit.modeling_multi_task_dit import MultiTaskDiTPolicy
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.feature_utils import build_dataset_frame


def build(policy_path: str, revision: str, device: str, rtc: RTCConfig, features: dict, robot, fps: float):
    """Load the policy and processors the way the rollout does, wrapped in a local RTC engine."""
    policy = MultiTaskDiTPolicy.from_pretrained(policy_path, revision=revision)
    policy.config.device = device
    policy.config.rtc_config = rtc if rtc.enabled else None
    policy.init_rtc_processor()
    policy.to(device).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=policy_path,
        pretrained_revision=revision,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    engine = RTCInferenceEngine(
        policy=policy,
        preprocessor=pre,
        postprocessor=post,
        robot_wrapper=robot,
        rtc_config=rtc,
        dataset_features=features,
        task="",
        fps=fps,
        device=device,
    )
    mode = ExecutionMode.RTC_GUIDED if rtc.enabled else ExecutionMode.CHUNK
    return SimpleNamespace(policy=policy, post=post, engine=engine, mode=mode, model_dim=None, prev=None)


def raw_observation(item: dict, state_names: list[str], cameras: list[str]) -> dict:
    """Turn a dataset frame back into the robot observation the control thread would publish."""
    raw = {name: float(value) for name, value in zip(state_names, item["observation.state"].tolist(), strict=True)}
    for camera in cameras:
        image = item[f"observation.images.{camera}"]
        if image.dtype != torch.uint8:
            image = (image * 255).round().clamp(0, 255).to(torch.uint8)
        raw[camera] = image.permute(1, 2, 0).contiguous().numpy()
    return raw


def predict(run, task: str, seed: int, *, horizon: int, delay: int, device: str):
    """One chunk through the engine's history, preprocessing and predict_chunk."""
    engine = run.engine
    with engine._obs_lock:
        raw = engine._obs_holder["obs"]
        capture_time = engine._obs_holder["capture_time"]
        history = engine._history.frames() if engine._history is not None else None
    frame = build_dataset_frame(engine._obs_features, raw, prefix="observation")
    snapshot = ObservationSnapshot(frame, capture_time, task, 0)
    guided = run.mode is ExecutionMode.RTC_GUIDED and run.prev is not None
    with chunk_inference_context(run.mode):
        prepared = engine._prepare(snapshot, history, task, torch.device(device))
        torch.manual_seed(seed)
        started = time.perf_counter()
        chunk = predict_chunk(
            run.policy,
            run.post,
            prepared,
            spec=engine._policy_spec,
            mode=run.mode,
            canonical_action_dim=engine._canonical_action_dim,
            model_action_dim=run.model_dim,
            device=device,
            rtc_horizon=horizon if run.mode is ExecutionMode.RTC_GUIDED else 0,
            inference_delay=delay if guided else 0,
            model_continuation=run.prev.model_actions[horizon:] if guided else None,
            canonical_continuation=run.prev.canonical_actions[horizon:] if guided else None,
            relative_step=engine._relative_step,
            normalizer_step=engine._normalizer_step,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
    run.model_dim = chunk.model_actions.shape[1]
    return chunk, seconds


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "p95": float(np.percentile(array, 95)), "n": int(array.size)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="diaoweiqing/multitask-dit-so101-small")
    parser.add_argument("--policy-revision", default="0f1e798737d78b8e3cdbeea71dc501bc6f5c2d5b")
    parser.add_argument("--dataset", default="diaoweiqing/act_35_20260503_134804")
    parser.add_argument("--dataset-revision", default="e71bd9f9989afdd7c30c536a75e4e1ad9bd87e57")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--episodes", default="all", help="'all' or comma-separated episode indices")
    parser.add_argument("--execution-horizon", type=int, default=10)
    parser.add_argument("--inference-delay", type=int, default=3)
    parser.add_argument("--max-guidance-weight", type=float, default=10.0)
    parser.add_argument("--schedule", default="LINEAR")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--code-revision", default="unknown")
    parser.add_argument("--seed-scheme", choices=["per-chunk", "per-episode"], default="per-chunk",
                        help="per-chunk: fresh noise for every chunk; per-episode: one seed for every chunk of an episode")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    meta_only = LeRobotDataset(args.dataset, root=args.dataset_root, revision=args.dataset_revision, return_uint8=True)
    episodes = (
        list(range(meta_only.meta.total_episodes))
        if args.episodes == "all"
        else [int(index) for index in args.episodes.split(",")]
    )
    features = meta_only.meta.features
    fps = float(meta_only.meta.fps)
    state_names = list(features["observation.state"]["names"])
    action_names = list(features["action"]["names"])
    cameras = [key.removeprefix("observation.images.") for key in features if key.startswith("observation.images.")]
    arm = [index for index, name in enumerate(action_names) if not name.startswith("gripper")]
    robot = SimpleNamespace(
        robot_type=meta_only.meta.robot_type,
        action_features=dict.fromkeys(action_names, float),
        supports_hold=False,
        observation_time=None,
    )
    guided = RTCConfig(
        enabled=True,
        execution_horizon=args.execution_horizon,
        max_guidance_weight=args.max_guidance_weight,
        prefix_attention_schedule=RTCAttentionSchedule[args.schedule],
    )
    runs = {
        "sync": build(args.policy, args.policy_revision, args.device, RTCConfig(enabled=False), features, robot, fps),
        "rtc": build(args.policy, args.policy_revision, args.device, guided, features, robot, fps),
    }

    horizon, delay = args.execution_horizon, args.inference_delay
    seams = {name: [] for name in runs}
    jumps = {name: [] for name in runs}
    steps = {name: [] for name in runs}
    seconds = {name: [] for name in runs}
    boundaries = []
    clock = 0.0
    started = time.time()
    for done, episode in enumerate(episodes, start=1):
        dataset = LeRobotDataset(
            args.dataset,
            root=args.dataset_root,
            revision=args.dataset_revision,
            episodes=[episode],
            return_uint8=True,
        )
        for run in runs.values():
            run.policy.reset()
            run.prev = None
            with run.engine._obs_lock:
                run.engine._drop_observations()
        length = len(dataset)
        clock += 1000.0
        for tick in range(length):
            item = dataset[tick]
            raw = raw_observation(item, state_names, cameras)
            robot.observation_time = clock + tick / fps
            for run in runs.values():
                run.engine.notify_observation(raw)
            if tick % horizon:
                continue
            chunk_index = tick // horizon
            seed = 100_000 * episode + (chunk_index if args.seed_scheme == "per-chunk" else 0)
            record = {"episode": episode, "chunk": chunk_index, "tick": tick}
            for name, run in runs.items():
                chunk, spent = predict(run, item["task"], seed, horizon=horizon, delay=delay, device=args.device)
                seconds[name].append(spent)
                actions = chunk.canonical_actions.detach().float().cpu()[:, arm]
                steps[name].append(float((actions[1:] - actions[:-1]).abs().mean()))
                if run.prev is not None:
                    old = run.prev.canonical_actions.detach().float().cpu()[:, arm]
                    seam = float((actions[delay] - old[horizon + delay]).abs().mean())
                    jump = float((actions[delay] - old[horizon + delay - 1]).abs().mean())
                    seams[name].append(seam)
                    jumps[name].append(jump)
                    record[f"seam_{name}"] = seam
                    record[f"jump_{name}"] = jump
                run.prev = chunk
            if chunk_index:
                boundaries.append(record)
        print(
            f"episode {episode} ({done}/{len(episodes)}) frames {length} boundaries {len(boundaries)} "
            f"seam sync {np.mean(seams['sync']):.3f} rtc {np.mean(seams['rtc']):.3f} "
            f"elapsed {time.time() - started:.0f}s",
            flush=True,
        )

    result = {
        "description": __doc__,
        "policy": {"repo": args.policy, "revision": args.policy_revision},
        "dataset": {"repo": args.dataset, "revision": args.dataset_revision, "episodes": episodes, "fps": fps},
        "code_revision": args.code_revision,
        "config": {
            "execution_horizon": horizon,
            "inference_delay": delay,
            "max_guidance_weight": args.max_guidance_weight,
            "prefix_attention_schedule": args.schedule,
            "first_request_tick": 0,
            "seed": f"{args.seed_scheme}: torch.manual_seed(100000 * episode + (chunk_index or 0)) before each chunk, same in both runs",
            "arm_joints": [action_names[index] for index in arm],
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "python": platform.python_version(),
        },
        "summary": {
            name: {
                "seam": summarize(seams[name]),
                "jump": summarize(jumps[name]),
                "step": summarize(steps[name]),
                "seconds_per_chunk": summarize(seconds[name]),
            }
            for name in runs
        },
        "boundaries": boundaries,
    }
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=1)
    print(json.dumps(result["summary"], indent=1), flush=True)


if __name__ == "__main__":
    main()
