"""Utilities for consolidating BC policy checkpoints across environment runs.

Example
-------
In the logs structure where each environment is trained under a separate
directory such as ``logs-bc/1001_catapult_diversereward`` containing numbered
checkpoint folders (``0``, ``1`` ... ``35``), the helper below collects the
``policies`` subdirectories from all matching runs and copies them into a
single run directory.

PYTHONPATH=src python3 src/util/merge_bc_policies.py \
  --source-root logs-bc \
  --target-root logs-bc/run1009 \
  --run-prefix 1009_ \
  --iterations 5 35

>>> merge_bc_policies(
...     source_root=Path("logs-bc"),
...     target_root=Path("logs-bc/run1001"),
...     run_prefix="1001_",
...     iteration_ids=(5, 35),
... )

After running, ``logs-bc/run1001/<iter>/policies`` contains the policies from
each environment run (``worlds_l_catapult.pkl``, ``worlds_l_grasp_easy.pkl``,
...).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass
class MergeReport:
    copied_files: int
    skipped_existing: int
    missing_iterations: dict[str, list[int]]
    missing_policies: dict[str, list[int]]


def _ensure_sequence(iteration_ids: Iterable[int]) -> Sequence[int]:
    unique_ids = sorted({int(i) for i in iteration_ids})
    if not unique_ids:
        raise ValueError("`iteration_ids` may not be empty")
    return unique_ids


def merge_bc_policies(
    *,
    source_root: Path,
    target_root: Path,
    run_prefix: str,
    iteration_ids: Iterable[int],
    policy_subdir: str = "policies",
    overwrite: bool = False,
) -> MergeReport:
    """Copy per-environment BC policies into a unified run directory.

    Parameters
    ----------
    source_root:
        Path containing the per-environment run directories (e.g. ``logs-bc``).
    target_root:
        Destination directory that will mirror the numbered iteration layout
        with all policy files gathered under ``policies/``.
    run_prefix:
        Prefix used to select environment run folders (e.g. ``"1001_"``).
    iteration_ids:
        Iterable of iteration numbers to gather (e.g. ``(5, 35)`` or
        ``range(36)``).
    policy_subdir:
        Name of the directory under each iteration that stores policy files.
    overwrite:
        If ``True`` existing files in the target directory are replaced.
    """

    source_root = source_root.resolve()
    target_root = target_root.resolve()
    iteration_list = _ensure_sequence(iteration_ids)

    if not source_root.exists():
        raise FileNotFoundError(f"source_root does not exist: {source_root}")

    env_run_dirs = [
        d for d in source_root.iterdir()
        if d.is_dir() and d.name.startswith(run_prefix)
    ]
    if not env_run_dirs:
        raise FileNotFoundError(
            f"No environment runs found under {source_root} with prefix '{run_prefix}'"
        )

    copied = 0
    skipped = 0
    missing_iters: dict[str, list[int]] = {}
    missing_policies: dict[str, list[int]] = {}

    for env_dir in sorted(env_run_dirs):
        env_name = env_dir.name
        for iteration in iteration_list:
            src_iter_dir = env_dir / str(iteration)
            if not src_iter_dir.exists():
                missing_iters.setdefault(env_name, []).append(iteration)
                continue

            src_policy_dir = src_iter_dir / policy_subdir
            if not src_policy_dir.exists():
                missing_policies.setdefault(env_name, []).append(iteration)
                continue

            dst_policy_dir = target_root / str(iteration) / policy_subdir
            dst_policy_dir.mkdir(parents=True, exist_ok=True)

            for policy_file in src_policy_dir.glob("*"):
                if not policy_file.is_file():
                    continue

                dst_path = dst_policy_dir / policy_file.name
                if dst_path.exists() and not overwrite:
                    skipped += 1
                    continue

                shutil.copy2(policy_file, dst_path)
                copied += 1

    return MergeReport(
        copied_files=copied,
        skipped_existing=skipped,
        missing_iterations=missing_iters,
        missing_policies=missing_policies,
    )


if __name__ == "__main__":
    import json
    from argparse import ArgumentParser

    try:
        import tyro
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        tyro = None

    @dataclass
    class MergeConfig:
        source_root: Path = Path("logs-bc")
        target_root: Path = Path("logs-bc/run1001")
        run_prefix: str = "1001_"
        iterations: tuple[int, ...] = tuple(range(36))
        overwrite: bool = False

    if tyro is not None:
        cfg = tyro.cli(MergeConfig)
    else:
        parser = ArgumentParser(description="Merge BC policies across env runs")
        parser.add_argument("--source-root", type=Path, default=MergeConfig.source_root)
        parser.add_argument("--target-root", type=Path, default=MergeConfig.target_root)
        parser.add_argument("--run-prefix", type=str, default=MergeConfig.run_prefix)
        parser.add_argument("--iterations", type=int, nargs="+", default=None)
        parser.add_argument("--overwrite", action="store_true", default=MergeConfig.overwrite)
        args = parser.parse_args()
        iterations = tuple(args.iterations) if args.iterations is not None else MergeConfig.iterations
        cfg = MergeConfig(
            source_root=args.source_root,
            target_root=args.target_root,
            run_prefix=args.run_prefix,
            iterations=iterations,
            overwrite=args.overwrite,
        )
    report = merge_bc_policies(
        source_root=cfg.source_root,
        target_root=cfg.target_root,
        run_prefix=cfg.run_prefix,
        iteration_ids=cfg.iterations,
        overwrite=cfg.overwrite,
    )
    print(json.dumps(report.__dict__, indent=2))
