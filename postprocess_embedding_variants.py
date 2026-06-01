from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


VARIANTS = {
    "default_embedding": "configs/llm/llm_config.json",
}


def copy_run_for_variant(source: Path, target: Path, overwrite: bool) -> None:
    if target.exists():
        if not overwrite:
            raise SystemExit(f"target exists, pass --overwrite to replace: {target}")
        shutil.rmtree(target)

    def ignore(_dir: str, names: list[str]) -> set[str]:
        return {
            name
            for name in names
            if name in {"embeddings", "clusters", "clusters_all.json", "embeddings_all.npy"}
        }

    shutil.copytree(source, target, ignore=ignore)
    (target / "embeddings").mkdir(exist_ok=True)
    (target / "clusters").mkdir(exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run post-processing with both embedding_large and embedding_small configs."
    )
    parser.add_argument("run_dir", type=Path, help="Completed run directory with predictions/*.jsonl")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing variant directories")
    parser.add_argument("--python", default=sys.executable, help="Python executable for job_sim.py")
    args = parser.parse_args()

    source = args.run_dir.resolve()
    if not source.exists():
        raise SystemExit(f"run directory not found: {source}")
    if not (source / "predictions").exists():
        raise SystemExit(f"predictions directory not found: {source / 'predictions'}")

    repo_root = Path(__file__).resolve().parent
    for variant_name, config_name in VARIANTS.items():
        config_path = repo_root / config_name
        if not config_path.exists():
            raise SystemExit(f"missing config: {config_path}")

        target = source.parent / f"{source.name}_{variant_name}"
        copy_run_for_variant(source, target, args.overwrite)

        env = os.environ.copy()
        env["LLM_CONFIG_PATH"] = str(config_path)
        env.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")

        print(f"=== postprocess {variant_name} -> {target} ===", flush=True)
        subprocess.run(
            [args.python, "job_sim.py", "--resume", str(target)],
            cwd=repo_root,
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
