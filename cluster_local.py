"""
本地 Embedding 脚本（sentence-transformers + 多进程）

从指定 run 目录读取 predictions/*.jsonl，用本地 Embedding 模型生成向量，
保存到 embeddings/{q}.npy。之后用 job_sim.py --resume 即可自动跳过 embedding。

用法:
    # 4 GPU 并行
    python cluster_local.py output/runs/run_20260322_133022 --gpus 0,1,2,3

    # 8 GPU 全开
    python cluster_local.py output/runs/run_20260322_133022 --gpus 0,1,2,3,4,5,6,7
"""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Process, Queue
from pathlib import Path

import numpy as np
import torch

EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    str(Path(__file__).parent / "Qwen" / "Qwen3-Embedding-8B"),
)
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "512"))


def _embed_worker(gpu_id: int, task_queue: Queue, result_queue: Queue, model_path: str, batch_size: int):
    """持久 worker：加载模型后循环处理任务。"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_path, device="cuda:0", trust_remote_code=True)
    print(f"  [GPU {gpu_id}] 模型加载完成", flush=True)

    while True:
        task = task_queue.get()
        if task is None:  # 结束信号
            break

        task_id, texts = task
        all_embs = []
        n_batches = (len(texts) + batch_size - 1) // batch_size
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            with torch.no_grad():
                embs = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
            all_embs.append(embs)
            if (len(all_embs) % 10 == 0 or len(all_embs) == n_batches):
                print(f"    [GPU {gpu_id}] {task_id}: {len(all_embs)}/{n_batches} 批", flush=True)

        result = np.vstack(all_embs).astype(np.float32)
        result_queue.put((task_id, result))


def main():
    parser = argparse.ArgumentParser(description="本地 Embedding")
    parser.add_argument("run_dir", type=Path, help="run 目录路径")
    parser.add_argument("--model", default=EMBED_MODEL, help="Embedding 模型路径")
    parser.add_argument("--gpus", default="0,1,2,3", help="GPU IDs, 逗号分隔")
    parser.add_argument("--resume", action="store_true", help="跳过已有 .npy 的 quarter")
    parser.add_argument("--batch-size", type=int, default=EMBED_BATCH_SIZE)
    args = parser.parse_args()

    run_dir = args.run_dir
    batch_size = args.batch_size
    if not run_dir.exists():
        raise SystemExit(f"run 目录不存在: {run_dir}")

    gpu_ids = [int(x) for x in args.gpus.split(",")]

    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        quarter_seq = meta["quarters"]
    else:
        pred_files = sorted(run_dir.glob("predictions/*.jsonl"))
        quarter_seq = [f.stem for f in pred_files]

    (run_dir / "embeddings").mkdir(exist_ok=True)

    print(f"=== 本地 Embedding ===")
    print(f"  Run: {run_dir}")
    print(f"  模型: {args.model}")
    print(f"  GPUs: {gpu_ids}")
    print(f"  半年期: {len(quarter_seq)} 个")

    # 启动持久 worker 进程
    task_queue = Queue()
    result_queue = Queue()
    procs = []
    for gid in gpu_ids:
        p = Process(target=_embed_worker, args=(gid, task_queue, result_queue, args.model, batch_size))
        p.start()
        procs.append(p)

    print(f"\n启动 {len(gpu_ids)} 个 worker 进程，加载模型中...")
    time.sleep(3)

    t0_all = time.time()
    done = 0
    skipped = 0
    pending_tasks = {}

    for q_label in quarter_seq:
        pred_path = run_dir / "predictions" / f"{q_label}.jsonl"
        emb_path = run_dir / "embeddings" / f"{q_label}.npy"

        if not pred_path.exists():
            print(f"  {q_label}: predictions 不存在, 跳过")
            continue

        if args.resume and emb_path.exists():
            print(f"  {q_label}: embedding 已存在, 跳过")
            skipped += 1
            continue

        texts = []
        with open(pred_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                major_name = d.get("major_name") or d.get("job_title") or ""
                rationale = d.get("rationale") or d.get("one_line_rationale") or d.get("description") or ""
                texts.append(f"{major_name}: {rationale}".strip())

        if not texts:
            print(f"  {q_label}: 无预测数据, 跳过")
            continue

        # 分配到多个 GPU
        n_gpus = len(gpu_ids)
        chunk_size = (len(texts) + n_gpus - 1) // n_gpus
        chunks = [texts[i * chunk_size : (i + 1) * chunk_size] for i in range(n_gpus)]

        print(f"\n[{q_label}] {len(texts)} 条文本, {n_gpus} GPU 并行: {[len(c) for c in chunks if c]}")
        t0 = time.time()

        task_ids = []
        for i, chunk in enumerate(chunks):
            if chunk:
                task_id = f"{q_label}_{i}"
                task_queue.put((task_id, chunk))
                task_ids.append(task_id)

        # 收集结果
        results = {}
        for _ in task_ids:
            task_id, emb = result_queue.get()
            results[task_id] = emb

        embeddings = np.vstack([results[tid] for tid in task_ids])
        np.save(str(emb_path), embeddings)
        elapsed = time.time() - t0
        done += 1
        print(f"  → shape={embeddings.shape}, {elapsed:.1f}s")

    # 发送结束信号
    for _ in gpu_ids:
        task_queue.put(None)

    for p in procs:
        p.join()

    total = time.time() - t0_all
    print(f"\n=== 完成 === {done} 个新生成, {skipped} 个跳过, 总耗时 {total:.1f}s")
    print(f"  现在可以运行: python job_sim.py --resume {run_dir}")


if __name__ == "__main__":
    main()
