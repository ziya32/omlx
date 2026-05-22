# SPDX-License-Identifier: Apache-2.0
"""Standalone A/B harness: time evaluator.run() for the sliding-window change.

Loads a model once (outside the timed region), warms it up to absorb JIT/Metal
compilation, then times a single evaluator.run() over a fixed deterministic
sample. Generation is greedy (temperature=0 is forced inside run), so the two
code versions generate the same completions — only the scheduling differs.

Run the NEW (working-tree) code first, then `git stash` and run again for OLD,
then `git stash pop`. The second run gets any warm SSD/OS cache, so a NEW-wins
result is conservative.
"""
import argparse
import asyncio
import time

from omlx.engine_pool import EnginePool
from omlx.eval import BENCHMARKS


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="?", help="free-form tag for the printout")
    ap.add_argument("--model", required=True)
    ap.add_argument("--bench", default="humaneval")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--model-dir", default="/Volumes/SD-1TB/hot-models")
    args = ap.parse_args()

    pool = EnginePool(max_model_memory=None)
    pool.discover_models(args.model_dir)
    ids = pool.get_model_ids()
    if args.model not in ids:
        raise SystemExit(f"{args.model!r} not found. Available: {ids}")

    engine = await pool.get_engine(args.model, force_lm=True)
    evaluator = BENCHMARKS[args.bench]()
    items = await evaluator.load_dataset(sample_size=args.n)

    # Warm up (load + Metal/JIT) outside the timed region.
    await engine.chat(
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=8, temperature=0.0,
    )

    t0 = time.perf_counter()
    result = await evaluator.run(
        engine, items, batch_size=args.batch_size, enable_thinking=False,
    )
    wall = time.perf_counter() - t0

    # Per-item generation latency spread. A large max/mean means long straggler
    # tails that a barrier wastes and a sliding window can reclaim; a ratio near
    # 1 means items finish together, so barrier ≈ sliding window.
    lat = sorted(qr.time_seconds for qr in result.question_results)
    n = len(lat)
    mean = sum(lat) / n if n else 0.0
    p50 = lat[n // 2] if n else 0.0
    p90 = lat[int(n * 0.9)] if n else 0.0
    stats = (
        f"lat_s[min={lat[0]:.2f} mean={mean:.2f} p50={p50:.2f} "
        f"p90={p90:.2f} max={lat[-1]:.2f} max/mean={lat[-1]/mean:.2f}]"
        if n else "lat_s[n/a]"
    )

    print(
        f"\nAB[{args.label}] bench={args.bench} model={args.model} "
        f"n={result.total_questions} batch_size={args.batch_size} "
        f"wall_s={wall:.2f} accuracy={result.accuracy:.4f} "
        f"correct={result.correct_count} thinking_used={result.thinking_used}\n"
        f"     {stats}\n"
    )

    try:
        await pool._unload_engine(args.model)
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())
