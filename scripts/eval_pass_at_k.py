"""
Offline pass@k evaluation for generated responses.

Input: a parquet file produced by verl.trainer.main_generation
       (contains original dataset columns + a 'responses' column
        holding a list of decoded strings per problem).

Usage:
    python scripts/eval_pass_at_k.py \
        --input  path/to/generated.parquet \
        --ks     1 5 10 \
        --workers 8 \
        --timeout 120 \
        --output results.json   # optional

The pass@k metric uses the unbiased estimator from Chen et al. (HumanEval):
    pass@k = 1 - C(n-c, k) / C(n, k)
where n = samples per problem, c = number of correct samples.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from functools import partial

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Pass@k estimator
# ---------------------------------------------------------------------------

def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021)."""
    if n < k:
        return float("nan")
    if n - c < k:
        return 1.0
    # Use log-space product for numerical stability
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_one(args, per_response_timeout: float) -> float:
    """Score a single response. Returns the raw reward score; > 0.0 is treated as a pass."""
    data_source, solution_str, ground_truth, constraints_list, extra_info = args
    from verl.utils.reward_score import _default_compute_score
    try:
        score = _default_compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            constraints_list=constraints_list,
            extra_info=extra_info,
        )
        return float(score)
    except Exception as e:
        return 0.0


def score_problem(row: pd.Series, ks: list[int], per_response_timeout: float, executor: ThreadPoolExecutor) -> dict:
    """Score all responses for one problem and return pass@k results."""
    responses = row["responses"]
    data_source = row.get("data_source", "")
    reward_model = row.get("reward_model", {})
    ground_truth = reward_model.get("ground_truth", "") if isinstance(reward_model, dict) else ""
    constraints_list = row.get("disallowed_constraint", []) or []
    extra_info = row.get("extra_info", None)

    n = len(responses)
    args_list = [
        (data_source, resp, ground_truth, constraints_list, extra_info)
        for resp in responses
    ]

    # Submit all responses for this problem in parallel
    futures = {
        executor.submit(_score_one, args, per_response_timeout): i
        for i, args in enumerate(args_list)
    }

    scores = [0.0] * n
    for future in as_completed(futures, timeout=per_response_timeout * n + 30):
        idx = futures[future]
        try:
            scores[idx] = future.result(timeout=1)
        except Exception:
            scores[idx] = 0.0

    c = int(sum(s > 0.0 for s in scores))
    result = {"n": n, "c": c, "scores": scores}
    for k in ks:
        result[f"pass@{k}"] = estimate_pass_at_k(n, c, k)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute pass@k on a generated parquet file.")
    parser.add_argument("--input", required=True, help="Path to generated parquet (with 'responses' column).")
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10], help="k values for pass@k.")
    parser.add_argument("--workers", type=int, default=8, help="ThreadPoolExecutor max_workers.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-response execution timeout (seconds).")
    parser.add_argument("--output", default=None, help="Optional path to save JSON results.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N problems (for debugging).")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    df = pd.read_parquet(args.input)
    if args.limit:
        df = df.head(args.limit)

    assert "responses" in df.columns, (
        "Parquet must have a 'responses' column. "
        "Run verl.trainer.main_generation first to produce it."
    )

    n_problems = len(df)
    n_samples = len(df["responses"].iloc[0])
    print(f"Problems: {n_problems}  |  Samples per problem: {n_samples}  |  Workers: {args.workers}")
    print(f"Evaluating pass@{args.ks} ...\n")

    # Validate k values
    valid_ks = [k for k in args.ks if k <= n_samples]
    skipped_ks = [k for k in args.ks if k > n_samples]
    if skipped_ks:
        print(f"Warning: k={skipped_ks} skipped because n_samples={n_samples} < k.\n")

    all_results = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for i, (_, row) in enumerate(df.iterrows()):
            result = score_problem(row, valid_ks, args.timeout, executor)
            all_results.append(result)
            if (i + 1) % max(1, n_problems // 10) == 0 or (i + 1) == n_problems:
                elapsed = time.time() - t0
                correct_so_far = sum(r["c"] > 0 for r in all_results)
                print(
                    f"  [{i+1}/{n_problems}] elapsed={elapsed:.1f}s  "
                    f"best-of-n correct so far: {correct_so_far}/{i+1}"
                )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s\n")

    # Aggregate
    summary = {}
    for k in valid_ks:
        values = [r[f"pass@{k}"] for r in all_results if not np.isnan(r[f"pass@{k}"])]
        summary[f"pass@{k}"] = float(np.mean(values)) if values else float("nan")

    # Also report best-of-n accuracy (any correct)
    bon = sum(r["c"] > 0 for r in all_results) / n_problems
    summary["best_of_n_accuracy"] = bon
    summary["n_problems"] = n_problems
    summary["n_samples_per_problem"] = n_samples

    print("=" * 40)
    print("Results")
    print("=" * 40)
    for k in valid_ks:
        print(f"  pass@{k:>3d} = {summary[f'pass@{k}']:.4f}")
    print(f"  best-of-{n_samples} accuracy = {bon:.4f}")
    print("=" * 40)

    if args.output:
        payload = {"summary": summary, "per_problem": all_results}
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
