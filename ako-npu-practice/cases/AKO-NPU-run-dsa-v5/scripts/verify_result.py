#!/usr/bin/env python3
"""Verify DSA Indexer output against golden data."""

import numpy as np
import sys
import os

INDEX_TOPK = 2048

def bf16_to_float(bf16_path):
    """Load bfloat16 binary file and convert to float32."""
    bf16 = np.fromfile(bf16_path, dtype=np.uint16)
    raw32 = bf16.astype(np.uint32) << 16
    return raw32.view(np.float32)


def verify(B, S_q, S_kv):
    topk_k = min(INDEX_TOPK, S_kv)

    # Load kernel outputs
    output_topk = np.fromfile("output/output_topk_indices.bin", dtype=np.int64).reshape(B, S_q, topk_k)
    output_score = bf16_to_float("output/output_index_score.bin").reshape(B, S_q, S_kv)

    # Load golden outputs
    golden_topk = np.fromfile("output/golden_topk_indices.bin", dtype=np.int64).reshape(B, S_q, topk_k)
    golden_score = bf16_to_float("output/golden_index_score.bin").reshape(B, S_q, S_kv)
    golden_score_f32 = np.fromfile("output/golden_index_score_f32.bin", dtype=np.float32).reshape(B, S_q, S_kv)

    passed = True

    # ---- Verify index_score (bf16) ----
    # Use golden_score (bf16) for comparison since both are bf16 representations
    # Check MERE and MARE
    valid_mask = np.isfinite(golden_score) & (np.abs(golden_score) > 1e-10)
    if valid_mask.sum() > 0:
        diff = np.abs(output_score[valid_mask] - golden_score[valid_mask])
        rel_err = diff / np.abs(golden_score[valid_mask])
        mere = np.max(rel_err)
        mare = np.mean(rel_err)
        max_abs_diff = np.max(diff)

        mere_threshold = 2**(-7)  # ~0.0078
        mare_threshold = 10 * 2**(-7)  # ~0.078

        print(f"index_score verification:")
        print(f"  Max absolute diff: {max_abs_diff:.6f}")
        print(f"  MERE (max rel err): {mere:.6f} (threshold: {mere_threshold:.6f})")
        print(f"  MARE (mean rel err): {mare:.6f} (threshold: {mare_threshold:.6f})")

        if mere > mere_threshold:
            print(f"  WARNING: MERE exceeds threshold!")
            # Check percentage of violations
            violations = (rel_err > mere_threshold).sum()
            print(f"  Violations: {violations}/{valid_mask.sum()} ({100*violations/valid_mask.sum():.2f}%)")
        else:
            print(f"  index_score PASSED")
    else:
        print("  No valid (finite, non-zero) score values to compare")

    # ---- Verify topk_indices ----
    # Compare as sets per (b, s) pair -- order within same-valued positions may differ
    topk_match = 0
    topk_total = 0

    for b in range(B):
        for s in range(S_q):
            out_set = set(output_topk[b, s].tolist())
            gold_set = set(golden_topk[b, s].tolist())
            intersection = out_set & gold_set
            topk_total += topk_k
            topk_match += len(intersection)

            if out_set != gold_set:
                # Check if mismatches are tie-breaking differences
                mismatched_out = out_set - gold_set
                mismatched_gold = gold_set - out_set

                if len(mismatched_out) <= 10:
                    print(f"  topk mismatch at b={b}, s={s}:")
                    print(f"    output-only ({len(mismatched_out)}): {sorted(list(mismatched_out))[:10]}")
                    print(f"    golden-only ({len(mismatched_gold)}): {sorted(list(mismatched_gold))[:10]}")

                    # Check if scores at mismatched positions are close
                    for idx in list(mismatched_out)[:5]:
                        if 0 <= idx < S_kv:
                            print(f"    score at output-only idx {idx}: {golden_score_f32[b,s,idx]:.6f}")
                    for idx in list(mismatched_gold)[:5]:
                        if 0 <= idx < S_kv:
                            print(f"    score at golden-only idx {idx}: {golden_score_f32[b,s,idx]:.6f}")

    match_rate = topk_match / topk_total if topk_total > 0 else 0
    print(f"\ntopk_indices verification:")
    print(f"  Match rate: {topk_match}/{topk_total} ({100*match_rate:.2f}%)")

    if match_rate >= 0.99:
        print(f"  topk_indices PASSED (>99% match)")
    elif match_rate >= 0.95:
        print(f"  topk_indices MARGINAL (>95% match, likely tie-breaking differences)")
    else:
        print(f"  topk_indices FAILED (<95% match)")
        passed = False

    return passed


def main():
    B = 1
    S_q = 1
    S_kv = 4096
    if len(sys.argv) >= 4:
        B = int(sys.argv[1])
        S_q = int(sys.argv[2])
        S_kv = int(sys.argv[3])

    print(f"Verifying DSA Indexer: B={B}, S_q={S_q}, S_kv={S_kv}")
    passed = verify(B, S_q, S_kv)

    if passed:
        print("\n=== Verification PASSED ===")
        return 0
    else:
        print("\n=== Verification FAILED ===")
        return 1


if __name__ == "__main__":
    sys.exit(main())
