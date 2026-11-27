from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import PipelineOptions, process_dataset, write_private_outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Private, auditable preprocessing of BD/NBD brain CT PDF reports (no ML)."
    )
    parser.add_argument("--base-dir", type=Path, required=True,
                        help="Parent folder containing BD/ and NBD/ PDF subfolders")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Local PRIVATE output folder (never commit to GitHub)")
    parser.add_argument("--private-signatures", type=Path,
                        help="Optional local UTF-8 physician footer markers (one per line)")
    parser.add_argument("--expect-bd", type=int,
                        help="Verify expected BD PDF count (e.g., 94)")
    parser.add_argument("--expect-nbd", type=int,
                        help="Verify expected NBD PDF count (e.g., 67)")
    args = parser.parse_args()
    opts = PipelineOptions(args.base_dir, args.out_dir, args.private_signatures)
    df = process_dataset(opts)
    summary = write_private_outputs(df, args.out_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.expect_bd is not None and summary["N_BD"] != args.expect_bd:
        raise SystemExit("BD count mismatch; inspect source folders")
    if args.expect_nbd is not None and summary["N_NBD"] != args.expect_nbd:
        raise SystemExit("NBD count mismatch; inspect source folders")
    if summary["empty_clinical_texts"] or summary["admin_residue_detected"]:
        raise SystemExit("QC FAILED: blank narratives or administrative residues; check PRIVATE audit")
    if summary["short_reports"] or summary["negation_unmatched_audit_only"]:
        print("QC REVIEW: short texts / unmatched lexical negation cues need clinical audit.")
    print("PREPROCESSING COMPLETE; TF-IDF was NOT fitted and NO ML was run.")
