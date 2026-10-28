#!/usr/bin/env bash
# Full pipeline: experiments -> pick best on VALIDATION -> error analysis -> optimisation -> tables.
set -euo pipefail
bash scripts/run_experiments.sh
BEST=$(python scripts/make_report_tables.py --print-best)
echo "Best experiment (by val segm mAP): $BEST"
mkdir -p models
cp "results/$BEST/best.pt" models/best.pt
python scripts/export_slim.py --ckpt models/best.pt --out models/best_fp16.pt
python -m src.error_analysis --exp "$BEST"
python -m src.optimize --exp "$BEST"
python scripts/make_report_tables.py
echo "Done. See results/RESULTS.md"
