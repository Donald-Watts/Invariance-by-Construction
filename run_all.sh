#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python}"

echo "============================================================"
echo " Constrained-Operation Catalogue — Full Reproduction"
echo "============================================================"
echo ""
echo "Python: $($PYTHON --version 2>&1)"
echo "Working dir: $SCRIPT_DIR"
echo ""

# Check dependencies
echo "Checking dependencies..."
$PYTHON -c "import torch, numpy, scipy, matplotlib" 2>/dev/null || {
    echo "ERROR: Missing dependencies. Run: pip install -r requirements.txt"
    exit 1
}
echo "All dependencies OK."
echo ""

TOTAL_START=$(date +%s)

# 1. e2 verification (quick sanity check)
echo "[1/5] e₂ polynomial verification..."
$PYTHON src/verify_e2.py 2>&1 | tee results/verify_e2_run.log
echo ""

# 2. Ablation study (Reynolds × depth factorial + component ablations)
echo "[2/5] Ablation study (Reynolds × depth factorial)..."
$PYTHON experiments/ablation_study.py 2>&1 | tee results/ablation_run.log
echo ""

# 3. Breadth study
echo "[3/5] Breadth study (3 tasks)..."
$PYTHON experiments/breadth_study.py 2>&1 | tee results/breadth_run.log
echo ""

# 4. Complexity analysis
echo "[4/5] Complexity analysis (FLOP scaling)..."
$PYTHON experiments/complexity_analysis.py 2>&1 | tee results/complexity_run.log
echo ""

# 5. Baseline comparison
echo "[5/5] Baseline comparison (4 models × 2 tasks)..."
$PYTHON experiments/baseline_comparison.py 2>&1 | tee results/baseline_run.log
echo ""

TOTAL_END=$(date +%s)
ELAPSED=$((TOTAL_END - TOTAL_START))

echo "============================================================"
echo " ALL EXPERIMENTS COMPLETE"
echo " Total time: ${ELAPSED}s ($((ELAPSED/60))m $((ELAPSED%60))s)"
echo "============================================================"
echo ""
echo "Output files in results/:"
echo "  - ablation_metrics.{json,csv}"
echo "  - breadth_metrics.{json,csv}"
echo "  - complexity_metrics.json"
echo "  - baseline_metrics.json"
echo "  - *.png (all plots)"
echo "  - *_run.log (execution logs)"
