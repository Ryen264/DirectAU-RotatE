#!/bin/bash
# Testing script to validate DirectAU-RotatE fixes
# Run after applying the critical patches

set -e

echo "=========================================="
echo "DirectAU-RotatE Fix Validation Tests"
echo "=========================================="

GPU_ID=0
DATASET="wn18rr"
MODEL="DirectAU_RotatE"

# Test 1: Disable uniformity to isolate alignment loss
echo ""
echo "[TEST 1] Alignment loss only (gamma_uni=0, gamma_neg=0)"
echo "Purpose: Verify that vector magnitude collapse is fixed"
echo "Expected: align_loss should decrease from ~1.0 to <0.1 within 5000 steps"
echo "Command:"
echo "bash run.sh train $MODEL $DATASET $GPU_ID test1 512 1024 500 6.0 0.5 0.0005 10000 8 -de --gamma_uni 0.0 --gamma_neg 0.0 --epsilon 1e-8"
echo ""
read -p "Run Test 1? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    bash run.sh train $MODEL $DATASET $GPU_ID test1 512 1024 500 6.0 0.5 0.0005 10000 8 -de --gamma_uni 0.0 --gamma_neg 0.0 --epsilon 1e-8
fi

# Test 2: Add negative sampling
echo ""
echo "[TEST 2] Add negative sampling (gamma_neg=1.0)"
echo "Purpose: Verify negative loss works with fixed vectors"
echo "Expected: MRR should start >0.3 by 10000 steps"
echo "Command:"
echo "bash run.sh train $MODEL $DATASET $GPU_ID test2 512 1024 500 6.0 0.5 0.0005 10000 8 -de --gamma_uni 0.0 --gamma_neg 1.0 --epsilon 1e-8"
echo ""
read -p "Run Test 2? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    bash run.sh train $MODEL $DATASET $GPU_ID test2 512 1024 500 6.0 0.5 0.0005 10000 8 -de --gamma_uni 0.0 --gamma_neg 1.0 --epsilon 1e-8
fi

# Test 3: Add uniformity with low weight
echo ""
echo "[TEST 3] Full training with balanced losses (gamma_uni=0.1)"
echo "Purpose: Verify uniformity loss no longer dominates"
echo "Expected: MRR >0.4 by 20000 steps, losses more balanced"
echo "Command:"
echo "bash run.sh train $MODEL $DATASET $GPU_ID test3 512 1024 500 6.0 0.5 0.0005 20000 8 -de --gamma_uni 0.1 --gamma_neg 1.0 --epsilon 1e-8"
echo ""
read -p "Run Test 3? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    bash run.sh train $MODEL $DATASET $GPU_ID test3 512 1024 500 6.0 0.5 0.0005 20000 8 -de --gamma_uni 0.1 --gamma_neg 1.0 --epsilon 1e-8
fi

# Test 4: Full training matching best_config.sh
echo ""
echo "[TEST 4] Full production training"
echo "Purpose: Verify model converges to competitive baseline"
echo "Expected: MRR >0.48 by end of training (wn18rr baseline ~0.48)"
echo "Command:"
echo "bash run.sh train $MODEL $DATASET $GPU_ID test4 512 1024 500 6.0 0.5 0.0005 80000 8 -de --gamma_uni 0.1 --gamma_neg 1.0 --epsilon 1e-8"
echo ""
read -p "Run Test 4? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    bash run.sh train $MODEL $DATASET $GPU_ID test4 512 1024 500 6.0 0.5 0.0005 80000 8 -de --gamma_uni 0.1 --gamma_neg 1.0 --epsilon 1e-8
fi

echo ""
echo "=========================================="
echo "Testing complete!"
echo "=========================================="
echo ""
echo "Results Interpretation:"
echo "✓ Test 1: align_loss decreasing = Fix 1 (mask normalization) works"
echo "✓ Test 2: MRR improving = negative sampling converges"
echo "✓ Test 3: Balanced losses = Fix 2 (uniformity stability) works"
echo "✓ Test 4: MRR >0.48 = all fixes working together"
echo ""
echo "If any test fails:"
echo "1. Check logs in models/ directory"
echo "2. Compare train/valid MRR ratio (should be close)"
echo "3. Verify no GPU OOM errors"
