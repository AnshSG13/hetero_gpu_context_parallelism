#!/bin/bash

# Clear the screen for a clean demo
clear

# --- Introduction ---
echo "=========================================================="
echo "      Heterogeneous Ring Attention Demo"
echo "=========================================================="
echo
echo "Key files that implement ring attention inside ibm-fms:"
echo "----------------------------------------------------------"
echo "1. Ring Attention Implementation:"
echo "   - fms/distributed/ring_attention.py: The core algorithm."
echo "   - fms/distributed/triton_block.py: Triton-accelerated kernel."
echo
echo "2. Benchmarking Scripts:"
echo "   - hpml_testing/generate_ring_profile.py: Generate performance profile"
echo "   - hpml_testing/run_sweep.py: Run full benchmark sweep"
echo "----------------------------------------------------------"
echo
sleep 2

echo ">>> Starting benchmark sweep..."
echo
python3 hpml_testing/run_sweep.py --profile-path hpml_testing/results/ring_attention_profile.csv

echo
echo "=========================================================="
echo "Demo Finished!"
echo "=========================================================="
