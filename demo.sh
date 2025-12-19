#!/bin/bash

# Clear the screen for a clean demo
clear

# --- Introduction ---
echo "=========================================================="
echo "      Heterogeneous Ring Attention W&B Demo"
echo "=========================================================="
echo
echo "Key files that implement ring attention inside ibm-fms:"
echo "----------------------------------------------------------"
echo "1. Ring Attention Implementation:"
echo "   - fms/distributed/ring_attention.py: The core algorithm."
echo "   - fms/distributed/triton_block.py: Triton-accelerated kernel."
echo
echo "2. Benchmarking Script in heterogeneous env"
echo "   - hpml_testing/simple_sweep.py"
echo "----------------------------------------------------------"
echo
sleep 2

echo ">>> Starting benchmark..."
echo
python3 hpml_testing/simple_sweep.py

echo
echo "=========================================================="
echo "Demo Finished!"
echo "=========================================================="
