# Changes from Official IBM FMS Repository

## Modified Files
- `fms/distributed/strategy.py`
    Added `RingAttentionStrategy` class with separate CUDA streams for communication and compute overlap. Supports heterogeneous `block_lens` for uneven token distribution across GPUs.
- `fms/models/__init__.py`
    Added `"ring"` as a distributed strategy option, parses `block_lens` from kwargs.
- `fms/models/llama.py`
    Modified `LLaMABlock` and `LLaMAHeadless` to support ring attention. Injects strategy onto blocks and handles input sharding/output gathering.

## New Files
- `fms/distributed/ring_attention.py`
    Core ring attention implementation. Uses Flash Attention for diagonal blocks (when no merging needed) and Triton kernel for off-diagonal blocks. Implements online softmax for merging attention across KV blocks.
- `fms/distributed/triton_offdiag_block.py`
    Custom Triton kernel that computes block softmax statistics (z, l, m) for online softmax merging. Handles causal masking with global position indices.
