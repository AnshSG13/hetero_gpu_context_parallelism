# Changes from Official IBM FMS Repository

## Modified Files
- `fms/distributed/strategy.py`
    Added `RingAttentionStrategy` class with separate CUDA streams for communication and compute overlap. Attempted to add `block_lens` parameter for uneven token distribution across GPUs. The block len parameter is still a work in progress.
- `fms/models/__init__.py`
    Added `"ring"` as a distributed strategy option, parses `block_lens` from kwargs.
- `fms/models/llama.py`
    Modified `LLaMABlock` and `LLaMAHeadless` to support ring attention. 

## New Files
- `fms/distributed/ring_attention.py`
    This file contains the core ring attention implementation. The ring_forward() method is called from LLaMABlock which replaces standard attention. _compute_attention_ring_pass_kv() has the core ring loop. There are two streams, default for compute and new stream for communications so communication happens asynchronously. It uses a custom triton kernel in order for the online softmax to be properly calculated.
- `fms/distributed/triton_block.py`
    Custom Triton kernel that computes block softmax statistics (z, l, m) for online softmax merging. 
- `hpml_testing/`
    Contains many files for testing and benchmarking. 

