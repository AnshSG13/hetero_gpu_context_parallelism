# Ring Attention Benchmark Results

## Summary Table

| Context Length | GPUs | Avg Time/Token (Ring) | Avg Time/Token (Regular) | Speedup | Outputs Match? |
|----------------|------|-----------------------|--------------------------|---------|----------------|
| 100            | 2    | ERROR                 | ERROR                    | N/A       | FAILED          |
| 500            | 2    | ERROR                 | ERROR                    | N/A       | FAILED          |
| 1000            | 2    | ERROR                 | ERROR                    | N/A       | FAILED          |
| 1200            | 2    | ERROR                 | ERROR                    | N/A       | FAILED          |

## Notes

- All benchmarks run with 30 tokens generated per test
- Model: 3-8b
- Speedup calculated as: Regular Time / Ring Time
- Correctness verified by comparing output logits between Ring and Regular attention
