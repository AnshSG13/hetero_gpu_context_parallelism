import math

def empirical_normalized_perf(seq_len: float, mps_pct: float) -> float:
    """
    Compute normalized performance (%) as a function of
    sequence length and MPS percentage.

    Parameters
    ----------
    seq_len : float
        Sequence length (must be > 0)
    mps_pct : float
        MPS percentage (e.g., 10, 25, 50, 100)
    Returns
    -------
    float
        Normalized performance (%)
    """
    log_seq = math.log10(seq_len)

    return (
        -662.478
        + 365.074 * log_seq
        + 0.195907 * mps_pct
        - 49.6378 * (log_seq ** 2)
        + 0.337949 * log_seq * mps_pct
        - 0.00710625 * (mps_pct ** 2)
    )
