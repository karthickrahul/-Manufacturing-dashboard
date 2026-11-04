"""p-chart helpers (attribute data: part flagged / not flagged), shared by simulation and dashboard."""
import numpy as np


def p_chart(flag_counts, sizes, k_baseline):
    """Control limits from the first k_baseline batches (Phase I), applied to every batch.

    Returns dict with p, centre line, ucl, lcl arrays and boolean alarm arrays.
    Alarm rules: (1) one point above UCL; (2) eight consecutive points above the centre line.
    Only batches after the baseline period can alarm.
    """
    flag_counts = np.asarray(flag_counts, float)
    sizes = np.asarray(sizes, float)
    p = flag_counts / sizes
    pbar = flag_counts[:k_baseline].sum() / sizes[:k_baseline].sum()
    sigma = np.sqrt(pbar * (1 - pbar) / sizes)
    ucl = pbar + 3 * sigma
    lcl = np.clip(pbar - 3 * sigma, 0, None)
    rule1 = p > ucl
    above = (p > pbar).astype(int)
    run8 = np.convolve(above, np.ones(8, dtype=int), mode="full")[: len(p)] == 8
    alarm = rule1 | run8
    alarm[:k_baseline] = False
    return {"p": p, "pbar": pbar, "ucl": ucl, "lcl": lcl, "rule1": rule1, "rule2": run8, "alarm": alarm}
