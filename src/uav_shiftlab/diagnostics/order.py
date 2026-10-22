# SPDX-License-Identifier: Apache-2.0
"""Stream-order sensitivity: does a method's score, or the ranking of methods, depend on the order of sequences?"""
from itertools import combinations


def kendall_tau(a, b):
    """Kendall tau-a between two rankings given as {item: score} on the same items (higher = better)."""
    items = sorted(set(a) & set(b))
    conc = disc = 0
    for x, y in combinations(items, 2):
        s = (a[x] - a[y]) * (b[x] - b[y])
        conc += s > 0
        disc += s < 0
    n = len(items) * (len(items) - 1) / 2
    return (conc - disc) / n if n else None


def order_sensitivity(scores):
    """scores: {order: {method: value}}. Per-method spread across orders and pairwise ranking agreement."""
    orders = sorted(scores)
    methods = sorted(set.intersection(*(set(scores[o]) for o in orders)))
    per_method = {m: {"min": min(scores[o][m] for o in orders), "max": max(scores[o][m] for o in orders),
                      "spread": max(scores[o][m] for o in orders) - min(scores[o][m] for o in orders),
                      "by_order": {o: scores[o][m] for o in orders}} for m in methods}
    ranking = {o: sorted(methods, key=lambda m: -scores[o][m]) for o in orders}
    taus = {f"{x}~{y}": kendall_tau({m: scores[x][m] for m in methods}, {m: scores[y][m] for m in methods})
            for x, y in combinations(orders, 2)}
    return {"methods": methods, "orders": orders, "per_method": per_method, "ranking": ranking,
            "kendall_tau": taus, "ranking_identical": len({tuple(r) for r in ranking.values()}) == 1}
