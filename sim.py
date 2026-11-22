# simulation logic for the newsvendor problem (christmas tree lot version)
# keeping this separate from the streamlit app so it can be tested on its own

import math
import numpy as np
from statistics import NormalDist


def sample_demand(dist, params, n, rng):
    # draw n random demand values from the chosen distribution
    if dist == "Normal":
        demand = rng.normal(params["mean"], params["sd"], n)
    elif dist == "Uniform":
        demand = rng.uniform(params["low"], params["high"], n)
    elif dist == "Poisson":
        demand = rng.poisson(params["mean"], n)
    elif dist == "Triangular":
        demand = rng.triangular(params["low"], params["mode"], params["high"], n)
    else:
        raise ValueError("unknown distribution: " + dist)

    # demand has to be a whole number and can't be negative
    demand = np.round(demand)
    demand = np.maximum(demand, 0)
    return demand.astype(int)


def profit(Q, demand, price, cost, salvage):
    # sell whatever we can, salvage the leftovers, pay for everything we ordered
    sold = np.minimum(Q, demand)
    leftover = np.maximum(Q - demand, 0)
    return price * sold + salvage * leftover - cost * Q


def q_range(demand, max_points=500):
    # go from 0 up to a bit past the high end of demand (99.5th percentile)
    top = int(np.ceil(1.1 * np.percentile(demand, 99.5)))
    # if demand is really big, skip some Qs so this doesn't take forever
    step = max(1, top // max_points)
    return np.arange(0, top + 1, step)


def confidence_interval(values, z=1.96):
    # 95% CI for the mean: mean +/- 1.96 * (sample std / sqrt(n))
    mean = np.mean(values)
    half_width = z * np.std(values, ddof=1) / np.sqrt(len(values))
    return mean, mean - half_width, mean + half_width


def expected_profit_curve(q_values, demand, price, cost, salvage):
    # use the SAME demand sample for every Q (common random numbers)
    # so differences between Qs come from Q itself and not from sampling noise
    # returns the mean profit at each Q plus the low/high ends of the 95% CI
    means, lows, highs = [], [], []
    for q in q_values:
        mean, low, high = confidence_interval(profit(q, demand, price, cost, salvage))
        means.append(mean)
        lows.append(low)
        highs.append(high)
    return np.array(means), np.array(lows), np.array(highs)


def ci_by_replications(dist, params, Q, price, cost, salvage, rep_counts, seed):
    # draw one big sample and use the first n values for each n,
    # so the only thing changing between rows is how many replications we use
    rng = np.random.default_rng(seed)
    demand = sample_demand(dist, params, max(rep_counts), rng)
    profits = profit(Q, demand, price, cost, salvage)

    rows = []
    for n in rep_counts:
        mean, low, high = confidence_interval(profits[:n])
        rows.append({"replications": n, "mean": mean, "low": low, "high": high,
                     "half_width": high - mean})
    return rows


def critical_fractile(price, cost, salvage):
    cu = price - cost    # underage cost - profit lost on each missed sale
    co = cost - salvage  # overage cost - money lost on each unsold unit
    return cu / (cu + co)


def implied_q(dist, params, fractile):
    # the order quantity where P(demand <= Q) = critical fractile
    if dist == "Normal":
        return NormalDist(params["mean"], params["sd"]).inv_cdf(fractile)
    elif dist == "Uniform":
        return params["low"] + fractile * (params["high"] - params["low"])
    elif dist == "Poisson":
        return poisson_quantile(params["mean"], fractile)
    elif dist == "Triangular":
        return triangular_quantile(params["low"], params["mode"], params["high"], fractile)
    else:
        raise ValueError("unknown distribution: " + dist)


def poisson_quantile(mean, fractile):
    # poisson is already whole numbers, so just add up probabilities
    # until we reach the fractile. doing it in logs so big means don't underflow
    k = 0
    total = 0.0
    while True:
        total += math.exp(k * math.log(mean) - mean - math.lgamma(k + 1))
        if total >= fractile:
            return k
        k += 1


def triangular_quantile(low, mode, high, fractile):
    # inverse cdf of the triangular distribution
    # the left side of the triangle covers this much of the probability
    left_share = (mode - low) / (high - low)
    if fractile < left_share:
        return low + math.sqrt(fractile * (high - low) * (mode - low))
    else:
        return high - math.sqrt((1 - fractile) * (high - low) * (high - mode))
