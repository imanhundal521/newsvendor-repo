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
