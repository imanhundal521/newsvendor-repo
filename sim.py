# simulation logic for the newsvendor problem (christmas tree lot version)
# keeping this separate from the streamlit app so it can be tested on its own

import numpy as np
from statistics import NormalDist


def sample_demand(dist, params, n, rng):
    # draw n random demand values from the chosen distribution
    if dist == "Normal":
        demand = rng.normal(params["mean"], params["sd"], n)
    elif dist == "Uniform":
        demand = rng.uniform(params["low"], params["high"], n)
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
    else:
        raise ValueError("unknown distribution: " + dist)
