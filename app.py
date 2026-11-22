import numpy as np
import plotly.express as px
import streamlit as st

import sim

st.title("Christmas Tree Order Simulator")
st.write("How many trees should the lot order before the season starts? "
         "Any trees left after Christmas get sold off cheap for mulch.")

# ---- sidebar inputs ----
st.sidebar.header("Costs and prices (per tree)")
cost = st.sidebar.number_input("Cost from the tree farm", min_value=0.0, value=2.0, step=0.1)
price = st.sidebar.number_input("Selling price", min_value=0.0, value=5.0, step=0.1)
salvage = st.sidebar.number_input("Salvage value (leftover trees)", min_value=0.0, value=0.5, step=0.1)
Q = st.sidebar.number_input("Trees to order (Q)", min_value=0, value=100, step=1)

st.sidebar.header("Tree demand")
dist = st.sidebar.selectbox("Distribution", ["Normal", "Uniform", "Poisson", "Triangular"])
if dist == "Normal":
    mean = st.sidebar.number_input("Mean", value=100.0)
    sd = st.sidebar.number_input("Standard deviation", min_value=0.0, value=25.0)
    params = {"mean": mean, "sd": sd}
elif dist == "Uniform":
    low = st.sidebar.number_input("Low", min_value=0.0, value=50.0)
    high = st.sidebar.number_input("High", min_value=0.0, value=150.0)
    params = {"low": low, "high": high}
elif dist == "Poisson":
    mean = st.sidebar.number_input("Mean", min_value=0.1, value=100.0)
    params = {"mean": mean}
elif dist == "Triangular":
    low = st.sidebar.number_input("Low", min_value=0.0, value=50.0)
    mode = st.sidebar.number_input("Most likely (mode)", min_value=0.0, value=90.0)
    high = st.sidebar.number_input("High", min_value=0.0, value=150.0)
    params = {"low": low, "mode": mode, "high": high}

st.sidebar.header("Simulation")
n_reps = st.sidebar.number_input("Number of replications", min_value=100, value=5000, step=500)
seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

# ---- input checks ----
if salvage >= cost:
    st.warning("Salvage value has to be less than unit cost, otherwise you'd just order an infinite amount.")
    st.stop()

if dist == "Uniform" and high <= low:
    st.warning("High needs to be bigger than Low.")
    st.stop()

if dist == "Triangular":
    if not (low <= mode <= high):
        st.warning("For Triangular, the numbers need to go Low ≤ Mode ≤ High.")
        st.stop()
    if low == high:
        st.warning("Low and High can't be the same number.")
        st.stop()

# ---- run the simulation ----
rng = np.random.default_rng(int(seed))
demand = sim.sample_demand(dist, params, int(n_reps), rng)
profits = sim.profit(Q, demand, price, cost, salvage)

expected_profit = profits.mean()
stockout_prob = (demand > Q).mean()

cf = sim.critical_fractile(price, cost, salvage)
best_q = sim.implied_q(dist, params, cf)

# ---- results ----
st.subheader("Results")
st.write(f"Estimated expected profit: **${expected_profit:,.2f}**")
st.write(f"Probability of running out of trees (demand > Q): **{stockout_prob:.1%}**")

st.subheader("Critical fractile")
st.write(f"Critical fractile Cu / (Cu + Co): **{cf:.1%}**")
st.write(f"Implied number of trees to order: **{best_q:.1f}** (you picked {Q})")

st.subheader("Simulated tree demand")
st.caption("This is the demand distribution you picked, with your Q as the red line.")
demand_fig = px.histogram(x=demand, nbins=40, labels={"x": "Trees demanded"})
demand_fig.add_vline(x=Q, line_color="red", line_dash="dash")
demand_fig.update_layout(yaxis_title="Count", height=300)
st.plotly_chart(demand_fig)

st.subheader("Profit histogram")
fig = px.histogram(x=profits, nbins=50, labels={"x": "Profit"})
fig.update_layout(yaxis_title="Count")
st.plotly_chart(fig)
