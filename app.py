import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

expected_profit, ci_low, ci_high = sim.confidence_interval(profits)
stockout_prob = (demand > Q).mean()

cf = sim.critical_fractile(price, cost, salvage)
best_q = sim.implied_q(dist, params, cf)

# ---- results ----
st.subheader("Results")
st.write(f"Estimated expected profit: **${expected_profit:,.2f}** "
         f"(95% CI: ${ci_low:,.2f} to ${ci_high:,.2f})")
st.caption("The 95% confidence interval is the range the true expected profit most likely falls in. "
           "If you ran this simulation over and over, about 95% of intervals built this way would contain it. "
           "More replications = narrower interval.")
st.write(f"Probability of running out of trees (demand > Q): **{stockout_prob:.1%}**")

# show how the interval shrinks as replications go up
with st.expander("How the confidence interval narrows with more replications"):
    rep_counts = sorted(set([500, 1000, 2500, 5000, 10000, 25000, 50000, int(n_reps)]))
    rows = sim.ci_by_replications(dist, params, Q, price, cost, salvage, rep_counts, int(seed))
    ci_table = pd.DataFrame(rows)

    ci_fig = go.Figure()
    ci_fig.add_trace(go.Scatter(
        x=ci_table["replications"], y=ci_table["mean"], mode="markers",
        error_y=dict(type="data", array=ci_table["half_width"]),
        name="Expected profit ± 95% CI"))
    # highlight the number of replications the user picked
    current = ci_table[ci_table["replications"] == int(n_reps)]
    ci_fig.add_trace(go.Scatter(
        x=current["replications"], y=current["mean"], mode="markers",
        marker=dict(size=14, color="red"), name="Your replications"))
    ci_fig.update_layout(xaxis_type="log", xaxis_title="Number of replications (log scale)",
                         yaxis_title="Expected profit", height=350)
    st.plotly_chart(ci_fig)

    st.caption("The width shrinks with 1/√n, so 4x the replications cuts the interval roughly in half.")
    st.dataframe(ci_table[["replications", "mean", "low", "high", "half_width"]].round(2),
                 hide_index=True)

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

# ---- find the optimal Q ----
st.subheader("Find the optimal Q")
st.caption("Expected profit for a range of order quantities, all using the same simulated demand.")

q_values = sim.q_range(demand)
curve, curve_low, curve_high = sim.expected_profit_curve(q_values, demand, price, cost, salvage)
best_index = curve.argmax()
sim_best_q = q_values[best_index]
sim_best_profit = curve[best_index]

st.write(f"Best Q from the simulation: **{sim_best_q}** trees, "
         f"expected profit **${sim_best_profit:,.2f}** "
         f"(95% CI: ${curve_low[best_index]:,.2f} to ${curve_high[best_index]:,.2f})")

q_fig = go.Figure()
# shaded 95% CI band: draw the upper edge, then fill down to the lower edge
q_fig.add_trace(go.Scatter(x=q_values, y=curve_high, mode="lines", line=dict(width=0),
                           showlegend=False, hoverinfo="skip"))
q_fig.add_trace(go.Scatter(x=q_values, y=curve_low, mode="lines", line=dict(width=0),
                           fill="tonexty", fillcolor="rgba(99, 110, 250, 0.2)",
                           name="95% CI"))
q_fig.add_trace(go.Scatter(x=q_values, y=curve, mode="lines", name="Expected profit",
                           line=dict(color="rgb(99, 110, 250)")))
q_fig.add_trace(go.Scatter(x=[sim_best_q], y=[sim_best_profit], mode="markers",
                           marker=dict(size=12, color="green"), name="Best Q (simulated)"))
q_fig.add_vline(x=best_q, line_color="red", line_dash="dash",
                annotation_text="Critical fractile Q")
q_fig.update_layout(xaxis_title="Trees ordered (Q)", yaxis_title="Expected profit")
st.plotly_chart(q_fig)
