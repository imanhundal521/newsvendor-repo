import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import sim

st.set_page_config(page_title="Christmas Tree Order Simulator", page_icon="🎄", layout="wide")

st.title("Christmas Tree Order Simulator")
st.caption("How many trees should the lot order before the season starts? "
           "Anything left after Christmas gets sold cheap for mulch.")

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
        st.warning("For Triangular, the numbers need to go Low <= Mode <= High.")
        st.stop()
    if low == high:
        st.warning("Low and High can't be the same number.")
        st.stop()

# ---- run the simulation ----
rng = np.random.default_rng(int(seed))
demand = sim.sample_demand(dist, params, int(n_reps), rng)
profits = sim.profit(Q, demand, price, cost, salvage)

expected_profit, ci_low, ci_high = sim.confidence_interval(profits)
stats = sim.risk_stats(Q, demand, profits)

cf = sim.critical_fractile(price, cost, salvage)
best_q = sim.implied_q(dist, params, cf)

# expected profit for a range of Qs, all with the same demand sample
q_values = sim.q_range(demand)
curve, curve_low, curve_high = sim.expected_profit_curve(q_values, demand, price, cost, salvage)
best_index = curve.argmax()
sim_best_q = q_values[best_index]
sim_best_profit = curve[best_index]

# ---- metric cards ----
col1, col2, col3, col4 = st.columns(4)
col1.metric("Expected profit at your Q", f"${expected_profit:,.2f}",
            help=f"95% CI: ${ci_low:,.2f} to ${ci_high:,.2f}")
col2.metric("Best Q (simulated)", f"{sim_best_q} trees",
            delta=f"{sim_best_q - Q:+d} vs your Q", delta_color="off")
col3.metric("Critical fractile Q", f"{best_q:.1f} trees")
col4.metric("Chance of running out", f"{stats['stockout_prob']:.1%}")

tab1, tab2, tab3 = st.tabs(["Optimal order", "Risk at your Q", "How it works"])

# ---- tab 1: optimal order ----
with tab1:
    st.write(f"Best Q from the simulation is **{sim_best_q}** trees, with expected profit "
             f"**${sim_best_profit:,.2f}** (95% CI: ${curve_low[best_index]:,.2f} to "
             f"${curve_high[best_index]:,.2f}). The critical fractile formula says "
             f"**{best_q:.1f}**. You picked **{Q}**.")

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
    q_fig.add_vline(x=Q, line_color="gray", line_dash="dot", annotation_text="Your Q",
                    annotation_position="bottom right")
    q_fig.update_layout(xaxis_title="Trees ordered (Q)", yaxis_title="Expected profit")
    st.plotly_chart(q_fig, use_container_width=True)

# ---- tab 2: risk at your Q ----
with tab2:
    st.write(f"Expected profit at Q = {Q}: **${expected_profit:,.2f}** "
             f"(95% CI: ${ci_low:,.2f} to ${ci_high:,.2f})")
    st.caption("The 95% CI is the range the true expected profit most likely falls in. "
               "If you ran the simulation over and over, about 95% of intervals built this way would contain it.")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Avg leftover trees", f"{stats['avg_leftover']:.1f}")
    s2.metric("Fill rate", f"{stats['fill_rate']:.1%}", help="% of total demand we were able to sell to")
    s3.metric("5th percentile profit", f"${stats['profit_p5']:,.2f}", help="basically a bad year")
    s4.metric("95th percentile profit", f"${stats['profit_p95']:,.2f}", help="basically a great year")

    s5, s6, s7, _ = st.columns(4)
    s5.metric("Avg trees sold", f"{stats['avg_sold']:.1f}")
    s6.metric("Avg missed sales", f"{stats['avg_lost_sales']:.1f}")
    s7.metric("Chance of losing money", f"{stats['prob_loss']:.1%}")

    left, right = st.columns(2)
    with left:
        st.subheader("Profit histogram")
        fig = px.histogram(x=profits, nbins=50, labels={"x": "Profit"})
        fig.add_vline(x=stats["profit_p5"], line_color="orange", line_dash="dash",
                      annotation_text="5th pct")
        fig.add_vline(x=stats["profit_p95"], line_color="orange", line_dash="dash",
                      annotation_text="95th pct")
        fig.update_layout(yaxis_title="Count", height=350)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Simulated tree demand")
        demand_fig = px.histogram(x=demand, nbins=40, labels={"x": "Trees demanded"})
        demand_fig.add_vline(x=Q, line_color="red", line_dash="dash", annotation_text="Your Q")
        demand_fig.update_layout(yaxis_title="Count", height=350)
        st.plotly_chart(demand_fig, use_container_width=True)

# ---- tab 3: how it works ----
with tab3:
    st.write("The lot has to order all its trees before the season without knowing how many people "
             "will show up. Too few and you miss sales, too many and you're stuck selling leftovers "
             "for mulch. This is the newsvendor problem.")

    st.subheader("Profit formula")
    st.latex(r"\text{profit} = p \cdot \min(Q, D) + s \cdot \max(Q - D, 0) - c \cdot Q")
    st.write("p = selling price, c = cost per tree, s = salvage value, Q = trees ordered, D = demand")

    st.subheader("Simulation")
    st.write(f"Draws {int(n_reps):,} random demand values from the distribution you picked (rounded to "
             "whole trees, no negatives), calculates profit for each and averages them. For the best Q "
             "it does this for a bunch of Qs using the same demand values each time (common random "
             "numbers) so the comparison between Qs is fair.")

    st.subheader("Critical fractile")
    st.latex(r"\frac{C_u}{C_u + C_o}, \quad C_u = p - c, \quad C_o = c - s")
    st.write("Cu = profit lost on a missed sale, Co = money lost on a leftover tree. The best Q is "
             f"where P(demand <= Q) equals this. With your numbers it's {cf:.1%}, so Q is about {best_q:.1f}.")

    st.subheader("Confidence intervals")
    st.latex(r"\bar{x} \pm 1.96 \cdot \frac{s}{\sqrt{n}}")
    st.write("Average profit +/- 1.96 x (std dev / sqrt(n)). Gets narrower with more replications, "
             "4x the replications cuts it about in half.")

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
        st.plotly_chart(ci_fig, use_container_width=True)

        st.dataframe(ci_table[["replications", "mean", "low", "high", "half_width"]].round(2),
                     hide_index=True)
