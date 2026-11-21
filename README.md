# Christmas Tree Order Simulator

Streamlit app for the newsvendor problem, but with christmas trees. A tree lot has to decide how many trees to order before the season, without knowing how many people will actually come buy one. Order too many and you're stuck selling leftovers for mulch, order too few and you miss sales.

The app simulates a bunch of random demand scenarios (5000 by default) and shows the expected profit, the chance of running out of trees, and a histogram of profits. It also shows the critical fractile and the order quantity it suggests so you can compare it to what you picked.

Demand can be normal or uniform.

To run it:

pip install -r requirements.txt
streamlit run app.py

app.py is the streamlit part and sim.py has the simulation code.
