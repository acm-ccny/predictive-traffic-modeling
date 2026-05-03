# This STREAMLIT APP should actually be a separate .py file, but for simplicity we include it here.
import streamlit as st
from streamlit_folium import st_folium

from routing_engine import G, predict_route, render_map_with_congestion

st.title("Predictive Traffic Routing")

node_ids = list(G.nodes)

start = st.selectbox("Start Node", node_ids)
end = st.selectbox("End Node", node_ids)
hour = st.slider("Departure Hour", 0, 23, 8)

if st.button("Find Route"):
    path, total_time = predict_route(start, end, hour)

    if path is None:
        st.error("No route found")
    else:
        st.write(f"Estimated travel time: {total_time / 60:.2f} minutes")

        m = render_map_with_congestion(G, path, hour)
        st_folium(m, width=700, height=500)