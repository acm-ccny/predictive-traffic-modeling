import pandas as pd
import numpy as np
import networkx as nx
import heapq
import folium
from sklearn.ensemble import RandomForestRegressor
from scipy.spatial import KDTree

# =========================
# 1. LOAD DATA
# =========================
nodes = pd.read_csv("data/ml_datasets/routing_nodes.csv")
edges = pd.read_csv("data/ml_datasets/routing_edges.csv")
ml_data = pd.read_csv("data/ml_datasets/congestion_ml.csv")

# =========================
# 2. TRAIN MODEL
# =========================
ml_features = [
    'hour', 'is_weekend', 'is_rush_hour',
    'borough_Bronx', 'borough_Brooklyn', 'borough_Manhattan',
    'borough_Queens', 'borough_Staten Island'
]

ml_data_clean = ml_data.dropna(subset=['avg_travel_time'])
X_train = ml_data_clean[ml_features]
y_train = ml_data_clean['avg_travel_time']


rf_model = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# =========================
# 3. BUILD GRAPH
# =========================
G = nx.DiGraph()
edge_lookup = {}

for _, n in nodes.iterrows():
    G.add_node(n['node_id'], lat=n['lat'], lon=n['lon'])

for _, row in edges.iterrows():
    u, v = row['from_node'], row['to_node']
    if u not in G or v not in G: 
        continue
    
    if (u, v) not in edge_lookup:
        edge_lookup[(u, v)] = {'times': {}, 'borough': row['borough']}
    
    # Floor travel time at 10s to prevent logical 'teleportation' errors
    t_val = max(row['est_travel_time_sec'], 10)
    edge_lookup[(u, v)]['times'][row['HH']] = t_val
    G.add_edge(u, v)

# =========================
# 4. HEAL GRAPH
# =========================
# Bridges gaps between isolated road segments to ensure a connected network
coords = nodes[['lat', 'lon']].values
tree = KDTree(coords)
pairs = tree.query_pairs(0.03) # Connecting nodes within ~2.2km radius

for i, j in pairs:
    u, v = nodes.iloc[i]['node_id'], nodes.iloc[j]['node_id']
    if not G.has_edge(u, v):
        dist = np.sqrt(
            (nodes.iloc[i]['lat'] - nodes.iloc[j]['lat']) ** 2
            + (nodes.iloc[i]['lon'] - nodes.iloc[j]['lon']) ** 2
        ) * 111000

        time_est = max(dist / 8.0, 10)

        edge_info = {
            'times': {},
            'borough': 'Manhattan',
            'is_virtual': True,
            'base_time': time_est,
        }
        # Same dict for both directions: G adds u→v and v→u; costs are symmetric.
        edge_lookup[(u, v)] = edge_info
        edge_lookup[(v, u)] = edge_info

        G.add_edge(u, v)
        G.add_edge(v, u)


def _virtual_edge_info(lat1, lon1, lat2, lon2):
    """Travel-time payload for synthetic edges (same formula as nearby-node healing)."""
    dist_m = np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) * 111000
    time_est = max(dist_m / 8.0, 10)
    return {
        'times': {},
        'borough': 'Manhattan',
        'is_virtual': True,
        'base_time': time_est,
    }


def _bridge_components(graph, lookup, components_fn):
    """
    Join disjoint components by adding a bidirectional virtual edge between the
    geographically closest node pair across two different components. Repeat until
    one component remains (weakly or strongly, depending on components_fn).
    """
    while True:
        comps = list(components_fn(graph))
        if len(comps) <= 1:
            break

        best_d, best_u, best_v = float('inf'), None, None
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                for u in comps[i]:
                    nu = graph.nodes[u]
                    for v in comps[j]:
                        nv = graph.nodes[v]
                        d = np.sqrt(
                            (nu['lat'] - nv['lat']) ** 2
                            + (nu['lon'] - nv['lon']) ** 2
                        )
                        if d < best_d:
                            best_d, best_u, best_v = d, u, v

        edge_info = _virtual_edge_info(
            graph.nodes[best_u]['lat'],
            graph.nodes[best_u]['lon'],
            graph.nodes[best_v]['lat'],
            graph.nodes[best_v]['lon'],
        )
        lookup[(best_u, best_v)] = edge_info
        lookup[(best_v, best_u)] = edge_info
        graph.add_edge(best_u, best_v)
        graph.add_edge(best_v, best_u)


# CSV edges leave ~44 weak regions; KDTree healing does not always connect them.
_bridge_components(G, edge_lookup, nx.weakly_connected_components)
# One-way streets can still block s→t even when the map is one weak component.
_bridge_components(G, edge_lookup, nx.strongly_connected_components)

# =========================
# 5. COST FUNCTION
# =========================
def get_dynamic_cost(u, v, current_time_sec):
    edge = edge_lookup[(u, v)]
    hour = (int(current_time_sec) // 3600) % 24
    
    # Priority 1: Historical Hourly Data
    if hour in edge['times']: return edge['times'][hour]
    # Priority 2: Virtual Road Estimate (Distance-based)
    if 'is_virtual' in edge: return edge['base_time']
    
    b = edge.get('borough', 'Manhattan')
    b_flags = {f'borough_{name}': (1 if name == b else 0) for name in ['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island']}
    inp = pd.DataFrame([{'hour': hour, 'is_weekend': 0, 'is_rush_hour': 1 if (7<=hour<=9 or 16<=hour<=19) else 0, **b_flags}])
    return rf_model.predict(inp[ml_features])[0]

# =========================
# 6. ROUTING
# =========================
def predict_route(source, target, start_hour):
    if source == target:
        return [source], 0

    start_sec = start_hour * 3600
    pq = [(start_sec, source)]
    best = {source: start_sec}
    parent = {}

    while pq:
        t, u = heapq.heappop(pq)
        if u == target: break
        if t > best.get(u, float('inf')): continue
        
        for v in G.successors(u):
            cost = get_dynamic_cost(u, v, t)
            new_t = t + max(5, cost) # Safety floor for Dijkstra
            
            if new_t < best.get(v, float('inf')):
                best[v] = new_t
                parent[v] = u
                heapq.heappush(pq, (new_t, v))
    
    if target not in parent: return None, 0
    path = []
    curr = target
    while curr != source:
        path.append(curr); curr = parent[curr]
    path.append(source)
    # Correct calculation: Arrival Time - Start Time
    return path[::-1], best[target] - start_sec

# =========================
# 7. MAP RENDERING
# =========================
def render_map(G, path):
    if not path: return None
    # Initialize map at the starting point
    m = folium.Map(location=[G.nodes[path[0]]['lat'], G.nodes[path[0]]['lon']], zoom_start=13)
    coords = [(G.nodes[n]['lat'], G.nodes[n]['lon']) for n in path]
    folium.PolyLine(coords, color="blue", weight=5, opacity=0.8).add_to(m)
    folium.Marker(coords[0], popup="Start", icon=folium.Icon(color='green')).add_to(m)
    folium.Marker(coords[-1], popup="End", icon=folium.Icon(color='red')).add_to(m)
    return m


def render_map_with_congestion(G, path, start_hour=None):
    """Alias for Streamlit; congestion styling does not vary by hour in this map."""
    return render_map(G, path)


# Example Usage
if __name__ == "__main__":
    start = "1 AVENUE @ EAST 116 STREET"
    end = "1 AVENUE @ EAST 34 STREET"
    path, time_sec = predict_route(start, end, 8)
    if path:
        print(f"Success! Route found with {len(path)} nodes.")
        print(f"Total travel time: {time_sec/60:.2f} minutes")
