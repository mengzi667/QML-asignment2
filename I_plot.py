import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ---------------------- Data Loading ----------------------
# File paths
routes_file = "routes_variables.csv"
start_times_file = "start_times_variables.csv"
time_window_file = "time_window_variables.csv"
quantities_file = "quantities_variables.csv"
diff_file = "data_large_multiTW_diff.txt"

# Load CSV files
routes_df = pd.read_csv(routes_file)
start_times_df = pd.read_csv(start_times_file)
time_window_df = pd.read_csv(time_window_file)
quantities_df = pd.read_csv(quantities_file)

# Load node coordinates, demand, and time windows
nodes = []
with open(diff_file, 'r') as file:
    for line in file:
        parts = line.strip().split('\t')
        node_id = int(parts[0])
        x_coord = float(parts[1])
        y_coord = float(parts[2])
        demand = float(parts[3])
        servicetime = float(parts[4])
        num_tw = int(parts[5])
        time_windows = [(float(parts[6 + 2 * i]), float(parts[7 + 2 * i])) for i in range(num_tw)]
        nodes.append({
            'id': node_id, 'x': x_coord, 'y': y_coord, 
            'demand': demand, 'time_windows': time_windows
        })

# Coordinates and demand dictionary
coordinates = {node['id']: (node['x'], node['y']) for node in nodes}
demands = {node['id']: node['demand'] for node in nodes}
time_window_dict = {node['id']: node['time_windows'] for node in nodes}

# Replace node 19 with 0 (Depot)
routes_df['Origin'] = routes_df['Origin'].replace(19, 0).astype(int)
routes_df['Destination'] = routes_df['Destination'].replace(19, 0).astype(int)

# ---------------------- Path Reconstruction Function ----------------------
def reconstruct_path(vehicle_routes):
    """
    Reconstruct the vehicle route in the correct order.
    """
    path = []
    current_node = 0  # Start from the depot
    while True:
        next_nodes = vehicle_routes[vehicle_routes['Origin'] == current_node]
        if next_nodes.empty:
            break
        next_node = next_nodes.iloc[0]['Destination']
        path.append((current_node, next_node))
        current_node = next_node
        if current_node == 0:  # Return to depot
            break
    return path

# ---------------------- Generate Result Summary ----------------------
def create_vehicle_summary(routes, start_times, quantities, time_windows):
    vehicle_summaries = {}
    for v in routes['Vehicle'].unique():
        vehicle_routes = routes[routes['Vehicle'] == v]
        vehicle_times = start_times[start_times['Vehicle'] == v]
        vehicle_quantities = quantities[quantities['Vehicle'] == v]
        vehicle_time_windows = time_windows[time_windows['Vehicle'] == v]

        # Reconstruct the route
        path = reconstruct_path(vehicle_routes)

        # Initialize load and create result table
        load = 0  
        vehicle_table = []

        for idx, (origin, destination) in enumerate(path):
            if idx == 0:  # Force Time O = 0 at the depot
                time_o = 0.0
            else:
                time_o = vehicle_times.loc[vehicle_times['Node'] == origin, 'Start Time'].values[0] + servicetime
            
            time_d = vehicle_times.loc[vehicle_times['Node'] == destination, 'Start Time'].values[0]
            y_iv = vehicle_quantities.loc[vehicle_quantities['Node'] == destination, 'Quantity'].values[0] if destination != 0 else 0
            tw_index = vehicle_time_windows.loc[vehicle_time_windows['Node'] == destination, 'Time Window Index'].values[0] if destination != 0 else -1

            load_o = load
            load_d = load + y_iv
            load = load_d

            tw_used = time_window_dict[destination][int(tw_index)] 

            vehicle_table.append({
                "Origin": int(origin),
                "Destination": int(destination),
                "Time O (Arrival)": round(time_o, 1),
                "Time D (Departure)": round(time_d, 1),
                "Pick-up Quantity by V at D": y_iv,  # Output the y_iv column
                "Load O": load_o,
                "Load D": load_d,
                "Time Window Used of D": tw_used
            })

        vehicle_summaries[v] = pd.DataFrame(vehicle_table)
    return vehicle_summaries

# Generate summaries
vehicle_summaries = create_vehicle_summary(routes_df, start_times_df, quantities_df, time_window_df)

# Display and save results
for v, table in vehicle_summaries.items():
    print(f"\n--- Vehicle {v} Summary ---")
    print(table)
    table.to_csv(f"vehicle_{v}_summary.csv", index=False)  # Save with y_iv included
    print(f"Saved to vehicle_{v}_summary.csv")

# ---------------------- Plot Vehicle Routes ----------------------
plt.figure(figsize=(12, 10))
colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']

# Plot all nodes
for node_id, (x, y) in coordinates.items():
    plt.scatter(x, y, color='black', zorder=2)
    plt.text(x + 0.5, y + 0.5, str(node_id), fontsize=10, color='black', zorder=3)

# Plot routes
for v, vehicle_routes in routes_df.groupby('Vehicle'):
    color = colors[v % len(colors)]
    path = reconstruct_path(vehicle_routes)
    for origin, destination in path:
        x1, y1 = coordinates[origin]
        x2, y2 = coordinates[destination]
        arrow = FancyArrowPatch((x1, y1), (x2, y2), color=color, arrowstyle='->', mutation_scale=15, zorder=1)
        plt.gca().add_patch(arrow)
    plt.plot([], [], color=color, label=f"Vehicle {v}")

plt.legend(loc='best')
plt.title("Routes for VRP with Split Pickup and Multiple TW")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(True)
plt.show()
