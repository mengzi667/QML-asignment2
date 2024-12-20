import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import math
import time
import json

# ---------------------- Data Parsing ----------------------
# Reading data file with variable time windows
data_file = "data_small_multiTW.txt"
nodes = []
with open(data_file, 'r') as file:
    for line in file:
        parts = line.strip().split()
        node_id = int(parts[0])
        x_coord = float(parts[1])
        y_coord = float(parts[2])
        demand = float(parts[3])
        service_time = float(parts[4])
        num_tw = int(parts[5])
        time_windows = [(float(parts[6 + 2 * i]), float(parts[7 + 2 * i])) for i in range(num_tw)]
        nodes.append({
            'id': node_id,
            'x': x_coord,
            'y': y_coord,
            'demand': demand,
            'service_time': service_time,
            'time_windows': time_windows
        })

# Duplicate depot as virtual depot (n+1)
virtual_depot = nodes[0].copy()
virtual_depot['id'] = len(nodes)  # Assign virtual depot ID
nodes.append(virtual_depot)

n = len(nodes)   # Number of nodes
N0 = range(1, n-1)  # Customer nodes (excluding depot and virtual depot)
N = range(n)        # All nodes (including depot and virtual depot)
V = range(3)        # Vehicles (3 vehicles)
b_v = 65            # Vehicle capacity
M = 1e6             # Large constant for time constraints
coordinates = [(node['x'], node['y']) for node in nodes]

# Calculate distances between nodes
distance = {(i, j): math.sqrt((nodes[i]['x'] - nodes[j]['x'])**2 + (nodes[i]['y'] - nodes[j]['y'])**2)
            for i in N for j in N if i != j}

# ---------------------- Model Creation ----------------------
model = gp.Model("Split_Pickup_MultiTW")

# Decision Variables
x = model.addVars(N, N, V, vtype=GRB.BINARY, name="x")  # Vehicle routing
y = model.addVars(N0, V, vtype=GRB.CONTINUOUS, name="y")  # Quantity picked up
z = model.addVars(N0, V, len(nodes[1]['time_windows']), vtype=GRB.BINARY, name="z")  # Time window selection
w = model.addVars(N, V, vtype=GRB.CONTINUOUS, name="w")  # Service start time

# ---------------------- Objective Function ----------------------
model.setObjective(gp.quicksum(distance[i, j] * x[i, j, v] for i in N for j in N for v in V if i != j), GRB.MINIMIZE)

# ---------------------- Constraints ----------------------
# (27) Vehicle capacity constraint
model.addConstrs((gp.quicksum(y[i, v] for i in N0) <= b_v for v in V), name="capacity")

# (28) Total demand satisfaction
model.addConstrs((gp.quicksum(y[i, v] for v in V) == nodes[i]['demand'] for i in N0), name="demand")

# (29) Linking y with x
model.addConstrs((y[i, v] <= nodes[i]['demand'] * gp.quicksum(x[i, j, v] for j in N if j != i)
                  for v in V for i in N0), name="link_y_x")

# (30) Each node visited at least once in a time window
model.addConstrs((gp.quicksum(z[i, v, t] for v in V for t in range(len(nodes[i]['time_windows']))) >= 1
                  for i in N0), name="visit")

# (31) Flow conservation
model.addConstrs((gp.quicksum(x[i, j, v] for i in N if i != j) == gp.quicksum(x[j, k, v] for k in N if k != j)
                  for v in V for j in N0), name="flow")

# (32) Vehicles start from the depot
model.addConstrs((gp.quicksum(x[0, j, v] for j in N) == 1 for v in V), name="start")

# (33) Vehicles return to the virtual depot
model.addConstrs((gp.quicksum(x[i, n-1, v] for i in N) == 1 for v in V), name="end")

# (34) Time window constraints
model.addConstrs((nodes[i]['time_windows'][t][0] * z[i, v, t] <= w[i, v] for i in N0 for v in V 
                  for t in range(len(nodes[i]['time_windows'])))
                 , name="time_window_lb")

model.addConstrs((w[i, v] <= nodes[i]['time_windows'][t][1] * z[i, v, t] + M * (1 - z[i, v, t])
                  for i in N0 for v in V for t in range(len(nodes[i]['time_windows'])))
                 , name="time_window_ub")

# (35) Service time constraints
model.addConstrs((w[i, v] + nodes[i]['service_time'] + distance[i, j] <= w[j, v] + M * (1 - x[i, j, v])
                  for v in V for i in N for j in N if i != j), name="service_time")

# (36) Linking z and x
model.addConstrs((gp.quicksum(z[i, v, t] for t in range(len(nodes[i]['time_windows']))) ==
                  gp.quicksum(x[i, j, v] for j in N if j != i)
                  for v in V for i in N0), name="link_z_x")

# ---------------------- Solve the Model ----------------------
start_time = time.time()  
model.optimize()
end_time = time.time() 

# ---------------------- Print Computation Time ----------------------
print("\nComputation Time: {:.2f} seconds".format(end_time - start_time))

# ---------------------- Results and Plotting ----------------------
if model.status == GRB.OPTIMAL:
    print("Optimal solution found!")

    # Extract x (vehicle routes), y (quantity picked), z (time window selection), and w (service start times)
    x_data = []
    y_data = []
    z_data = []
    w_data = []

    for v in V:  # Vehicles
        # Extract x and y
        for i in N:  # Origin nodes
            for j in N:  # Destination nodes
                if i != j and x[i, j, v].X > 0.5:  # Vehicle routes
                    x_data.append({
                        "Vehicle": v + 1,
                        "Origin": i,
                        "Destination": j,
                        "Value": x[i, j, v].X
                    })
            if i in N0:  # Quantity picked at customer nodes
                y_data.append({
                    "Vehicle": v + 1,
                    "Node": i,
                    "Quantity": round(y[i, v].X, 3)
                })

        # Extract z (time window selection)
        for i in N0:  # Customer nodes
            for t in range(len(nodes[i]['time_windows'])):
                if z[i, v, t].X > 0.5:
                    z_data.append({
                        "Vehicle": v + 1,
                        "Node": i,
                        "Time Window Index": t,
                        "Value": z[i, v, t].X
                    })

        # Extract w (service start times)
        for i in N:  # All nodes
            w_data.append({
                "Vehicle": v + 1,
                "Node": i,
                "Start Time": round(w[i, v].X, 3)
            })

    # Convert to DataFrames
    x_df = pd.DataFrame(x_data)
    y_df = pd.DataFrame(y_data)
    z_df = pd.DataFrame(z_data)
    w_df = pd.DataFrame(w_data)

    # Print data
    print("\n--- Decision Variable: x (Routes) ---")
    print(x_df)

    print("\n--- Decision Variable: y (Quantities Picked) ---")
    print(y_df)

    print("\n--- Decision Variable: z (Time Window Selection) ---")
    print(z_df)

    print("\n--- Decision Variable: w (Service Start Times) ---")
    print(w_df)

    # Save data to CSV files
    x_df.to_csv("routes_variables.csv", index=False)
    y_df.to_csv("quantities_variables.csv", index=False)
    z_df.to_csv("time_window_variables.csv", index=False)
    w_df.to_csv("start_times_variables.csv", index=False)

    print("\nDecision variable data saved to:")
    print(" - 'routes_variables.csv' for x")
    print(" - 'quantities_variables.csv' for y")
    print(" - 'time_window_variables.csv' for z")
    print(" - 'start_times_variables.csv' for w")

# ---------------------- Plot Vehicle Routes ----------------------
'''
!!! Please note! The following code is for plotting. To run it, create a new `.py` file to 
save this code, then run it after running the above code to generate the charts and 
results table. Since only one code file can be uploaded, I have combined them in this manner.
Thank you for your understanding.
'''
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.patches import FancyArrowPatch

# # ---------------------- Data Loading ----------------------
# # File paths
# routes_file = "routes_variables.csv"
# start_times_file = "start_times_variables.csv"
# time_window_file = "time_window_variables.csv"
# quantities_file = "quantities_variables.csv"
# diff_file = "data_small_multiTW.txt"

# # Load CSV files
# routes_df = pd.read_csv(routes_file)
# start_times_df = pd.read_csv(start_times_file)
# time_window_df = pd.read_csv(time_window_file)
# quantities_df = pd.read_csv(quantities_file)

# # Load node coordinates, demand, and time windows
# nodes = []
# with open(diff_file, 'r') as file:
#     for line in file:
#         parts = line.strip().split( )
#         node_id = int(parts[0])
#         x_coord = float(parts[1])
#         y_coord = float(parts[2])
#         demand = float(parts[3])
#         servicetime = float(parts[4])
#         num_tw = int(parts[5])
#         time_windows = [(float(parts[6 + 2 * i]), float(parts[7 + 2 * i])) for i in range(num_tw)]
#         nodes.append({
#             'id': node_id, 'x': x_coord, 'y': y_coord, 
#             'demand': demand, 'time_windows': time_windows
#         })

# # Coordinates and demand dictionary
# coordinates = {node['id']: (node['x'], node['y']) for node in nodes}
# demands = {node['id']: node['demand'] for node in nodes}
# time_window_dict = {node['id']: node['time_windows'] for node in nodes}

# # Replace node 19 with 0 (Depot)
# routes_df['Origin'] = routes_df['Origin'].replace(9, 0).astype(int)
# routes_df['Destination'] = routes_df['Destination'].replace(9, 0).astype(int)

# # ---------------------- Path Reconstruction Function ----------------------
# def reconstruct_path(vehicle_routes):
#     """
#     Reconstruct the vehicle route in the correct order.
#     """
#     path = []
#     current_node = 0  # Start from the depot
#     while True:
#         next_nodes = vehicle_routes[vehicle_routes['Origin'] == current_node]
#         if next_nodes.empty:
#             break
#         next_node = next_nodes.iloc[0]['Destination']
#         path.append((current_node, next_node))
#         current_node = next_node
#         if current_node == 0:  # Return to depot
#             break
#     return path

# # ---------------------- Generate Result Summary ----------------------
# def create_vehicle_summary(routes, start_times, quantities, time_windows):
#     vehicle_summaries = {}
#     for v in routes['Vehicle'].unique():
#         vehicle_routes = routes[routes['Vehicle'] == v]
#         vehicle_times = start_times[start_times['Vehicle'] == v]
#         vehicle_quantities = quantities[quantities['Vehicle'] == v]
#         vehicle_time_windows = time_windows[time_windows['Vehicle'] == v]

#         # Reconstruct the route
#         path = reconstruct_path(vehicle_routes)

#         # Initialize load and create result table
#         load = 0  
#         vehicle_table = []

#         for idx, (origin, destination) in enumerate(path):
#             if idx == 0:  # Force Time O = 0 at the depot
#                 time_o = 0.0
#             else:
#                 time_o = vehicle_times.loc[vehicle_times['Node'] == origin, 'Start Time'].values[0] + servicetime
            
#             time_d = vehicle_times.loc[vehicle_times['Node'] == destination, 'Start Time'].values[0]
#             y_iv = vehicle_quantities.loc[vehicle_quantities['Node'] == destination, 'Quantity'].values[0] if destination != 0 else 0
#             tw_index = vehicle_time_windows.loc[vehicle_time_windows['Node'] == destination, 'Time Window Index'].values[0] if destination != 0 else -1

#             load_o = load
#             load_d = load + y_iv
#             load = load_d

#             tw_used = time_window_dict[destination][int(tw_index)] 

#             vehicle_table.append({
#                 "Origin": int(origin),
#                 "Destination": int(destination),
#                 "Time O (Arrival)": round(time_o, 1),
#                 "Time D (Departure)": round(time_d, 1),
#                 "Pick-up Quantity by V at D": y_iv,  # Output the y_iv column
#                 "Load O": load_o,
#                 "Load D": load_d,
#                 "Time Window Used of D": tw_used
#             })

#         vehicle_summaries[v] = pd.DataFrame(vehicle_table)
#     return vehicle_summaries

# # Generate summaries
# vehicle_summaries = create_vehicle_summary(routes_df, start_times_df, quantities_df, time_window_df)

# # Display and save results
# for v, table in vehicle_summaries.items():
#     print(f"\n--- Vehicle {v} Summary ---")
#     print(table)
#     table.to_csv(f"vehicle_{v}_summary.csv", index=False)  # Save with y_iv included
#     print(f"Saved to vehicle_{v}_summary.csv")

# # ---------------------- Plot Vehicle Routes ----------------------
# plt.figure(figsize=(12, 10))
# colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']

# # Plot all nodes
# for node_id, (x, y) in coordinates.items():
#     plt.scatter(x, y, color='black', zorder=2)
#     plt.text(x + 0.5, y + 0.5, str(node_id), fontsize=10, color='black', zorder=3)

# # Plot routes
# for v, vehicle_routes in routes_df.groupby('Vehicle'):
#     color = colors[v % len(colors)]
#     path = reconstruct_path(vehicle_routes)
#     for origin, destination in path:
#         x1, y1 = coordinates[origin]
#         x2, y2 = coordinates[destination]
#         arrow = FancyArrowPatch((x1, y1), (x2, y2), color=color, arrowstyle='->', mutation_scale=15, zorder=1)
#         plt.gca().add_patch(arrow)
#     plt.plot([], [], color=color, label=f"Vehicle {v}")

# plt.legend(loc='best')
# plt.title("Routes for VRP with Split Pickup and Multiple TW")
# plt.xlabel("X Coordinate")
# plt.ylabel("Y Coordinate")
# plt.grid(True)
# plt.show()
