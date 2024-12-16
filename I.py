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
data_file = "data_large_multiTW_diff.txt"
nodes = []
with open(data_file, 'r') as file:
    for line in file:
        parts = line.strip().split('\t')
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
b_v = 200            # Vehicle capacity
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
