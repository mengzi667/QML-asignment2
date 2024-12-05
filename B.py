import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Load data from data_small.txt
data = pd.read_csv('data_small.txt', delim_whitespace=True, header=None,
                   names=["LOC_ID", "XCOORD", "YCOORD", "DEMAND", "SERVICETIME", "READYTIME", "DUETIME"])

# uncomment the following line to use the test dataset
# data = pd.read_csv('data_small_test.txt', delim_whitespace=True, header=None,
#                    names=["LOC_ID", "XCOORD", "YCOORD", "DEMAND", "SERVICETIME", "READYTIME", "DUETIME"])

# Extract relevant information
locations = data["LOC_ID"].values
coordinates = data[["XCOORD", "YCOORD"]].values
d = data["DEMAND"].values
s = data["SERVICETIME"].values
e = data["READYTIME"].values
l = data["DUETIME"].values

# Number of vehicles and vehicle capacity
V = 2
b = 130

# Calculate distance matrix (Euclidean distances)
num_locations = len(locations)
c = np.zeros((num_locations, num_locations))
for i in range(num_locations):
    for j in range(num_locations):
        c[i, j] = np.linalg.norm(coordinates[i] - coordinates[j])

# Gurobi Model
model = Model("CVRP-TW")

# Decision variables
x = model.addVars(num_locations, num_locations, V, vtype=GRB.BINARY, name="x")
z = model.addVars(num_locations, V, vtype=GRB.BINARY, name="z")
w = model.addVars(num_locations, V, vtype=GRB.CONTINUOUS, name="w")

# Objective function: minimize total distance
model.setObjective(quicksum(c[i, j] * x[i, j, v]
                            for i in range(num_locations)
                            for j in range(num_locations)
                            for v in range(V)), GRB.MINIMIZE)

# Constraints
# Each customer is visited exactly once
model.addConstrs(quicksum(x[i, j, v] for j in range(num_locations) for v in range(V)) == 1
                 for i in range(1, num_locations))

# Flow conservation
model.addConstrs(quicksum(x[i, j, v] for j in range(num_locations)) ==
                 quicksum(x[j, i, v] for j in range(num_locations))
                 for i in range(num_locations) for v in range(V))

# Each vehicle starts and ends at the depot
model.addConstrs(quicksum(x[0, j, v] for j in range(1, num_locations)) == 1 for v in range(V))
model.addConstrs(quicksum(x[i, 0, v] for i in range(1, num_locations)) == 1 for v in range(V))

# Vehicle capacity constraint
model.addConstrs(
    quicksum(d[i] * quicksum(x[i, j, v] for j in range(num_locations)) for i in range(1, num_locations))
    <= b for v in range(V))

# Time window constraints
model.addConstrs((w[i, v] >= e[i] * z[i, v]) for i in range(num_locations) for v in range(V))
model.addConstrs((w[i, v] <= l[i] * z[i, v]) for i in range(num_locations) for v in range(V))

# Subtour elimination
M = 1e5  # A sufficiently large number
model.addConstrs((w[i, v] + s[i] + c[i, j] - w[j, v]
                  <= M * (1 - x[i, j, v]))
                 for i in range(num_locations)
                 for j in range(1, num_locations)
                 for v in range(V))

# Solve the model
model.optimize()

# Extract results
routes = {v: [] for v in range(V)}
summary_data = []
total_distance = model.objVal
for v in range(V):
    current_node = 0  # Start from the depot
    load = 0  # Track vehicle load
    while True:
        for j in range(num_locations):
            if x[current_node, j, v].x > 0.5:
                next_node = j
                start_time = w[current_node, v].x
                end_time = w[next_node, v].x

                # Calculate load at origin and destination
                load_o = load
                load += d[next_node]
                load_d = load

                # Add record to summary data
                summary_data.append({
                    "Vehicle": v + 1,
                    "Origin": current_node,
                    "Destination": next_node,
                    "Time O": start_time,
                    "Time D": end_time,
                    "Load O": load_o,
                    "Load D": load_d,
                })

                # Add to routes and update current node
                routes[v].append((current_node, next_node))
                current_node = next_node

                # Break if returned to depot
                if next_node == 0:
                    break
        if current_node == 0:
            break

# Format results into a DataFrame sorted by route order
summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values(by=["Vehicle", "Time O"])
print(summary_df)

# Plot routes with directional arrows and labels, including all points
plt.figure(figsize=(12, 10))
colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']  # Color palette

# Draw all points as a base
for i, coord in enumerate(coordinates):
    plt.scatter(coord[0], coord[1], color='black', zorder=1)
    plt.text(coord[0] + 1, coord[1] + 1, str(i), fontsize=12, color='black', zorder=2)

# Draw the routes with arrows
for v, route in routes.items():
    color = colors[v % len(colors)]
    for idx, (i, j) in enumerate(route):
        # Draw directional arrow for the route
        arrow = FancyArrowPatch((coordinates[i][0], coordinates[i][1]),
                                (coordinates[j][0], coordinates[j][1]),
                                color=color, arrowstyle='->', mutation_scale=12, zorder=3)
        plt.gca().add_patch(arrow)
    # Add legend entry for the vehicle
    plt.plot([], [], color=color, label=f'Vehicle {v + 1}')

# Add legend
plt.legend(loc='best')

# Set plot details
plt.title("Routes for CVRP with Time Windows (All Points Visible)")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(True)
plt.show()