import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
import time

def run_test_py_algorithm(V, b):
    # Load data from data_small.txt
    data = pd.read_csv('data_small.txt', delim_whitespace=True, header=None,
                       names=["LOC_ID", "XCOORD", "YCOORD", "DEMAND", "SERVICETIME", "READYTIME", "DUETIME"])

    # Extract relevant information
    locations = data["LOC_ID"].values
    coordinates = data[["XCOORD", "YCOORD"]].values
    d = data["DEMAND"].values
    s = data["SERVICETIME"].values

    # Calculate distance matrix (Euclidean distances)
    num_locations = len(locations)
    c = np.zeros((num_locations, num_locations))
    for i in range(num_locations):
        for j in range(num_locations):
            c[i, j] = np.linalg.norm(coordinates[i] - coordinates[j])

    # Gurobi Model
    model = Model("CVRP")

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

    # Subtour elimination
    M = 1e5  # A sufficiently large number
    model.addConstrs((w[i, v] + s[i] + c[i, j] - w[j, v]
                      <= M * (1 - x[i, j, v]))
                     for i in range(num_locations)
                     for j in range(1, num_locations)
                     for v in range(V))

    # Measure the start time
    start_time = time.time()

    # Solve the model
    model.optimize()

    # Measure the end time
    end_time = time.time()
    computation_time = end_time - start_time

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

    # Print computation time
    print(f"Computation Time: {computation_time:.2f} seconds")

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
    plt.title("Routes for CVRP (All Points Visible)")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()

# Manually input parameters
V = int(input("Enter the number of vehicles: "))
capacity = int(input("Enter the capacity limit for each vehicle: "))
time_window_option = int(input("Enter the number of time windows (0: No time windows, 1: Single time window, 3: Multiple time windows): "))

if time_window_option == 0:
    run_test_py_algorithm(V, capacity)
else:
    # Load data
    data = pd.read_csv('data_small_multiTW.txt', sep='\s+', header=None,
                       names=["LOC_ID", "XCOORD", "YCOORD", "DEMAND", "SERVICETIME", "NUM_TW",
                              "TW1_START", "TW1_END", "TW2_START", "TW2_END", "TW3_START", "TW3_END"])

    # Extract relevant information
    locations = data["LOC_ID"].values
    coordinates = data[["XCOORD", "YCOORD"]].values
    demands = data["DEMAND"].values
    service_times = data["SERVICETIME"].values
    time_windows = data.iloc[:, 6:].values

    # Calculate distance matrix
    num_locations = len(locations)
    distance_matrix = np.zeros((num_locations, num_locations))
    for i in range(num_locations):
        for j in range(num_locations):
            distance_matrix[i, j] = np.linalg.norm(coordinates[i] - coordinates[j])

    # Gurobi Model
    model = Model("CVRP-Multi-TW")

    # Decision variables
    x = model.addVars(num_locations, num_locations, V, vtype=GRB.BINARY, name="x")

    # Create time-related variables only if there are time windows
    w, z = None, None
    if time_window_option > 0:
        w = model.addVars(num_locations, V, vtype=GRB.CONTINUOUS, name="w")
        z = model.addVars(num_locations, time_window_option, V, vtype=GRB.BINARY, name="z")

    # Objective function
    model.setObjective(quicksum(distance_matrix[i, j] * x[i, j, v]
                                for i in range(num_locations)
                                for j in range(num_locations)
                                for v in range(V)), GRB.MINIMIZE)

    # Node visit constraints
    model.addConstrs(quicksum(x[i, j, v] for j in range(num_locations) for v in range(V)) == 1
                     for i in range(1, num_locations))

    # Flow conservation constraints
    model.addConstrs(quicksum(x[i, j, v] for j in range(num_locations)) ==
                     quicksum(x[j, i, v] for j in range(num_locations))
                     for i in range(num_locations) for v in range(V))

    # Each vehicle starts and ends at the depot
    model.addConstrs(quicksum(x[0, j, v] for j in range(1, num_locations)) == 1 for v in range(V))
    model.addConstrs(quicksum(x[i, 0, v] for i in range(1, num_locations)) == 1 for v in range(V))

    # Capacity constraints
    model.addConstrs(
        quicksum(demands[i] * quicksum(x[i, j, v] for j in range(num_locations)) for i in range(1, num_locations))
        <= capacity for v in range(V))

    # Add time window constraints only if there are time windows
    if time_window_option > 0:
        # Time window selection constraints
        for i in range(1, num_locations):
            model.addConstr(quicksum(z[i, tw, v] for tw in range(time_window_option) for v in range(V)) == 1)
            for v in range(V):
                for tw in range(time_window_option):
                    tw_start = time_windows[i, tw * 2]
                    tw_end = time_windows[i, tw * 2 + 1]
                    model.addConstr(w[i, v] >= tw_start - (1 - z[i, tw, v]) * 1e5)
                    model.addConstr(w[i, v] <= tw_end + (1 - z[i, tw, v]) * 1e5)

        # Bind paths to time windows
        for i in range(1, num_locations):
            for v in range(V):
                model.addConstr(quicksum(x[j, i, v] for j in range(num_locations)) ==
                                quicksum(z[i, tw, v] for tw in range(time_window_option)))

        # Time and path transmission constraints
        for i in range(num_locations):
            for j in range(1, num_locations):
                for v in range(V):
                    model.addConstr(w[j, v] >= w[i, v] + service_times[i] + distance_matrix[i, j] - 1e5 * (1 - x[i, j, v]))

    # Measure the start time
    start_time = time.time()

    # Solve the model
    model.optimize()

    # Measure the end time
    end_time = time.time()
    computation_time = end_time - start_time

    # Output routes and results
    if model.status == GRB.OPTIMAL:
        print("Optimal solution found!")
        routes = {v: [] for v in range(V)}
        summary_data = []

        for v in range(V):
            current_node = 0
            load = 0
            while True:
                for j in range(num_locations):
                    if x[current_node, j, v].x > 0.5:
                        start_time = w[current_node, v].x if w is not None else 0
                        end_time = w[j, v].x if w is not None else 0

                        # Record the route
                        summary_data.append({
                            "Vehicle": v + 1,
                            "Origin": current_node,
                            "Destination": j,
                            "Time O": start_time,
                            "Time D": end_time,
                            "Load O": load,
                            "Load D": load + demands[j] if j != 0 else 0,
                        })

                        # Update node and load
                        routes[v].append((current_node, j))
                        load += demands[j]
                        current_node = j
                        break
                if current_node == 0:
                    break

        # Print the results table
        summary_df = pd.DataFrame(summary_data)
        print(summary_df)

        # Print computation time
        print(f"Computation Time: {computation_time:.2f} seconds")

        # Plot the routes
        plt.figure(figsize=(12, 10))
        colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']

        # Draw all nodes
        for i, coord in enumerate(coordinates):
            plt.scatter(coord[0], coord[1], color='black', zorder=2)
            plt.text(coord[0] + 1, coord[1] + 1, str(i), fontsize=12, color='black', zorder=3)

        # Draw the routes
        for v, route in routes.items():
            color = colors[v % len(colors)]
            for i, j in route:
                arrow = FancyArrowPatch((coordinates[i][0], coordinates[i][1]),
                                        (coordinates[j][0], coordinates[j][1]),
                                        color=color, arrowstyle='->', mutation_scale=12, zorder=1)
                plt.gca().add_patch(arrow)
            plt.plot([], [], color=color, label=f'Vehicle {v + 1}')

        # Set legend and title
        plt.legend(loc='best')
        plt.title("Routes for CVRP without Time Windows (All Points Visible)")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True)
        plt.show()
    else:
        print("No solution found.")
        model.computeIIS()
        model.write("infeasible.ilp")
