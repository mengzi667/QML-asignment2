import gurobipy as gp
from gurobipy import GRB
import numpy as np

# 解析数据文件
def parse_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    locations = []
    for line in lines:
        parts = line.strip().split()
        loc_id, x, y, demand, service_time = map(int, parts[:5])
        num_tw = int(parts[5])  # 第六列是时间窗口数量
        time_windows = []
        for i in range(num_tw):
            ready_time, due_time = map(int, parts[6 + 2 * i:8 + 2 * i])  # 从第7列开始依次读取时间窗口
            time_windows.append((ready_time, due_time))
        locations.append({
            'loc_id': loc_id,
            'x': x,
            'y': y,
            'demand': demand,
            'service_time': service_time,
            'time_windows': time_windows
        })
    # 设置出发点的时间窗为 (0, ∞)
    locations[0]['time_windows'] = [(0, float('inf'))]
    return locations

# 计算欧几里得距离
def euclidean_distance(loc1, loc2):
    return np.sqrt((loc1['x'] - loc2['x'])**2 + (loc1['y'] - loc2['y'])**2)

# 建立并求解多时间窗口的模型
def solve_vrp(locations, num_vehicles, vehicle_capacity):
    n = len(locations)
    dist = np.array([[euclidean_distance(locations[i], locations[j]) for j in range(n)] for i in range(n)])
    
    model = gp.Model("MultiTW_VRP")
    
    # 决策变量
    x = model.addVars(n, n, num_vehicles, vtype=GRB.BINARY, name="x")  # 路径决策变量
    z = {}  # 时间窗口选择变量
    for i in range(n):
        for t in range(len(locations[i]['time_windows'])):
            for k in range(num_vehicles):
                z[i, t, k] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}_{t}_{k}")

    start_time = model.addVars(n, num_vehicles, vtype=GRB.CONTINUOUS, name="start_time")  # 服务开始时间

    # 固定出发点的服务时间
    for k in range(num_vehicles):
        model.addConstr(start_time[0, k] == 0, f"StartTimeDepot_{k}")
    
    # 目标函数：最小化总距离
    model.setObjective(gp.quicksum(dist[i][j] * x[i, j, k]
                                   for i in range(n) for j in range(n) for k in range(num_vehicles)), GRB.MINIMIZE)
    
    # 约束条件
    for k in range(num_vehicles):
        # 每辆车从仓库出发并返回
        model.addConstr(gp.quicksum(x[0, j, k] for j in range(1, n)) == 1)
        model.addConstr(gp.quicksum(x[i, 0, k] for i in range(1, n)) == 1)
    
    for i in range(1, n):
        # 每个地点只能被访问一次
        model.addConstr(gp.quicksum(x[i, j, k] for j in range(n) for k in range(num_vehicles)) == 1)
        # 起点和终点流平衡
        for k in range(num_vehicles):
            model.addConstr(gp.quicksum(x[i, j, k] for j in range(n)) == gp.quicksum(x[j, i, k] for j in range(n)))
    
    for k in range(num_vehicles):
        # 车辆容量限制
        model.addConstr(gp.quicksum(locations[i]['demand'] * gp.quicksum(x[i, j, k] for j in range(n)) for i in range(1, n)) <= vehicle_capacity)
    
    for i in range(1, n):
        for k in range(num_vehicles):
            # 每个地点必须选择一个时间窗口
            model.addConstr(gp.quicksum(z[i, t, k] for t in range(len(locations[i]['time_windows']))) == 1)
            
            # 时间窗约束
            for t, (ready_time, due_time) in enumerate(locations[i]['time_windows']):
                model.addConstr(start_time[i, k] >= ready_time * z[i, t, k], f"ReadyTime_{i}_{k}_{t}")
                model.addConstr(start_time[i, k] <= due_time * z[i, t, k], f"DueTime_{i}_{k}_{t}")
            
            # 服务时间和旅行时间约束
            for j in range(1, n):
                if i != j:
                    model.addConstr(start_time[j, k] >= start_time[i, k] + locations[i]['service_time'] + dist[i, j]
                                    - 1e5 * (1 - x[i, j, k]), f"ServiceTime_{i}_{j}_{k}")
    
    # 求解模型
    model.optimize()
    
    # 输出结果
    if model.status == GRB.OPTIMAL:
        print(f"Optimal total distance: {model.objVal}")
        for k in range(num_vehicles):
            print(f"Vehicle {k + 1} route:")
            for i in range(n):
                for j in range(n):
                    if x[i, j, k].X > 0.5:
                        print(f"From {i} to {j}")
    
    return model

# 数据加载与模型运行
file_path = "data_small_multiTW.txt"
locations = parse_data(file_path)
solve_vrp(locations, num_vehicles=2, vehicle_capacity=130)
