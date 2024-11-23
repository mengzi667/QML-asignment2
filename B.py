import numpy as np
import matplotlib.pyplot as plt
from gurobipy import *
import pandas as pd

def read_data(filename):
    # 读取数据文件
    data = pd.read_csv(filename, delimiter='\s+', header=None)
    x_coords = data[1].values  # x坐标
    y_coords = data[2].values  # y坐标
    demands = data[3].values   # 需求量
    return x_coords, y_coords, demands

def calculate_distances(x_coords, y_coords):
    # 计算距离矩阵
    distances = np.sqrt((x_coords[:, None] - x_coords[None, :]) ** 2 + 
                       (y_coords[:, None] - y_coords[None, :]) ** 2)
    distances[np.isnan(distances)] = 0
    distances[np.isinf(distances)] = 0
    return distances

def solve_vrp(x_coords, y_coords, demands, num_vehicles=2, vehicle_capacity=130):
    n = len(x_coords)
    distances = calculate_distances(x_coords, y_coords)
    
    # 创建模型
    model = Model('VRP')
    
    # 决策变量
    x = {}
    for i in range(n):
        for j in range(n):
            for k in range(num_vehicles):
                x[i,j,k] = model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}_{k}')
    
    # 负载变量
    u = {}
    for i in range(n):
        for k in range(num_vehicles):
            u[i,k] = model.addVar(name=f'u_{i}_{k}')
    
    # 目标函数：最小化总距离
    obj = quicksum(distances[i,j] * x[i,j,k] 
                  for i in range(n) for j in range(n) for k in range(num_vehicles))
    model.setObjective(obj, GRB.MINIMIZE)
    
    # 约束条件
    # 1. 每个客户点只能被访问一次
    for j in range(1, n):
        model.addConstr(quicksum(x[i,j,k] 
                               for i in range(n) for k in range(num_vehicles)) == 1)
    
    # 2. 每辆车必须从depot出发
    for k in range(num_vehicles):
        model.addConstr(quicksum(x[0,j,k] for j in range(1, n)) <= 1)
    
    # 3. 流量守恒约束
    for k in range(num_vehicles):
        for h in range(n):
            model.addConstr(
                quicksum(x[i,h,k] for i in range(n)) ==
                quicksum(x[h,j,k] for j in range(n))
            )
    
    # 4. 容量约束和子回路消除约束
    for i in range(1, n):
        for k in range(num_vehicles):
            model.addConstr(u[i,k] <= vehicle_capacity)
            model.addConstr(u[i,k] >= demands[i])
    
    for i in range(1, n):
        for j in range(1, n):
            for k in range(num_vehicles):
                if i != j:
                    model.addConstr(
                        u[i,k] - u[j,k] + vehicle_capacity * x[i,j,k] <= 
                        vehicle_capacity - demands[j]
                    )
    
    # 求解
    model.optimize()
    
    # 提取结果
    routes = []
    total_distance = model.objVal
    
    for k in range(num_vehicles):
        route = [0]
        current = 0
        while True:
            next_stop = None
            for j in range(n):
                if x[current,j,k].x > 0.5:
                    next_stop = j
                    break
            if next_stop is None or next_stop == 0:
                break
            route.append(next_stop)
            current = next_stop
        if len(route) > 1:
            route.append(0)
            routes.append(route)
    
    return routes, total_distance

def plot_solution(x_coords, y_coords, routes):
    plt.figure(figsize=(10, 10))
    # 绘制所有点
    plt.scatter(x_coords[1:], y_coords[1:], c='blue', label='Customers')
    plt.scatter(x_coords[0], y_coords[0], c='red', marker='s', label='Depot')
    
    # 为每条路线使用不同的颜色
    colors = ['g', 'm']
    for i, route in enumerate(routes):
        route_coords_x = x_coords[route]
        route_coords_y = y_coords[route]
        plt.plot(route_coords_x, route_coords_y, c=colors[i], 
                label=f'Vehicle {i+1}')
    
    plt.legend()
    plt.title('VRP Solution')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.grid(True)
    plt.show()

def print_solution(routes, x_coords, y_coords, demands, distances):
    print("\nSolution Details:")
    print(f"Total number of vehicles used: {len(routes)}")
    
    for i, route in enumerate(routes):
        print(f"\nVehicle {i+1} Route:")
        print("Stop  Location(x,y)  Demand  Distance to next")
        total_load = 0
        route_distance = 0
        
        for j in range(len(route)-1):
            current = route[j]
            next_stop = route[j+1]
            total_load += demands[current]
            distance_to_next = distances[current][next_stop]
            route_distance += distance_to_next
            
            print(f"{current:3d}  ({x_coords[current]:6.1f},{y_coords[current]:6.1f})  {demands[current]:6.1f}  {distance_to_next:8.1f}")
        
        print(f"Total distance for vehicle {i+1}: {route_distance:.2f}")
        print(f"Total load for vehicle {i+1}: {total_load:.2f}")

def main():
    # 读取数据
    x_coords, y_coords, demands = read_data("data_small.txt")
    distances = calculate_distances(x_coords, y_coords)
    
    # 求解VRP
    routes, total_distance = solve_vrp(x_coords, y_coords, demands)
    
    # 打印结果
    print(f"\nTotal distance traveled: {total_distance:.2f}")
    print_solution(routes, x_coords, y_coords, demands, distances)
    
    # 绘制解决方案
    plot_solution(x_coords, y_coords, routes)

if __name__ == "__main__":
    main()