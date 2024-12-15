from gurobipy import Model, GRB, quicksum
import pandas as pd

# Load the CSV file
csv_file_path = "data.csv"  # Replace with the actual file path
data = pd.read_csv(csv_file_path)
# Extract arcs as a list of tuples
arcs = list(zip(data['start node'], data['end node']))

# Extract distances as a dictionary
distances = {tuple(row): data['arc length (miles)'][i] for i, row in data[['start node', 'end node']].iterrows()}


nodes = set(data['start node']).union(set(data['end node']))
# Use nodes, arcs, and distances from the loaded data
hospital = [node for node in nodes if node.startswith("H")]
contractors = [node for node in nodes if node.startswith("C")]
demand_nodes = [node for node in nodes if node.startswith("R")]
disposal_nodes = [node for node in nodes if node.startswith("D")]
nodes = hospital + contractors + demand_nodes + disposal_nodes
depots = hospital + contractors

# Load node demands

node_demand_data = pd.read_csv('Node_Demands.csv')

# Load depot capacities
depot_capacity_data = pd.read_csv('Depot_Capacities.csv')

# Create dictionaries for demands and capacities
demand = {row['node']: row['demand'] for _, row in node_demand_data.iterrows()}
vehicle_capacity = {row['depot']: row['capacity'] for _, row in depot_capacity_data.iterrows()}
operating_cost = {row['depot']: row['operating_cost'] for _, row in depot_capacity_data.iterrows()}
cost_per_miles = {row['depot']: row['cost_per_miles'] for _, row in depot_capacity_data.iterrows()}
# Create model
model = Model("Refined_VRP_with_Load_and_MultiDepots")

# Decision variables
# x[i,j,d] = 1 if a vehicle from depot d travels from node i to node j
x = model.addVars(arcs, depots, vtype=GRB.BINARY, name="x")

# y[d] = 1 if depot d is used
y = model.addVars(depots, vtype=GRB.BINARY, name="y")

# u[n,d] = load on the route from depot d after visiting node n
u = model.addVars(nodes, depots, vtype=GRB.CONTINUOUS, lb=0.0, name="u")

# Objective: Minimize total distance and operating cost
model.setObjective(
    quicksum(distances[i, j] * x[i, j, d] * cost_per_miles[d] for (i, j) in arcs for d in depots) +
    quicksum(operating_cost[d] * y[d] for d in depots),
    GRB.MINIMIZE
)

# Each demand node is served exactly once (sum over all depots)
for r in demand_nodes:
    model.addConstr(
        quicksum(x[i, r, d] for (i, j) in arcs for d in depots if j == r) == 1,
        name=f"DemandSatisfied_{r}"
    )

# If a depot is used, exactly one route starts from it
for d in depots:
    model.addConstr(
        quicksum(x[d, j, d_] for (i, j) in arcs for d_ in depots if i == d and d_ == d) == y[d],
        name=f"DepotOutflow_{d}"
    )

# Flow conservation for each demand node and depot
for n in demand_nodes:
    for d in depots:
        model.addConstr(
            quicksum(x[i, n, d] for (i, j) in arcs if j == n) ==
            quicksum(x[n, j, d] for (i, j) in arcs if i == n),
            name=f"FlowConservation_{n}_{d}"
        )

# Ensure that all routes end at the disposal node
model.addConstr(
    quicksum(x[i, 'D1', d] for (i, j) in arcs for d in depots if j == 'D1') ==
    quicksum(y[d] for d in depots),
    name="FlowIntoDisposal"
)

# Load constraints: Ensure correct load propagation and respect for vehicle capacity
for d in depots:
    for (i, j) in arcs:
        if j in demand_nodes:
            # Load transitions if arc (i, j) is used
            model.addConstr(
                u[j, d] >= u[i, d] + demand[j] * x[i, j, d] - vehicle_capacity[d] * (1 - x[i, j, d]),
                name=f"LoadLB_{i}_{j}_{d}"
            )
            model.addConstr(
                u[j, d] <= u[i, d] + demand[j] * x[i, j, d] + vehicle_capacity[d] * (1 - x[i, j, d]),
                name=f"LoadUB_{i}_{j}_{d}"
            )

    # Capacity constraints: no node can have load greater than the depot capacity if depot is not used
    for n in nodes:
        model.addConstr(
            u[n, d] <= vehicle_capacity[d] * y[d],
            name=f"Capacity_{n}_{d}"
        )

# Depot load initialization
for d in depots:
    for dep in depots:
        model.addConstr(u[d,dep] == 0, name=f"DepotLoad_{d}")

# Arcs-to-Load linking: If no arc leads into a node for depot d, then u[node, d] must be zero
for d in depots:
    # For demand and disposal nodes:
    for n in demand_nodes + disposal_nodes:
        # If no incoming arc for depot d, load must be zero
        model.addConstr(
            u[n, d] <= vehicle_capacity[d] * quicksum(x[i, n, d] for (i, j) in arcs if j == n),
            name=f"LoadExistsIfArcs_{n}_{d}"
        )

# Final load at disposal node matches total serviced demand
for d in depots:
    model.addConstr(
        u['D1', d] == quicksum(demand[j] * x[j, 'D1', d] for (i, j) in arcs if j == 'D1' and j in demand_nodes),
        name=f"FinalLoad_{d}"
    )

# Solve the model
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    print("Optimal solution found.")
    for var in model.getVars():
        if var.x > 1e-6:
            print(f"{var.varName} = {var.x}")
else:
    print("No optimal solution found. Model status:", model.status)
    if model.status == GRB.INFEASIBLE:
        model.computeIIS()
        model.write("infeasible.ilp")
        print("IIS written to infeasible.ilp")
