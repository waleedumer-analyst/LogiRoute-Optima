import pandas as pd
from pyomo.environ import *

def optimize_freight_network(shipment_data_path):
    df = pd.read_csv(shipment_data_path)
    model = ConcreteModel(name="LogiRoute_Network_Optimization")
    model.LANES = Set(initialize=df['lane_id'].tolist())
    model.x = Var(model.LANES, domain=NonNegativeReals)
    
    costs = df.set_index('lane_id')['freight_cost_per_fcl'].to_dict()
    capacities = df.set_index('lane_id')['max_teu_capacity'].to_dict()
    demands = df.set_index('lane_id')['required_demand_teu'].to_dict()
    
    def objective_rule(model):
        return sum(model.x[i] * costs[i] for i in model.LANES)
    model.MinimizeCost = Objective(rule=objective_rule, sense=minimize)
    
    def capacity_constraint_rule(model, i):
        return model.x[i] <= capacities[i]
    model.CapacityConstraint = Constraint(model.LANES, rule=capacity_constraint_rule)
    
    def demand_constraint_rule(model, i):
        return model.x[i] >= demands[i]
    model.DemandConstraint = Constraint(model.LANES, rule=demand_constraint_rule)
    
    solver = SolverFactory('cbc')
    results = solver.solve(model, tee=False)
    
    output_records = []
    for i in model.LANES:
        output_records.append({
            'Lane_ID': i,
            'Optimal_Allocated_Volume_TEU': value(model.x[i]),
            'Total_Allocated_Cost_USD': value(model.x[i]) * costs[i]
        })
        
    return pd.DataFrame(output_records)

if __name__ == "__main__":
    print("Optimization computational layer parsed successfully.")
