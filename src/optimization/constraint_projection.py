from src.optimization.util import dict_to_matrix
from pyomo.environ import *

def project_constraints(p, v, n_devices, u_t, p_lbs_t, p_ubs_t, v_lbs_t, v_ubs_t, conductance_matrix, i_max_matrix,
                             lossless=False, tee=False):
    p_lbs_t, p_ubs_t = p_lbs_t * 1000, p_ubs_t * 1000  # Transform powers to W from  kW
    model = ConcreteModel()
    model.devices = Set(initialize=range(u_t.shape[0]))
    model.p = Var(model.devices)
    model.v = Var(model.devices)
    # Lower and upper bounds
    for d_ind in model.devices:
        model.p[d_ind].setlb(p_lbs_t[d_ind])
        model.p[d_ind].setub(p_ubs_t[d_ind])
        model.v[d_ind].setlb(v_lbs_t[d_ind])
        model.v[d_ind].setub(v_ubs_t[d_ind])
    # Power balance
    model.power_balance = ConstraintList()
    for i in model.devices:
        v_i = model.v[i] if not lossless else v_ubs_t[0]
        p_i = -v_i * sum([conductance_matrix[i, j] * (model.v[i] -  model.v[j]) for j in model.devices if i != j])
        model.power_balance.add(model.p[i] == p_i)
    # Line currents
    model.line_constraints = ConstraintList()
    for i in model.devices:
        for j in model.devices:
            if conductance_matrix[i, j] > 0:
                i_line = conductance_matrix[i, j] * (model.v[i] - model.v[j])
                i_line_max = i_max_matrix[i, j]
                model.line_constraints.add(inequality(-i_line_max, i_line, i_line_max))
    # Constraint Projection Objective
    model.distance_to_solution = []
    for (p_val, v_val, d_ind) in zip(p, v, model.devices):
        p_distance = distance(p_val, model.p[d_ind])
        v_distance = distance(v_val, model.v[d_ind])
        model.distance_to_solution.append(p_distance)
        model.distance_to_solution.append(v_distance)
    model.f = Objective(sense=minimize, expr=sum(model.distance_to_solution))

    if lossless:
        solver = SolverFactory('glpk')
    else:
        solver = SolverFactory('ipopt')
    solver.solve(model, tee=tee)
    new_p = dict_to_matrix(model.p, model.devices.data()) / 1000
    new_v = dict_to_matrix(model.v, model.devices.data())
    return new_p, new_v, model


def distance(val1, val2):
    # Euclidean and Manhattan distance is the same in 1d
    return abs(val1 - val2)

