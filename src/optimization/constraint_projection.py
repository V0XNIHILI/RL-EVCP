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
        p_distance = (p_val - model.p[d_ind])**2
        #NOTE(Frans): Voltage has no influence on our actual reward so don't put it in the objective
        # v_distance = distance(v_val, model.v[d_ind])
        model.distance_to_solution.append(p_distance)
        # model.distance_to_solution.append(v_distance)
    model.f = Objective(sense=minimize, expr=sum(sqrt(model.distance_to_solution)))

    if lossless:
        solver = SolverFactory('glpk')
    else:
        solver = SolverFactory('ipopt')
    try:
        solver.solve(model, tee=tee)
        new_p = dict_to_matrix(model.p, model.devices.data()) / 1000
        new_v = dict_to_matrix(model.v, model.devices.data())
    except ValueError:
        return p, v, None
    return new_p, new_v, model


def project_constraints_ev(p_EV, EV_devices, u_t, p_lbs_t, p_ubs_t, v_lbs_t, v_ubs_t, conductance_matrix, i_max_matrix,
                             lossless=False, tee=False, iterations=100):

    p_EV, p_lbs_t, p_ubs_t = p_EV * 1000, p_lbs_t * 1000, p_ubs_t * 1000  # Transform powers to W from  kW

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
        p_i = -v_i * sum([conductance_matrix[i, j] * (model.v[i] - model.v[j]) for j in model.devices if i != j])
        model.power_balance.add(model.p[i] == p_i)

    # Line currents
    model.line_constraints = ConstraintList()
    for i in model.devices:
        for j in model.devices:
            if conductance_matrix[i, j] > 0:
                i_line = conductance_matrix[i, j] * (model.v[i] - model.v[j])
                i_line_max = i_max_matrix[i, j]
                model.line_constraints.add(inequality(-i_line_max, i_line, i_line_max))

    # Constraint Projection EV power target
    model.distance_to_solution = []
    for (p_val, d_ind) in zip(p_EV, EV_devices):
        p_distance = (p_val - model.p[d_ind])**2
        model.distance_to_solution.append(p_distance)

    # Objective for all other non EV devices
    model.per_device_utility = []
    for d_ind in model.devices:
        if d_ind not in EV_devices:
            val = u_t[d_ind] * model.p[d_ind]
            model.per_device_utility.append(val)

    # model.distance_to_solution.append(v_distance)
    model.f = Objective(sense=maximize, expr=-sum(sqrt(model.distance_to_solution)) + sum(model.per_device_utility))

    if lossless:
        solver = SolverFactory('glpk') #, executable='E:\\Boeken\\Jaar 5\\Q4 Project\\winglpk-4.55\\glpk-4.55\\w64\\glpsol')
    else:
        solver = SolverFactory('ipopt') #, executable='E:\\Boeken\\Jaar 5\\Q4 Project\\Ipopt-3.11.1-win64-intel13.1\\Ipopt-3.11.1-win64-intel13.1\\bin\\ipopt')
        solver.options['max_iter'] = iterations  # number of iterations you wish

    solver.solve(model, tee=tee)

    new_p = dict_to_matrix(model.p, model.devices.data()) / 1000 # convert from w to kw
    new_v = dict_to_matrix(model.v, model.devices.data())

    return new_p, new_v, model


def distance(val1, val2):
    # Euclidean and Manhattan distance is the same in 1d
    return abs(val1 - val2)
















