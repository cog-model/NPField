from acados_template import AcadosModel
import torch

import l4casadi as l4c

from casadi import SX, DM, vertcat, sin, cos, tan, exp, if_else, pi , atan , logic_and , sqrt , fabs , atan2 , MX , DM

    
def robot_model(model_loaded):
    model_name = "robot_model"
    # yamle file paramters
    # State
    x = MX.sym('x') 
    y = MX.sym('y')   
    v = MX.sym('v')  
    theta = MX.sym('theta') 

    sym_x = vertcat(x, y, v ,theta)

    # Input
    a = MX.sym('a')
    w = MX.sym('w')
    sym_u = vertcat(a, w)

    # Derivative of the States
    x_dot = MX.sym('x_dot')
    y_dot = MX.sym('y_dot')
    theta_dot = MX.sym('theta_dot')
    v_dot = MX.sym('v_dot')
    x_dot = vertcat(x_dot, y_dot, v_dot, theta_dot)

    ## Model of Robot
    f_expl = vertcat(   sym_x[2] * cos(sym_x[3]),
                        sym_x[2] * sin(sym_x[3]),
                        sym_u[0],
                        sym_u[1])
    f_impl = x_dot - f_expl

    model = AcadosModel()

    print(model)

    l4c_model = l4c.L4CasADi(model_loaded, model_expects_batch_dim=True , name='y_expr',device='cuda')

    torch.cuda.empty_cache()

    print("l4model " , l4c_model)
  
    input_model = MX.sym('in',4288+32+32)
    sym_p = vertcat(input_model)

    model.cost_y_expr = vertcat(sym_x, sym_u , l4c_model(vertcat(sym_p,x,y,theta)))
    model.cost_y_expr_e = vertcat(sym_x, l4c_model(vertcat(sym_p,x,y,theta)))
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = sym_x
    model.xdot = x_dot
    model.u = sym_u
    model.p = sym_p
    model.name = "robot_model"

    return model , l4c_model

