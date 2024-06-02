#ifndef DATA_MPC_L4C2
#define DATA_MPC_L4C2
#include "lib.h"
#include "types.h"


class MPC_Planner 
{
   
public:
    X0 x0 ;
    Yref yref;
    Yref_e yref_e;
    OptTraj opt_traj;
    double max_length_path , start_angle_tol , start_angle , stop_solving_length;
    vector<double> x_path , y_path;
    bool print_filter , print_initial_guess , print_params , print_local_path , print_modified_filter;
    bool is_solved;
    vector<double> x_filter , y_filter , theta_filter_segments, length_filter_segments;
    double length_filter_path , v_path;
    double initial_so[ROBOT_MODEL_NX];
    double initial_co[ROBOT_MODEL_NU];
    double initial_gu[ROBOT_MODEL_NY];
    double initial_gu_terminal[ROBOT_MODEL_NYN];
    double solver_params[ROBOT_MODEL_NP];
    double x_out[ROBOT_MODEL_NX] , u_out[ROBOT_MODEL_NU];

       
    
    int N = ROBOT_MODEL_N;
    int NP = ROBOT_MODEL_NP;

    MPC_Planner();
    
    OptTraj get_path();
    
    void filter_path();
       
    void initial_guess();
    
    void initialize_solver(robot_model_solver_capsule *acados_ocp_capsule);
    
    void set_x0(robot_model_solver_capsule *acados_ocp_capsule);
    
    void set_yref(robot_model_solver_capsule *acados_ocp_capsule);
    
    void set_initial_solution(robot_model_solver_capsule *acados_ocp_capsule);
    
    void set_parameters(robot_model_solver_capsule *acados_ocp_capsule);
    
    OptTraj solve_problem(robot_model_solver_capsule *acados_ocp_capsule);
    
    void free_solver(robot_model_solver_capsule *acados_ocp_capsule);
    
};

#endif
