#include "MPC_Planner.h"
#include <iostream>
using namespace std;
MPC_Planner::MPC_Planner()
{
    for(int i=0; i<ROBOT_MODEL_NP; i++)
    {
        solver_params[i] = 0;
    }
}

OptTraj MPC_Planner::get_path()
{
    robot_model_solver_capsule *acados_ocp_capsule = robot_model_acados_create_capsule();
    int status = robot_model_acados_create(acados_ocp_capsule);
    cout<<status<<endl;
    initial_guess();
    initialize_solver(acados_ocp_capsule);
    set_x0(acados_ocp_capsule);
    set_yref(acados_ocp_capsule);
    set_initial_solution(acados_ocp_capsule);
    set_parameters(acados_ocp_capsule);
    opt_traj = solve_problem(acados_ocp_capsule);
    free_solver(acados_ocp_capsule);
    return opt_traj;
}
  
void MPC_Planner::filter_path()
{
    x_filter.clear();
    y_filter.clear();

    theta_filter_segments.clear();
    length_filter_segments.clear();

    x_filter.push_back(x_path[0]);
    y_filter.push_back(y_path[0]);

    double len_path = 0 , old_len = 0;

    for(int i=1; i<x_path.size() ; i++)
    {
        len_path = len_path + sqrt(pow((x_path[i]-x_path[i-1]),2)+pow((y_path[i]-y_path[i-1]),2));
        if(len_path == max_length_path)
        {
            x_filter.push_back(x_path[i]);
            y_filter.push_back(y_path[i]);
            break;
        }
        if(len_path < max_length_path)
        {
            x_filter.push_back(x_path[i]);
            y_filter.push_back(y_path[i]);
        }
        else if(len_path > max_length_path)
        {
            x_filter.push_back(x_path[i-1] + (max_length_path-old_len)*cos(atan2(y_path[i]-y_path[i-1], x_path[i]-x_path[i-1])));
            y_filter.push_back(y_path[i-1] + (max_length_path-old_len)*sin(atan2(y_path[i]-y_path[i-1], x_path[i]-x_path[i-1])));
            break;
        }
        old_len = len_path;
    }

    length_filter_path = 0;

    for(int i=0 ; i<(x_filter.size()-1);i++)
    {
        length_filter_segments.push_back(sqrt(pow((x_filter[i+1]- x_filter[i]),2) + pow((y_filter[i+1]- y_filter[i]),2)));

        length_filter_path = length_filter_path + sqrt(pow((x_filter[i+1]- x_filter[i]),2) + pow((y_filter[i+1]- y_filter[i]),2));

        theta_filter_segments.push_back(atan2(y_filter[i+1]-y_filter[i], x_filter[i+1]-x_filter[i]));
    }

    if(print_filter)
    {
        ROS_INFO("THERE ARE %i FILTERED POINTS, LENGTH FILTERED PATH IS %f m" , x_filter.size() , length_filter_path);
        for(int i=0; i<theta_filter_segments.size(); i++)
        {
            ROS_INFO("FILTERED PATH %f %f theta %f length segment %f" , x_filter[i] , y_filter[i] , theta_filter_segments[i] , length_filter_segments[i]);
        }
        ROS_INFO("last FILTERED Point %f %f" , x_filter[x_filter.size()-1] , y_filter[x_filter.size()-1]);

        ROS_INFO(" Length Filtered Path %f m " , length_filter_path);
    }
    

}

void MPC_Planner::initial_guess()
{
    double step_line = length_filter_path / N;
    cout<<"step line "<< step_line<<endl;

    yref.x[0] = x_filter[0];
    yref.y[0] = y_filter[0];
    yref.v[0] = v_path;

    start_angle = std::abs(x0.theta - theta_filter_segments[0]);

    if(start_angle > start_angle_tol)
    {
        yref.theta[0] = theta_filter_segments[0];
        x0.theta = theta_filter_segments[0];
    }
    else
    {
        yref.theta[0] = x0.theta;
    }

    if(yref.theta[0]<=M_PI && yref.theta[0]>= 0.9 && x0.theta>-(2*M_PI) && x0.theta<=-0.9)
    {
        yref.theta[0] = -2*M_PI + yref.theta[0];
    }
    else if(yref.theta[0]>-M_PI && yref.theta[0]<= -0.9 && x0.theta<=(2*M_PI) && x0.theta>=0.9)
    {
        yref.theta[0] = 2*M_PI + yref.theta[0];
    }

    int k = 0;

    int num_segments = x_filter.size() - 1;
    for (int i=1; i<N ; i++)
    {
        yref.x[i] = yref.x[i-1] + step_line * cos(theta_filter_segments[k]);
        yref.y[i] = yref.y[i-1] + step_line * sin(theta_filter_segments[k]);
        yref.theta[i] = theta_filter_segments[k];

        
        double d = sqrt(pow((yref.x[i]-x_filter[k]),2)+pow((yref.y[i]-y_filter[k]),2));
        if(d > length_filter_segments[k] && k <= num_segments)
        {
            k++;
            yref.x[i] = x_filter[k] + (d - length_filter_segments[k-1]) * cos(theta_filter_segments[k]);
            yref.y[i] = y_filter[k] + (d - length_filter_segments[k-1]) * sin(theta_filter_segments[k]);
            yref.theta[i] = theta_filter_segments[k];
        }
        else if (k > num_segments)
            break;

        if(yref.theta[i] <=M_PI && yref.theta[i]>= 0.9 && yref.theta[i-1]>-(2*M_PI) && yref.theta[i-1]<=-0.9)
        {
            yref.theta[i] = -2*M_PI + yref.theta[i];
        }
        else if(yref.theta[i]>-M_PI && yref.theta[i]<= -0.9 && yref.theta[i-1]<=(2*M_PI) && yref.theta[i-1]>=0.9)
        {
            yref.theta[i] = 2*M_PI + yref.theta[i];
        }

        yref.v[i] = v_path;
    }

    yref.theta[N-1] = yref.theta[N-2];

    yref_e.x = x_filter.back();
    yref_e.y = y_filter.back();
    if(x_filter.back()==x_path.back() &&  y_filter.back()==y_path.back())
    {
        yref_e.v = 0;
    }
    else
    {
        yref_e.v = v_path;
    }
    
    yref_e.theta = yref.theta[N-1];
     

    if(print_initial_guess)
    {
      //  ROS_INFO("INITIAL POSITION X0 %i %f %f %f %f" , 0 , x0.x , x0.y , x0.v , x0.theta);
        cout<< 0 << " "<< x0.x <<" "<< x0.y <<endl;

        cout<<"INITIAL PATH "<<endl;
        for(int i=0; i<N; i++)
        {
         //   ROS_INFO("INITIAL PATH Yref   %i %f %f %f %f" , i , yref.x[i] , yref.y[i] , yref.v[i] , yref.theta[i]);
            cout<< i << " "<< yref.x[i]<<" "<< yref.y[i] <<endl;
        }

      //  ROS_INFO("INITIAL PATH Yref_e %i %f %f %f %f" , N , yref_e.x , yref_e.y , yref_e.v , yref_e.theta);
        cout<< N << " "<< yref_e.x<<" "<< yref_e.y <<endl;
    }
}

void MPC_Planner::initialize_solver(robot_model_solver_capsule *acados_ocp_capsule)
{
    double new_time_steps[N];
    double time_path = length_filter_path / v_path ; 
    for (int i=0; i<N ; i++)
    {
        new_time_steps[i] = time_path/N ; 
    }

    // set a new time step for the solver
    int status1 = robot_model_acados_create_with_discretization(acados_ocp_capsule, N, new_time_steps);
    if (status1) {
        std::stringstream error;
        error << "robot_model_acados_create_with_discretization() returned status: " << status1 << ". Exiting.";
      //  ROS_ERROR_STREAM(error.str());
        throw std::runtime_error{error.str()};
    }
}

void MPC_Planner::set_x0(robot_model_solver_capsule *acados_ocp_capsule)
{
    int nx = ROBOT_MODEL_NX , nu = ROBOT_MODEL_NU;
    ocp_nlp_config *nlp_config = robot_model_acados_get_nlp_config(acados_ocp_capsule);
    ocp_nlp_dims *nlp_dims = robot_model_acados_get_nlp_dims(acados_ocp_capsule);
    ocp_nlp_in *nlp_in = robot_model_acados_get_nlp_in(acados_ocp_capsule);

    int idxbx0[nx];
    idxbx0[0] = 0;
    idxbx0[1] = 1;
    idxbx0[2] = 2;
    idxbx0[3] = 3;

    double lbx0[nx];
    double ubx0[nx];
    lbx0[0] = x0.x;
    ubx0[0] = x0.x;
    lbx0[1] = x0.y;
    ubx0[1] = x0.y;
    lbx0[2] = x0.v;
    ubx0[2] = x0.v;
    lbx0[3] = x0.theta;
    ubx0[3] = x0.theta;

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "idxbx", idxbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "ubx", ubx0);
}

void MPC_Planner::set_yref(robot_model_solver_capsule *acados_ocp_capsule)
{
    ocp_nlp_config *nlp_config = robot_model_acados_get_nlp_config(acados_ocp_capsule);
    ocp_nlp_dims *nlp_dims = robot_model_acados_get_nlp_dims(acados_ocp_capsule);
    ocp_nlp_in *nlp_in = robot_model_acados_get_nlp_in(acados_ocp_capsule);
    ocp_nlp_out *nlp_out = robot_model_acados_get_nlp_out(acados_ocp_capsule);

    for (int i = 0; i < N; i++)
    {
        initial_gu[0] = yref.x[i];
        initial_gu[1] = yref.y[i];
        initial_gu[2] = yref.v[i];
        initial_gu[3] = yref.theta[i];
        initial_gu[4] = 0;
        initial_gu[5] = 0;
        
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "yref", initial_gu);
    }
    
    initial_gu_terminal[0] = yref_e.x;  
    initial_gu_terminal[1] = yref_e.y;
    initial_gu_terminal[2] = yref_e.v;
    initial_gu_terminal[3] = yref_e.theta;

    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "yref", initial_gu_terminal);  
}

void MPC_Planner::set_initial_solution(robot_model_solver_capsule *acados_ocp_capsule)
{
    ocp_nlp_config *nlp_config = robot_model_acados_get_nlp_config(acados_ocp_capsule);
    ocp_nlp_dims *nlp_dims = robot_model_acados_get_nlp_dims(acados_ocp_capsule);
    ocp_nlp_in *nlp_in = robot_model_acados_get_nlp_in(acados_ocp_capsule);
    ocp_nlp_out *nlp_out = robot_model_acados_get_nlp_out(acados_ocp_capsule);
    if(is_solved)
    {
        for (int i = 0; i < N; i++)
        {
            initial_so[0] = opt_traj.x[i];
            initial_so[1] = opt_traj.y[i];
            initial_so[2] = opt_traj.v[i];
            initial_so[3] = opt_traj.theta[i];

            initial_co[0] = opt_traj.a[i];
            initial_co[1] = opt_traj.w[i];
            
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "x", initial_so);
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "u", initial_co);
        }
        initial_so[0] = opt_traj.x[N];
        initial_so[1] = opt_traj.y[N];
        initial_so[2] = opt_traj.v[N];
        initial_so[3] = opt_traj.theta[N];
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, N, "x", initial_so);
    }
    else
    {
        for (int i = 0; i < N; i++)
        {
            initial_so[0] = yref.x[i];
            initial_so[1] = yref.y[i];
            initial_so[2] = yref.v[i];
            initial_so[3] = yref.theta[i];

            initial_co[0] = 0;
            initial_co[1] = 0;
            
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "x", initial_so);
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "u", initial_co);
        }
        initial_so[0] = yref_e.x;
        initial_so[1] = yref_e.y;
        initial_so[2] = yref_e.v;
        initial_so[3] = yref_e.theta;
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, N, "x", initial_so);
    }
}

void MPC_Planner::set_parameters(robot_model_solver_capsule *acados_ocp_capsule)
{
    ocp_nlp_dims *nlp_dims = robot_model_acados_get_nlp_dims(acados_ocp_capsule);
    /*if(print_params)
    {
        for(int i=0 ; i<40 ; i+=4)
        {
            ROS_INFO("Solver Parameters %i %f %f %f %f", i , solver_params[i], solver_params[i+1] , solver_params[i+2] , solver_params[i+3]);
        }
    }*/
    for(int i=0 ; i<=N ; i++)
    {
        robot_model_acados_update_params(acados_ocp_capsule , i , solver_params , ROBOT_MODEL_NP);
    } 
}

OptTraj MPC_Planner::solve_problem(robot_model_solver_capsule *acados_ocp_capsule)
{
    ocp_nlp_config *nlp_config = robot_model_acados_get_nlp_config(acados_ocp_capsule);
    ocp_nlp_dims *nlp_dims = robot_model_acados_get_nlp_dims(acados_ocp_capsule);
    ocp_nlp_out *nlp_out = robot_model_acados_get_nlp_out(acados_ocp_capsule);
    ocp_nlp_in *nlp_in = robot_model_acados_get_nlp_in(acados_ocp_capsule);
    ocp_nlp_solver *nlp_solver = robot_model_acados_get_nlp_solver(acados_ocp_capsule);
    void *nlp_opts = robot_model_acados_get_nlp_opts(acados_ocp_capsule); 
    int rti_phase = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "rti_phase", &rti_phase);
    int status3 = robot_model_acados_solve(acados_ocp_capsule);
    if (status3 == 0 || status3 == 2)
    { 
        opt_traj.is_solved = true;
        ocp_nlp_get(nlp_config, nlp_solver, "time_tot", &opt_traj.time_solving);
        for (int i = 0; i <= N; i++)            
        {
            ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, i, "x", (void *)x_out);
            ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, i, "u", (void *)u_out);
         //   opt_traj.x_p.clear();
         //   opt_traj.y_p.clear();
          //  opt_traj.x.at(i) = x_out[0];
          //  opt_traj.y.at(i) = x_out[1];
            opt_traj.x_p.push_back(x_out[0]);
            opt_traj.y_p.push_back(x_out[1]);
            opt_traj.x[i] = x_out[0];
            opt_traj.y[i] = x_out[1];
            opt_traj.v[i] = x_out[2];
            opt_traj.theta[i] = x_out[3];
            opt_traj.time_solving = opt_traj.time_solving;
            if(i<N)
            {
                opt_traj.a[i] = u_out[0];
                opt_traj.w[i] = u_out[1];
            }
        }
        if(print_local_path)
        {
            for(int i=0 ; i<=N; i++)
            {
              //  ROS_INFO("Local Path %f %f %f %f " , opt_traj.x[i] , opt_traj.y[i] , opt_traj.v[i] , opt_traj.theta[i]);
               // cout<< i << " "<< opt_traj.x[i] <<" "<< opt_traj.y[i] <<endl;
               // cout<< i << " "<< opt_traj.x.at(i) <<" "<< opt_traj.y.at(i) <<endl;
            }
        }
        ocp_nlp_eval_cost(nlp_solver, nlp_in, nlp_out);
        ocp_nlp_get(nlp_config, nlp_solver, "cost_value", &opt_traj.cost);
   //     ROS_INFO("\033[1;33m-Elapsed Time %f ms and cost %f \033[0m", 1000*opt_traj.time_solving , opt_traj.cost);
    }
    else
    {
        ocp_nlp_get(nlp_config, nlp_solver, "time_tot", &opt_traj.time_solving);
        opt_traj.is_solved = false;
    }

    return opt_traj;
}

void MPC_Planner::free_solver(robot_model_solver_capsule *acados_ocp_capsule)
{
    int status4 = robot_model_acados_free(acados_ocp_capsule);
    /*if (status4) {
        ROS_ERROR("robot_model_acados_free() returned status %d. \n", status4);
    }*/
    status4 = robot_model_acados_free_capsule(acados_ocp_capsule);
    /*if (status4) {
        ROS_ERROR("robot_model_acados_free_capsule() returned status %d. \n", status4);
    }*/
}
