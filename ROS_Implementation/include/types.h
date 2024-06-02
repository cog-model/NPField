#ifndef DATA_TYPES2
#define DATA_TYPES2

#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"
#include "acados/ocp_nlp/ocp_nlp_constraints_bgh.h"
#include "acados/ocp_nlp/ocp_nlp_cost_ls.h"
#include "blasfeo/include/blasfeo_d_aux.h"
#include "blasfeo/include/blasfeo_d_aux_ext_dep.h"
#include "robot_model_model/robot_model_model.h"
#include "acados_solver_robot_model.h"
#include <vector>

using std::vector;
struct OptTraj
{
    bool   is_solved;
    vector<double> x_p, y_p;
    double x[ROBOT_MODEL_N+1];
    double y[ROBOT_MODEL_N+1];
    double v[ROBOT_MODEL_N+1];
    double theta[ROBOT_MODEL_N+1];
    double time_solving;
    double cost;
    double a[ROBOT_MODEL_N] , w[ROBOT_MODEL_N];
};

struct X0
{
    double x;
    double y;
    double v;
    double theta;
};

struct Yref
{
    double x[ROBOT_MODEL_N];
    double y[ROBOT_MODEL_N];
    double v[ROBOT_MODEL_N];
    double theta[ROBOT_MODEL_N];
};

struct Yref_e
{
    double x;
    double y;
    double v;
    double theta;
};

#endif
