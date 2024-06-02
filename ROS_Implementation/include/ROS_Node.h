#ifndef ROS_NODE2
#define ROS_NODE2
#include "lib.h"

// publishers and subscribers
ros::Publisher      pub_marker_opt_traj , which_map_pub , array_imag_pub , part_map_pub , opt_traj_pub;
ros:: Subscriber     map_emb_sub , global_path_sub , global_map_sub , position_sub;
// ROS messages
visualization_msgs::Marker      opt_traj_mk , init_gues_mk , filterd_path_mk , obst_cells_mk;

nav_msgs::OccupancyGrid         part_map;
std_msgs::Int32MultiArray array_image;
bool path_stored = false , potential_rcieved = false;;
//visualization_msgs::MarkerArray markers;


nav_msgs::Path                  path_msg;
geometry_msgs:: PoseStamped     path_msg_poses;
geometry_msgs::Pose             path_msg_pose;

geometry_msgs::Point            opt_traj_pt , init_gues_pt , filtere_path_pt , obst_cells_pt;
std_msgs::Int32 nn;
std_msgs::Float64MultiArray origin_sub;
MPC_Planner mpc;
X0 x0 , x1;
double x_origin , y_origin;
OptTraj opt_traj;
using namespace std;

//double x_ref[4] = {1.3 , 2.34, 3.34, 3.3};
//double y_ref[4] = {1.76, 1.46, 2.16, 4.};
//double x_ref[3] = {1, 2.8, 3.6};
//double y_ref[3] = {4, 3.3, 0.8};


double x_ref[3] = {48.1, 49.8, 51};
double y_ref[3] = {-8.36, -9, -13};

//double x_ref[3] = {51 , 51.6 , 54.4};
//double y_ref[3] = {-12 , -14.3 , -14.7};

double x_ref_submap[10];
double y_ref_submap[10];

double x_submap , y_submap;

double get_yaw(geometry_msgs::Quaternion q);



void initialize_markers();
void publish_local_path(OptTraj traj);
void define_pub_sub(ros::NodeHandle n);




#endif
