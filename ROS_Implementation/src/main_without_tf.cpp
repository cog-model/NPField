#include "MPC_Planner.h"
#include "ROS_Node.h"
#include <iostream>
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

void initialize_markers()
{
    opt_traj_mk.points.clear();
    opt_traj_mk.header.frame_id = "local_map_lidar";
    opt_traj_mk.header.stamp = ros::Time::now();
    opt_traj_mk.ns = "opt_traj_mk" ;
    opt_traj_mk.action = visualization_msgs::Marker::ADD;;
    opt_traj_mk.pose.orientation.w = 1.0;
    opt_traj_mk.type = 4;
    opt_traj_mk.scale.x = 0.05;
    opt_traj_mk.scale.y = 0.05;
    opt_traj_mk.color.a = 1.0;
    opt_traj_mk.color.r = 0.0;
    opt_traj_mk.color.g = 0.0;
    opt_traj_mk.color.b = 1.0;
}

void publish_local_path(OptTraj traj)
{
    opt_traj_mk.points.clear();
    path_msg.header.stamp = ros::Time::now();
    path_msg.poses.clear();
    path_msg.header.frame_id = "local_map_lidar";

    for(int i=0 ; i <= mpc.N ; i++)
    {
        path_msg_pose.position.x = traj.x[i]+ x_origin;
        path_msg_pose.position.y = traj.y[i]+ y_origin;
        path_msg_pose.orientation.x = 0;
        path_msg_pose.orientation.y = 0;
        path_msg_pose.orientation.z = 0; 
        path_msg_pose.orientation.w = 0;
        path_msg_poses.pose = path_msg_pose;
        path_msg_poses.header.frame_id = "local_map_lidar";
        path_msg_poses.header.stamp = ros::Time::now();
        path_msg.poses.push_back(path_msg_poses);
        
        opt_traj_pt.x = traj.x[i]+ x_origin;
        opt_traj_pt.y = traj.y[i]+ y_origin;
        opt_traj_pt.z = 0;
        opt_traj_mk.points.push_back(opt_traj_pt);
    }
    pub_marker_opt_traj.publish(opt_traj_mk);
    opt_traj_pub.publish(path_msg);
}
void map_emb_back(const std_msgs::Float64MultiArray& msg)
{
    for(int i = 0; i<mpc.NP; i++)
    {
        mpc.solver_params[i] = msg.data[i];
    }
    potential_rcieved = true;
}

void position_back(const std_msgs::Float32MultiArray& msg)
{
    x0.x = msg.data[0];
    x0.y = msg.data[1];
    x0.theta = msg.data[2];
}

void planner_back(const nav_msgs::Path msg) 
{
 //   ROS_INFO("Global Path Recieved");
    int size_path = msg.poses.size();
    if(size_path != 0)
    {   
        mpc.x_path.clear();
        mpc.y_path.clear();
        mpc.x_path.push_back(x0.x);
        mpc.y_path.push_back(x0.y); 
        ROS_INFO(" %f , %f" , mpc.x_path[0] , mpc.y_path[0]);             
        for (int j=0; j<size_path ; j++)
        {   
            mpc.x_path.push_back(msg.poses[j].pose.position.x);
            mpc.y_path.push_back(msg.poses[j].pose.position.y);
         //   ROS_INFO(" %f , %f" , mpc.x_path[j+1] , mpc.y_path[j+1]);
        }
        path_stored= true;
    }
}

void global_map_back(const nav_msgs::OccupancyGrid& msg)
{
        ROS_WARN("x_origin %f" , x_origin);
        part_map.data.clear();
        array_image.data.clear();
        part_map.header.stamp = ros::Time::now();
        part_map.header.frame_id = "local_map_lidar";
        part_map.info.width = 50;
        part_map.info.height = 50;
        part_map.info.resolution = 0.1;
        part_map.info.origin.position.x = x_origin;
        part_map.info.origin.position.y = y_origin;
        for (int i =int(1177*(10*abs(-25.3 + abs(y_origin)))) ; i<int(1177*(10*abs(-25.3 + abs(y_origin)))+58850); i++)  //10*abs(y_origin)
        {
            if(i%1177 >= int(10*(x_origin+20.4)) && i%1177 <int(10*(x_origin+20.4)+50))
            {
                if(msg.data[i]!=-1)
                {
                    part_map.data.push_back(msg.data[i]);
                    array_image.data.push_back(msg.data[i]);
                }
                else
                {
                    part_map.data.push_back(msg.data[i]);
                    array_image.data.push_back(100);
                }
            }
        }
        part_map_pub.publish(part_map);
        array_imag_pub.publish(array_image);
}


void define_pub_sub(ros::NodeHandle n)
{
  //  which_map_pub    = n.advertise<std_msgs::Int32>       ("which_map",   1);

    pub_marker_opt_traj     = n.advertise<visualization_msgs::Marker> ("mpc_planner/mk_local_path", 10);

    opt_traj_pub            = n.advertise<nav_msgs::Path>             ("path_mpcl4c/local_path",        10);

  //  pub_origin_submap       = n.advertise<std_msgs::Float64MultiArray> ("mpc_planner/origin_submap", 10);

    global_path_sub     = n.subscribe("path",    10, planner_back);

    map_emb_sub   = n.subscribe("map_footprint_emb",     10, map_emb_back);

    array_imag_pub    = n.advertise<std_msgs::Int32MultiArray>       ("image",   5);

    part_map_pub    = n.advertise<nav_msgs::OccupancyGrid>       ("part_map",   5);

    global_map_sub      = n.subscribe("grid_map",     10, global_map_back);

    position_sub      = n.subscribe("position",     10, position_back);
}

/*void publish_origin_submap(double x_origin , double y_origin)
{
  origin_sub.data.clear();
  origin_sub.data.push_back(x_origin);
  origin_sub.data.push_back(y_origin);
  pub_origin_submap.publish(origin_sub);

}*/


int main(int argc, char **argv)
{
    ros::init(argc, argv, "mpc_l4c_planner");

    ros::NodeHandle n;

    define_pub_sub(n);

    initialize_markers();

    mpc.start_angle_tol = 0.6;
    mpc.max_length_path = 2;
    mpc.stop_solving_length = 1;
    mpc.v_path = 0.2;
    // mpc.print_local_path = true;
    mpc.print_filter = false;
    mpc.is_solved = false;
    
    mpc.print_initial_guess = false;

    tf2_ros::Buffer tfBuffer;

    tf2_ros::TransformListener tfListener(tfBuffer);

    ros::Rate loop_rate(4); 

    
    
    while(ros::ok())
    {
        auto start_loop = high_resolution_clock::now();


        ROS_INFO("x y %f %f" , x0.x , x0.y);
        
      //  x0.x = x_ref[0];
      //  x0.y = y_ref[0];
        x0.v = 0;

        

        if(path_stored)
        {
            auto start = std::chrono::steady_clock::now();
            /*mpc.x_path.clear();
            mpc.y_path.clear();
            for (int i=0 ; i<3 ; i++)
            {
                mpc.x_path.push_back(x_ref[i]);
                mpc.y_path.push_back(y_ref[i]); 
            }*/
            mpc.x0 = x0;
            mpc.filter_path();
            int size_filter_path = mpc.x_filter.size();
            for (int i=0 ; i<mpc.x_filter.size(); i++)
            {
                x_ref_submap[i] = mpc.x_filter[i];
                y_ref_submap[i] = mpc.y_filter[i];
            }
            x_submap = *min_element(mpc.x_filter.begin(), mpc.x_filter.end())-1;
            y_submap = *min_element(mpc.y_filter.begin(), mpc.y_filter.end())-1;
      //      ROS_INFO("length filtered path %f" , mpc.length_filter_path);
            
            x_origin = x_submap;
            y_origin = y_submap;
            ROS_INFO(" origin submap %f %f" , x_origin , y_origin);
            mpc.x_filter.clear();
            mpc.y_filter.clear();

            // convert filtered path to submap
            for (int i=0 ; i<size_filter_path; i++)
            {
                mpc.x_filter[i] = x_ref_submap[i] - x_submap; 
                mpc.y_filter[i] = y_ref_submap[i] - y_submap; 
            }
            x1.x = x0.x - x_submap;
            x1.y = x0.y - y_submap;
            x1.v = 0;
            x1.theta = x0.theta;
       //     ROS_INFO("x1 %f %f ", x1.x , x1.y);
         //   publish_origin_submap(mpc.x_submap , mpc.y_submap);
            mpc.x0 = x1;
            mpc.x_path.clear();
            mpc.y_path.clear();
       //     ROS_INFO("theta 0 %f" , x1.theta);
            for (int i=0 ; i<size_filter_path ; i++)
            {
                mpc.x_path.push_back(x_ref_submap[i] - x_submap);
                mpc.y_path.push_back(y_ref_submap[i] - y_submap); 
              //  ROS_INFO("filter path %f %f" , x_ref_submap[i] - x_submap , y_ref_submap[i] - y_submap);
            }
            mpc.filter_path();

            opt_traj = mpc.get_path();
            publish_local_path(opt_traj);
          //  nn.data++;
            potential_rcieved = false;

            
            auto end = std::chrono::steady_clock::now();
            auto dur = end - start;
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
            std::cout << "loop time: "<< ms << " -- solver time: "<< opt_traj.time_solving<< "\n";
            
            path_stored = false;
        }
        // plt::plot(opt_traj.x_p , opt_traj.y_p);

        // plt::axis("square");
        // plt::xlim(0.0 ,5.12);
        // plt::ylim(0.0, 5.12);
        // plt::show();

        ros::spinOnce();
        loop_rate.sleep();
    }
    return 0;
}


