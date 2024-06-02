cmake_minimum_required(VERSION 3.0.2)
project(mpc_l4c_planner_10cm)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  cv_bridge
  std_msgs
  tf
  sensor_msgs
)
find_package(Python3 COMPONENTS Interpreter Development NumPy)
find_package(Torch REQUIRED)

set(acados_include "/home/vladislav/acados/include")
set(acados_lib "/home/vladislav/acados/lib")

include_directories(
include
${catkin_INCLUDE_DIRS}
${Python3_INCLUDE_DIRS}
scripts/c_generated_code
${PROJECT_SOURCE_DIR}/scripts/c_generated_code/robot_model_model
${PROJECT_SOURCE_DIR}/scripts/c_generated_code/robot_model_cost
${PROJECT_SOURCE_DIR}/scripts/c_generated_code
/home/vladislav/.local/lib/python3.9/site-packages/l4casadi/include
/home/vladislav/.local/lib/python3.9/site-packages/numpy/core/include
${acados_include}
${acados_include}/blasfeo/include
${acados_include}/hpipm/include
${acados_include}/acados/
${acados_include}/qpOASES_e
)


catkin_package(

)

add_executable(mpc_l4c_planner_10cm src/main.cpp src/MPC_Planner.cpp) 
target_link_libraries(mpc_l4c_planner_10cm
	${catkin_LIBRARIES}
	${PROJECT_SOURCE_DIR}/scripts/c_generated_code/libacados_ocp_solver_robot_model.so
	${acados_lib}/libacados.so
	/home/vladislav/catkin_ws/src/mpc_l4c_planner_10cm/scripts/_l4c_generated/liby_expr.so
	/home/vladislav/.local/lib/python3.9/site-packages/l4casadi/lib/libl4casadi.so
)

target_link_directories(mpc_l4c_planner_10cm PRIVATE /home/vladislav/.local/lib/python3.9/site-packages/l4casadi/lib)
target_link_libraries(mpc_l4c_planner_10cm l4casadi)
set(CMAKE_PREFIX_PATH /home/vladislav/.local/lib/python3.9/site-packages/torch)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_include_directories(mpc_l4c_planner_10cm PUBLIC ${TORCH_INCLUDE_DIRS})
target_link_libraries(mpc_l4c_planner_10cm ${Python3_LIBRARIES})

#target_link_libraries(mpc_l4c_planner_10cm ${TORCH_LIBRARIES})
# set_property(TARGET mpc_l4c_planner_10cm PROPERTY CXX_STANDARD 17)

#install(TARGETS mpc_planner LIBRARY DESTINATION l4casadi)
