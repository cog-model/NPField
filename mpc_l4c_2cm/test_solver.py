import torch
import json
from matplotlib import colors , transforms
import sys
import time
import json
import matplotlib.ticker as ticker
from json import JSONEncoder
import l4casadi as l4c

import matplotlib.pyplot as plt
import pickle
import create_solver
from math import cos , sin

import numpy as np
import math
from model_nn import Autoencoder_path

def test_solver(acados_solver , x_ref_points , y_ref_points , theta_0 , num_map , ax1 , map_inp):

    

    nx = 4
    nu = 2
    ny = nx + nu
    N = 30
    yref = np.zeros([N,ny+1])


    theta_e = 0
    v_0 = 0
    v_e = 0

    x_ref = []
    y_ref = []
    theta = []
    theta_ref = []
    init_x = []
    init_y = []
    init_theta = []
    len_segments = []
    theta = np.append(theta , theta_0 ) # current orientation robot
    theta_ref = np.append(theta_ref , theta_0 )
    num_segment = len(x_ref_points)-1
    length_path = 0
    for i in range(num_segment):
        length_path = length_path + math.sqrt((x_ref_points[i+1]-x_ref_points[i])**2+(y_ref_points[i+1]-y_ref_points[i])**2)
        theta = np.append(theta , math.atan2(y_ref_points[i+1]-y_ref_points[i], x_ref_points[i+1]-x_ref_points[i]))
        len_segments = np.append(len_segments , math.sqrt((x_ref_points[i+1]-x_ref_points[i])**2+(y_ref_points[i+1]-y_ref_points[i])**2))

    step_line = length_path / N

    print("length path",length_path)

    v_max = 0.2

    new_time_step = ((length_path)/(v_max*N)) * np.ones(N)

    print(new_time_step)

    k = 0
    x_ref = np.append(x_ref , x_ref_points[0])
    y_ref = np.append(y_ref , y_ref_points[0]) 
    for i in range(N+1):
        x_ref = np.append(x_ref , x_ref[i] + step_line * math.cos(theta[k+1]))
        y_ref = np.append(y_ref , y_ref[i] + step_line * math.sin(theta[k+1]))
        theta_ref = np.append(theta_ref , theta[k+1])
        d = math.sqrt((x_ref[-1]-x_ref_points[k])**2+(y_ref[-1]-y_ref_points[k])**2)
        if(d>len_segments[k] and k<(num_segment-1)):
            k = k+1
            x_ref[i] = x_ref_points[k]
            y_ref[i] = y_ref_points[k]
        elif (k>(num_segment-1)):
            break
    x0 = np.array([x_ref_points[0],y_ref_points[0],v_0,theta_0])

    init_x = x_ref[0:N+1]
    init_y = y_ref[0:N+1]
    init_theta = theta_ref[0:N+1]
    x_goal = np.array([init_x[-1],init_y[-1], v_e,init_theta[-1]])

    yref[:,0]=init_x[0:N]
    yref[:,1]=init_y[0:N]
    yref[:,2] = v_max
    yref[:,3] = init_theta[0:N]

    a = np.zeros(1)

    result = model_loaded.encode_map_footprint(map_inp).detach()
    result2 = torch.clamp(result, min=0.)
    print("result shape ", result.shape)

    paramters_static = [100] * 4352

    for i in range(4352):
        paramters_static[i] = result[0,i].cpu().data.numpy()

    parameter_values = np.concatenate([paramters_static])

    yref_e = np.concatenate([x_goal,a])  
    x_traj_init = np.transpose([ yref[:,0] , yref[:,1] , yref[:,2], yref[:,3]])


    simX = np.zeros((N+1, 3))
    simU = np.zeros((N, nu))

    for i in range(N):
        acados_solver.set(i,'p',parameter_values)
        acados_solver.set(i,'y_ref',yref[i])
        acados_solver.set(i, 'x', x_traj_init[i])
        acados_solver.set(i, 'u', np.array([0.0, 0.0]))
    acados_solver.set(N, 'p',  parameter_values)
    acados_solver.set(N, 'y_ref', yref_e)
    acados_solver.set(N, 'x', x_goal)
    acados_solver.set(0,'lbx', x0)
    acados_solver.set(0,'ubx', x0)
    acados_solver.options_set('rti_phase', 0)
    acados_solver.set_new_time_steps(new_time_step)

    t = time.time()
    status = acados_solver.solve()
    print("status" , status)
    elapsed = 1000 * (time.time() - t)
    print("Elapsed time: {} ms".format(elapsed))
    print("optimal path")
    ROB_x = np.zeros([N+1,9])
    ROB_y = np.zeros([N+1,9])
    for i in range(N + 1):
        x = acados_solver.get(i, "x")
     #   print(x[0],", ",x[1] , ",")
        simX[i,0]=x[0]
        simX[i,1]=x[1]
        simX[i,2]=x[3]
        ROB_x[i,0] = simX[i,0] + 0.6 * cos(simX[i,2]-0.59)
        ROB_x[i,1] = simX[i,0] + 0.514 * cos(simX[i,2]-0.24)
        ROB_x[i,2] = simX[i,0] + 0.75 * cos(simX[i,2]-0.16)
        ROB_x[i,3] = simX[i,0] + 0.75 * cos(simX[i,2]+0.16)
        ROB_x[i,4] = simX[i,0] + 0.514 * cos(simX[i,2]+0.24)
        ROB_x[i,5] = simX[i,0] + 0.6 * cos(simX[i,2]+0.59)
        ROB_x[i,6] = simX[i,0] - 0.6 * cos(simX[i,2]-0.59)
        ROB_x[i,7] = simX[i,0] - 0.6 * cos(simX[i,2]+0.59)
        ROB_x[i,8] = simX[i,0] + 0.6 * cos(simX[i,2]-0.59)
        ROB_y[i,0] = simX[i,1] + 0.6 * sin(simX[i,2]-0.59)
        ROB_y[i,1] = simX[i,1] + 0.514 * sin(simX[i,2]-0.24)
        ROB_y[i,2] = simX[i,1] + 0.75 * sin(simX[i,2]-0.16)
        ROB_y[i,3] = simX[i,1] + 0.75 * sin(simX[i,2]+0.16)
        ROB_y[i,4] = simX[i,1] + 0.514 * sin(simX[i,2]+0.24)
        ROB_y[i,5] = simX[i,1] + 0.6 * sin(simX[i,2]+0.59)
        ROB_y[i,6] = simX[i,1] - 0.6 * sin(simX[i,2]-0.59)
        ROB_y[i,7] = simX[i,1] - 0.6 * sin(simX[i,2]+0.59)
        ROB_y[i,8] = simX[i,1] + 0.6 * sin(simX[i,2]-0.59)

    
    # for i in range(31):
    #     path_mpc[i,0] = path_mpc[i,0] * 100/5.12
    #     path_mpc[i,1] = 100 - path_mpc[i,1] * 100/5.12
    print("initial path")
    
    for i in range(N + 1):
        initial_path[i,0] = init_x[i]
        initial_path[i,1] = init_y[i]
        initial_path[i,2] = init_theta[i]
        print(initial_path[i,0],", ",initial_path[i,1] , ", ", initial_path[i,2])

       
    for i in range(N):
        u = acados_solver.get(i, "u")
        simU[i,:]=u
    print("status" , status)
    cost = acados_solver.get_cost()
    print("cost", cost)

    
    if (num_map ==-1):
        ax1.plot(simX[:, 0], simX[:, 1] , linewidth=4 , marker='o')
        ax1.plot(init_x , init_y, marker='o')
        ax1.set_aspect('equal', 'box')
        ax1.plot([3.5,5.1,5.1,3.5,3.5],[1.52,1.52,3.12,3.12,1.52] , linewidth=2 )
        ax1.plot([1.8,2.6,2.6,1.8,1.8],[3.92,3.92,4.72,4.72,3.92] , linewidth=2 )
        ax1.plot([0.6,2,2,0.6,0.6],[1.12,1.12,2.52,2.52,1.12] , linewidth=2 )
        plt.xlabel("x")
        plt.ylabel("y")
    else:
        for i in range(N+1):
            simX[i, 0] = simX[i, 0] * 50
            simX[i, 1] = simX[i, 1] * 50
            init_x[i]  = init_x[i] * 50
            init_y[i]  = init_y[i] * 50
            ROB_x[i,0] = ROB_x[i,0] * 50
            ROB_x[i,1] = ROB_x[i,1] * 50
            ROB_x[i,2] = ROB_x[i,2] * 50
            ROB_x[i,3] = ROB_x[i,3] * 50
            ROB_x[i,4] = ROB_x[i,4] * 50
            ROB_x[i,5] = ROB_x[i,5] * 50
            ROB_x[i,6] = ROB_x[i,6] * 50
            ROB_x[i,7] = ROB_x[i,7] * 50
            ROB_x[i,8] = ROB_x[i,8] * 50
            ROB_y[i,0] = ROB_y[i,0] * 50
            ROB_y[i,1] = ROB_y[i,1] * 50
            ROB_y[i,2] = ROB_y[i,2] * 50
            ROB_y[i,3] = ROB_y[i,3] * 50
            ROB_y[i,4] = ROB_y[i,4] * 50
            ROB_y[i,5] = ROB_y[i,5] * 50
            ROB_y[i,6] = ROB_y[i,6] * 50
            ROB_y[i,7] = ROB_y[i,7] * 50
            ROB_y[i,8] = ROB_y[i,8] * 50

        
       # ax1.xaxis.set_visible(False)
       # ax1.yaxis.set_visible(False)
        ax1.plot(simX[:, 0], simX[:, 1] , linewidth=2 , label="NPField path" )#'NPField path')  # marker='o',
        ax1.plot(init_x , init_y, linestyle='dashed', linewidth=2 , label='Initial path')
        ax1.legend()
        scale_x = 50
        scale_y = 50
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_x))
        ax1.xaxis.set_major_formatter(ticks_x)

        ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_y))
        ax1.yaxis.set_major_formatter(ticks_y)
        #ax1.set_xlabel("x , meters")
        #ax1.set_ylabel("y , meters")

    # for i in range(N):
       # ax1.plot([ROB_x[i,0],ROB_x[i,1],ROB_x[i,2],ROB_x[i,3],ROB_x[i,0]], [ROB_y[i,0],ROB_y[i,1],ROB_y[i,2],ROB_y[i,3],ROB_y[i,0]], color='k')
      # ax1.plot([ROB_x[i,0],ROB_x[i,1],ROB_x[i,2],ROB_x[i,3],ROB_x[i,4],ROB_x[i,5],ROB_x[i,6],ROB_x[i,7],ROB_x[i,8]], [ROB_y[i,0],ROB_y[i,1],ROB_y[i,2],ROB_y[i,3],ROB_y[i,4],ROB_y[i,5],ROB_y[i,6],ROB_y[i,7],ROB_y[i,8]], color='k')
    
    path_mpc = simX
    


   
   

    return path_mpc , parameter_values , elapsed

def fill_map_inp(num , map , fprint):
    map_inp_1 = torch.zeros((131072))
    k = 0
    for i in range (256):
        for j in range(256):
            if(num==-1):
                map_inp_1[k] = map[i,j]
            else:
                map_inp_1[k] = map[num][i,j]
            if(map_inp_1[k]==10):
                map_inp_1[k] = 0
            k = k + 1

    k = 0
    for i in range (256):
        for j in range(256):
            map_inp_1[65536+k] = fprint[i,j]
            k = k +1
    return map_inp_1


######## load CNN Model
device = torch.device("cuda")
model_loaded = Autoencoder_path(mode="k")
model_loaded.to(device)


load_check = torch.load("1000maps_zones_husky.pth")

model_dict = model_loaded.state_dict()
pretrained_dict = {k: v for k, v in load_check.items() if k in model_dict}
model_dict.update(pretrained_dict) 
model_loaded.load_state_dict(model_dict)
model_loaded.eval();

##### create solver
acados_solver = create_solver.create_solver()

############## Load Datasets to test the created solver ###########
data_sub_maps_1000 = pickle.load( open( "dataset_sub_maps.pkl", "rb" ))
d_footprint = pickle.load( open( "dataset_footprints.pkl", "rb" ))
map = data_sub_maps_1000['sub_maps']
footprint = d_footprint['footprint_husky']

###################### Reference path #######################


N = 30
path_mpc = np.zeros((N+1, 3))
initial_path = np.zeros((N+1 , 3))

####### case 1 : sub maps 1000 num 2
x_ref_points = np.array([50 , 82,  82])*5.12/256
y_ref_points = np.array([30 , 102.5 ,  206])*5.12/256
theta_0 = 1.57
num_map = 2
map_inp = fill_map_inp(num_map , map , footprint)
fig, ax3 = plt.subplots(figsize=(5.12,5.12))
path_mpc , parameters , elapsed = test_solver(acados_solver, x_ref_points , y_ref_points , theta_0, num_map , ax3 , map_inp)
if(num_map != -1):
    cmap = colors.ListedColormap(['white' , 'black'])
    ax3.pcolor(map[num_map][::-1],cmap=cmap,edgecolors='w', linewidths=0.1)

######### case 2 : sub maps 1000 num 3
x_ref_points = np.array([0.8 ,1.24, 2.56])
y_ref_points = np.array([4.6 , 3.5 , 2.2])
theta_0 = -1.57
num_map = 3
map_inp = fill_map_inp(num_map , map , footprint)
fig, ax4 = plt.subplots(figsize=(5.12,5.12))
path_mpc, parameters, elapsed = test_solver(acados_solver, x_ref_points , y_ref_points , theta_0, num_map , ax4 , map_inp)
if(num_map != -1):
    cmap = colors.ListedColormap(['white' , 'black'])
    ax4.pcolor(map[num_map][::-1],cmap=cmap,edgecolors='w', linewidths=0.1)

######### case 3 : sub maps 1000 num 4
x_ref_points = np.array([65 , 117 , 167 , 165])*5.12/256
y_ref_points = np.array([88 , 73 , 108 , 200])*5.12/256
theta_0 = 0
num_map = 4
map_inp = fill_map_inp(num_map , map , footprint)
fig, ax5 = plt.subplots(figsize=(5.12,5.12))
path_mpc , parameters, elapsed = test_solver(acados_solver, x_ref_points , y_ref_points , theta_0, num_map , ax5 , map_inp)
if(num_map != -1):
   cmap = colors.ListedColormap(['white' , 'black'])
   ax5.pcolor(map[num_map][::-1],cmap=cmap,edgecolors='w', linewidths=0.1)

######### case 4 : sub maps 1000 num 8
x_ref_points = np.array([200 , 225 , 210])*5.12/256
y_ref_points = np.array([100 , 200 , 230])*5.12/256
theta_0 = 1.56
num_map = 8
map_inp = fill_map_inp(num_map , map , footprint)
fig, ax6 = plt.subplots(figsize=(5.12,5.12))
path_mpc, parameters, elapsed = test_solver(acados_solver, x_ref_points , y_ref_points , theta_0, num_map , ax6 , map_inp)
if(num_map != -1):
    cmap = colors.ListedColormap(['white' , 'black'])
    ax6.pcolor(map[num_map][::-1],cmap=cmap,edgecolors='w', linewidths=0.1)

plt.show()
