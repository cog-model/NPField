## NPField: Neural Potential Field for Obstacle-Aware Local Motion Planning

### Abstarct
For the problem of local path planning with Model Predictive Control (MPC) algorithm, a neural network model that returns a differentiable collision cost based on robot pose, obstacle map, global path, and robot footprint is suggested. The proposed approach provides path for different robots footprints, without needing detection obstacles stage, and with safe distance from obstacles.

## Prerequisites
- Python3.9 or above, Pytorch, cuda and all other libraries for Acados and L4CasADi
- Install [Acados](https://github.com/acados/acados) and make sure that it works by testing examples in exampls/acados_python
- Install [L4CasADi](https://github.com/Tim-Salzmann/l4casadi)

### Steps of running the algorithm:
- Training the neural model written in file model_nn.py
- Runing the file create_solver.py will create slover for MPC local planning problem for a differential-drive mobile robot
- Use test_solver.py for testing the results of algorithm

### Demonstration video:
The code is tested on Ubuntu 20.04. Video for testing the proposed algorithm on Unmanned Ground Vehicle Husky:

https://www.youtube.com/watch?v=KL3bfvUwGqs
