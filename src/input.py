# Simulation mode
SIMULATION_MODE = 1     # 0: Lid driven cavity | 1: airfoil flow | 2: channel flow | 3: Heat conduction
# Mesh
READ_MESH = True
inflow_length = 0.4     # For airflow case

# Input definition
u_inf, v_inf = 1.0, 0.0
tol_inner = 1e-06
tol_outer = 1e-06
iter_outer = 2000
iter_mom = 1
iter_pp = 100

relax_uv = 0.8
relax_p = 0.1
noise = 0.02

# Flow properties
mu = 0.1
rho = 1.0

# Post-Processing
n_plot = 100
streamline_mode = True