"""
Simulates advection-diffusion in Fourier space using the same methodology as Chen and Kraichnan 2018.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dedalus.public as d3
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


def gaussian(grid, mu_x, mu_y, var_x, var_y):
    dim = len(grid)
    output_grid = np.zeros((grid[0].shape[0], grid[1].shape[1]))
    for i in range(output_grid.shape[0]):
        for j in range(output_grid.shape[1]):
            x, y = grid[0][i, 0], grid[1][0, j]
            output_grid[i][j] = np.exp(-0.5*(((x-mu_x)**2)/var_x + ((y-mu_y)**2)/var_y))
    return output_grid
      

# Constants
kappa = 0.4
Nx, Ny = 5, 5
Lx, Ly = 1000, 1000 
dealias = 3/2
timestepper = d3.RK222
stop_sim_time = 50

# initialize variables
coords = d3.CartesianCoordinates('x','y')
dist = d3.Distributor(coords, dtype=np.float64)
xbasis = d3.RealFourier(coords['x'], size=Lx, bounds=(-Nx, Nx), dealias=dealias)
ybasis = d3.RealFourier(coords['y'],size=Ly, bounds=(-Ny,Ny), dealias=dealias)

theta = dist.Field(name='theta',bases=(xbasis,ybasis))
theta_y = dist.Field(name='theta_y', bases=(xbasis, ybasis))
tau1 = dist.Field(name='tau1', bases=(xbasis))
tauf = dist.VectorField(coords,name='tauf', bases=(xbasis,ybasis))
v_0 = dist.VectorField(coords,name='v_0',bases=(xbasis,ybasis))
v_1 = dist.VectorField(coords,name='v_1',bases=(xbasis,ybasis))
t = dist.Field()

# scalar field at t = 0 is the standard normal gaussian mutiplied by a constant 
grid = dist.local_grids(xbasis, ybasis)
theta['g'] = gaussian(grid, 0, 0, 1, 1)
theta['g'] *= 2

# build solver
problem = d3.IVP([theta],time=t, namespace=locals())
problem.add_equation("dt(theta) - kappa * lap(theta) = - dot(v, grad(theta))")
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# time step loop
timestep = 0.005
theta.change_scales(1)
theta_list = [np.copy(theta['g'])]
time_list = [solver.sim_time]
while solver.proceed:
    solver.step(timestep)
    # Update velocity field every time step. Kraichnan paper doesn't say exactly to do this
    v_0['g'] = np.roll(v_0['g'], shift=1, axis=1)
    v_1['g'] = np.roll(v_1['g'], shift=1, axis=2)
    v = (v_0 + v_1).evaluate()
    if solver.iteration % 50 == 0:
        theta.change_scales(1)
        theta_list.append(np.copy(theta['g']))
        time_list.append(solver.sim_time)
        print(f"Iteration {solver.iteration}")


# Save data into a gif
fig = plt.figure()
im = plt.imshow(theta_list[0])
plt.colorbar()
def update(frame):
    print(frame)
    im.set_array(theta_list[frame])
    return [im] 

ani = animation.FuncAnimation(fig=fig, func=update, frames=len(theta_list), interval=100)
ani.save(filename='../gifs/randomdynamicfield.gif',writer='pillow')