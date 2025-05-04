#===============================================================================
# 2D Steady Heat Conduction Solver using PINN (Physics-Informed Neural Networks)
#
# Author: Christodoulos Negkoglou, Eleftheria Christoforou, Andreas Masri Alexandrou
# Course: MME419
# Date: March, 2025
# Version: 1.1
#
# DESCRIPTION:
# ------------
# Solves the steady-state heat conduction equation ∇2T = 0 (Laplace equation)
# in the 2D unit square domain [0,1]x[0,1] using a PINN approach.
#
#  - A multi-layer perceptron (MLP) models the solution T(x, y).
#  - Interior points are used to minimize the PDE residual: ∇2T = 0.
#  - Boundary points enforce Dirichlet boundary conditions.
#  - Adam optimizer is used for training.
#  - Solution is compared with a finite difference method.
#===============================================================================

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import matplotlib.pyplot as plt
import numpy as np
import math

mx.random.seed(42)
N_dimension = 32
loss_weight = 1e-4
LR = 1e-3
num_epochs = 40_000
LR_later = 1e-4
epoch_later = 15_000
LR_later2 = 1e-5
epoch_later2 = 20_000
optimizer = optim.Adam(learning_rate=LR)

#-------------------------------------------------------------------------------
# Define a fully-connected MLP model for T(x, y).
#-------------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = []
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def __call__(self, x: mx.array, y: mx.array) -> mx.array:
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        xy = mx.concatenate([x, y], axis=1)
        for layer in self.layers[:-1]:
            xy = nn.silu(layer(xy))
        return self.layers[-1](xy)

mlp_model = MLP(num_layers=4, input_dim=2, hidden_dim=64, output_dim=1)
mx.eval(mlp_model.parameters())

#-------------------------------------------------------------------------------
# Scalar evaluation wrappers
#-------------------------------------------------------------------------------
def T_fn_scalar(xy_scalar: mx.array) -> mx.array:
    xy_reshaped = xy_scalar.reshape((1, 2))
    out = mlp_model(xy_reshaped[:,0], xy_reshaped[:,1])
    return out[0, 0]

def second_derivative_T_scalar(xy_scalar: mx.array, dim: int) -> mx.array:
    def first_deriv_fn(z):
        return mx.grad(T_fn_scalar)(z)[dim]
    return mx.grad(first_deriv_fn)(xy_scalar)

#-------------------------------------------------------------------------------
# Generate training data
#-------------------------------------------------------------------------------
xy_interior = mx.random.uniform(0, 1, (N_dimension*N_dimension, 2))

x_boundary = mx.linspace(0, 1, N_dimension)
y_boundary = mx.linspace(0, 1, N_dimension)

T_bcy_bottom = mx.zeros((N_dimension,))
T_bcy_top = mx.ones((N_dimension,))
T_bcx_left = mx.zeros((N_dimension,))
T_bcx_right = mx.ones((N_dimension,))

T_bcy = mx.concatenate([T_bcy_bottom, T_bcy_top], axis=0)
T_bcx = mx.concatenate([T_bcx_left, T_bcx_right], axis=0)
T_bc = mx.concatenate([T_bcx, T_bcy], axis=0)

xy_b0 = mx.stack([x_boundary, mx.zeros_like(x_boundary)], axis=1)
xy_b1 = mx.stack([x_boundary, mx.ones_like(x_boundary)], axis=1)
xy_b2 = mx.stack([mx.zeros_like(y_boundary), y_boundary], axis=1)
xy_b3 = mx.stack([mx.ones_like(y_boundary), y_boundary], axis=1)
xy_boundary = mx.concatenate([xy_b0, xy_b1, xy_b2, xy_b3], axis=0)

#-------------------------------------------------------------------------------
# Loss function
#-------------------------------------------------------------------------------
def loss_fn(xy_interior, xy_boundary, T_bc):
    def second_derivative_x(xy): return second_derivative_T_scalar(xy, 0)
    def second_derivative_y(xy): return second_derivative_T_scalar(xy, 1)

    T_d2x = mx.vmap(second_derivative_x)(xy_interior)
    T_d2y = mx.vmap(second_derivative_y)(xy_interior)
    loss_interior = mx.mean((T_d2x + T_d2y) ** 2)

    T_b = mx.vmap(T_fn_scalar)(xy_boundary)
    loss_boundary = mx.mean((T_b - T_bc) ** 2)

    return loss_interior * loss_weight + loss_boundary

#-------------------------------------------------------------------------------
# Training loop
#-------------------------------------------------------------------------------
def train_step(xy_interior, xy_boundary, T_bc):
    loss_and_grad_fn = nn.value_and_grad(mlp_model, loss_fn)
    loss_val, grads = loss_and_grad_fn(xy_interior, xy_boundary, T_bc)
    return loss_val, grads

print("Starting PINN training...")
for epoch in range(num_epochs):
    if epoch == epoch_later:
        optimizer.learning_rate = LR_later
    elif epoch == epoch_later2:
        optimizer.learning_rate = LR_later2

    loss_val, grads = train_step(xy_interior, xy_boundary, T_bc)
    optimizer.update(mlp_model, grads)
    mx.eval(mlp_model.parameters(), optimizer.state)

    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch + 1}, Loss = {float(loss_val):.6e}, LR = {float(optimizer.learning_rate):.1e}")

#-------------------------------------------------------------------------------
# Evaluate trained model on grid
#-------------------------------------------------------------------------------
xx, yy = mx.meshgrid(mx.linspace(0, 1, N_dimension), mx.linspace(0, 1, N_dimension))
xy_test = mx.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)
T_pinn = mx.vmap(T_fn_scalar)(xy_test)
mx.eval(T_pinn)
T_pinn = T_pinn.reshape((N_dimension, N_dimension))

#-------------------------------------------------------------------------------
# Finite difference reference solution
#-------------------------------------------------------------------------------
@mx.compile
def solve_heat_conduction_fd_2D(nx: int, ny: int, num_steps: int, alpha: float) -> mx.array:
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    dt = 0.25 * min(dx**2, dy**2) / alpha

    T = mx.zeros((ny, nx))
    T[0, :]  = 0.0
    T[-1, :] = 1.0
    T[:, 0]  = 0.0
    T[:, -1] = 1.0

    for step in range(num_steps):
        T_new = mx.array(T)
        mx.eval(T_new)

        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                d2T_dx2 = (T[i, j + 1] - 2 * T[i, j] + T[i, j - 1]) / dx**2
                d2T_dy2 = (T[i + 1, j] - 2 * T[i, j] + T[i - 1, j]) / dy**2
                T_new[i, j] = T[i, j] + dt * (d2T_dx2 + d2T_dy2)

        T_new[0, :]  = 0.0
        T_new[-1, :] = 1.0
        T_new[:, 0]  = 0.0
        T_new[:, -1] = 1.0

        T = T_new

    return T

num_steps = 1_000
alpha = 1.0
T_fd = solve_heat_conduction_fd_2D(N_dimension, N_dimension, num_steps, alpha)

#-------------------------------------------------------------------------------
# Plotting
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    plt.figure(figsize=(8, 5))
    plt.contourf(np.array(xx), np.array(yy), np.array(T_pinn), cmap=plt.cm.jet, vmin=0, vmax=1)
    plt.title("PINN Solution")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.contourf(np.array(xx), np.array(yy), np.array(T_fd), cmap=plt.cm.jet, vmin=0, vmax=1)
    plt.title("FD Solution")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    T_error = mx.abs(T_pinn - T_fd)
    plt.contourf(np.array(xx), np.array(yy), np.array(T_error), cmap=plt.cm.jet, vmin=0, vmax=0.15)
    plt.title("Absolute Error |T_PINN - T_FD|")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
