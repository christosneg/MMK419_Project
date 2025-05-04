"""
===============================================================================
1D Steady Heat Conduction (T''(x) = 0) with MLX

Author: Stavros Kassinos
Course: MME419
Date:  [March, 2025
Version: 1.0

DESCRIPTION:
------------
This script solves the 1D heat conduction equation (Laplace's equation) on the
unit interval [0,1], enforcing boundary conditions T(0)=0 and T(1)=1.

1) PINN Approach:
   - A small MLP predicts T(x).def __init__(self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
   - We sample interior points to penalize ∂²T/∂x² = 0 (the PDE).
   - We sample boundary points to enforce T(0)=0, T(1)=1 (Dirichlet BCs).
   - We train via MLX's autodiff and optimizer (e.g., Adam or Lamb).

2) Finite Difference (FD):
   - We also solve ∂²T/∂x² = 0 with T(0)=0, T(1)=1 by iterating an explicit solver
     until approximate steady-state is reached.
   - We compare the FD solution, the PINN solution, and the exact solution T(x)=x.

===============================================================================
"""

import mlx.core as mx      # Primary MLX library
import mlx.nn as nn        # MLX library on Neural Nets
import mlx.optimizers as optim  # MLX library for optimizers
from Kourkoutas_optimizer import FullySchedulableKourkoutasWithMomentum
from functools import partial   #
import matplotlib.pyplot as plt

mx.random.seed(42)

def solve_with_pinn(activation):
    # ------------------------------------------------------------------------------
    # 1) Define a small MLP for 1D input (x) -> 1D output (T).
    # ------------------------------------------------------------------------------
    class MLP(nn.Module):
        """
        A fully-connected MLP to approximate T(x).
        MLP = Multi Layer Perception

        Args:
            num_layers (int): Number of layers in the network.
            input_dim (int): Dimension of the input (1 for [x] in 1D).
            hidden_dim (int): Number of neurons in each hidden layer.
            output_dim (int): Dimension of the output (1 for [T]).

        The layers are stacked linearly with tanh activations between them,
        except for the final layer which is a direct output (no activation).
        """

        def __init__(self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int, activationFcn):
            super().__init__()
            self.activationFcn = activationFcn
            self.layers = []
            if num_layers < 1:
                raise ValueError("num_layers must be >= 1")

            # If there's only one layer, it directly connects input to output.
            if num_layers == 1:
                self.layers.append(nn.Linear(input_dim, output_dim))
            else:
                # First layer from input_dim -> hidden_dim
                self.layers.append(nn.Linear(input_dim, hidden_dim))
                # Middle hidden layers
                for _ in range(num_layers - 2):
                    self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                # Final layer from hidden_dim -> output_dim
                self.layers.append(nn.Linear(hidden_dim, output_dim))

        def __call__(self, x: mx.array) -> mx.array:
            """
            Forward pass through the network.

            Args:
                x (mx.array): shape (N,1) or (1,) representing x-values

            Returns:
                mx.array: shape (N,1) or (1,) representing the predicted T(x)
            """
            # For each layer except the last, apply tanh activation
            for layer in self.layers[:-1]:
                #x = nn.tanh(layer(x))
                x = self.activationFcn(layer(x))
                #x = nn.silu(layer(x))   # Pass through activation
                # x = layer(x)  #Pass without activation
            # The last layer is linear (no activation)
            return self.layers[-1](x)


    # Instantiate the PINN model (4 layers, 32 hidden neurons each)
    mlp_model = MLP(num_layers=4, input_dim=1, hidden_dim=32, output_dim=1, activationFcn=activation)

    # MLX uses lazy evaluation. So for instantiation to take effect we need to evaluate it
    mx.eval(mlp_model.parameters())



    # ------------------------------------------------------------------------------
    # 2) Single-x version of T (scalars only).
    # ------------------------------------------------------------------------------
    def T_fn_scalar(x_scalar: mx.array) -> mx.array:
        """
        Evaluate the MLP at a single x, returning a scalar T(x).

        We reshape x into (1,1) so that the MLP sees a batch of size 1.
        Then we return out[0,0], a single scalar.

        Args:
            x_scalar (mx.array): shape () or shape(1,)

        Returns:
            mx.array: shape (), the predicted T(x).
        """
        x_reshaped = x_scalar.reshape((1, 1))
        out = mlp_model(x_reshaped)  # shape (1,1)
        return out[0, 0]  # shape (), a single scalar



    # ------------------------------------------------------------------------------
    # 3) Single-x second derivative (T''(x)).
    # ------------------------------------------------------------------------------
    def second_derivative_T_scalar(x_scalar: mx.array) -> mx.array:
        """
        Compute T''(x) at a single scalar x using MLX auto-differentiation.

        Steps:
          1) T'(x) = derivative wrt x of T_fn_scalar(x)
          2) T''(x) = derivative wrt x of T'(x)

        This returns a single scalar T''(x).

        Args:
            x_scalar (mx.array): shape () or shape(1,)

        Returns:
            mx.array: shape (), T''(x).
        """
        # First derivative T'(x)
        #dTdx = mx.grad(T_fn_scalar)(x_scalar)

        # Second derivative T''(x): derivative wrt x of that first derivative
        def first_deriv_fn(z):
            return mx.grad(T_fn_scalar)(z)

        d2Tdx2 = mx.grad(first_deriv_fn)(x_scalar)
        return d2Tdx2


    # ------------------------------------------------------------------------------
    # 4) Prepare domain sampling for PDE & Boundary Conditions
    #     Define collocation points
    # ------------------------------------------------------------------------------
    N_interior = 64
    N_boundary = 2  # x=0 and x=1

    # Random interior points in (0,1):
    x_interior = mx.random.uniform(0, 1, (N_interior,))  # shape (N,)
    # Evaluate T''(x) on these interior points (vectorized via vmap)
    #T_d2 = mx.vmap(second_derivative_T_scalar)(x_interior)
    #loss_interior = mx.mean(T_d2 ** 2)

    # Boundary points: x=0, x=1
    x_boundary = mx.array([[0.0], [1.0]])  # shape (2,1)
    T_bc = mx.array([0.0, 1.0])            # T(0)=0, T(1)=1


    # ------------------------------------------------------------------------------
    # 5) Define the PINN loss function
    # ------------------------------------------------------------------------------
    def loss_fn(x_interior, x_boundary):
        """
        Combine:
          - PDE residual loss: T''(x)=0 for x in the interior
          - Boundary condition loss: T(0)=0, T(1)=1

        We compute each, then return their sum.

        Args:
            x_interior (mx.array): shape (N,) interior points
            x_boundary (mx.array): shape (2,1) boundary points [0.0, 1.0]
        Returns:
            mx.array: scalar total loss
        """
        # PDE residual: T''(x) = 0 => penalty is (T''(x))^2
        T_d2 = mx.vmap(second_derivative_T_scalar)(x_interior)  # shape (N,)
        loss_interior = mx.mean((T_d2+1) ** 2)

        # Boundary: T(0)=0, T(1)=1 => penalty is (T(x_boundary)-T_bc)^2
        T_b = mx.vmap(T_fn_scalar)(x_boundary)  # shape (2,)
        loss_boundary = mx.mean((T_b - T_bc) ** 2)

        return loss_interior + loss_boundary


    # ------------------------------------------------------------------------------
    # 6) PINN Training Loop
    # ------------------------------------------------------------------------------
    optimizer = optim.Adam(learning_rate=1e-3)
    '''
    optimizer = FullySchedulableKourkoutasWithMomentum(
        learning_rate=0.0001,
        sand_temperature=0.2,
        desert_haze=0.0005,
        sunbathing=0.9999,
        rock_bottom=0.11,  # 0.1999,
        rock_ceiling=1e2,
        grad_clip=1e2,
        eps=9e-4,
        beta_m=0.001
    )
    '''
    num_epochs = 10_000

    # We gather the "state" for MLX: model params, optimizer state, RNG, domain arrays.
    state = [mlp_model.state, optimizer.state, mx.random.state, x_interior, x_boundary]
    mx.eval(state)

    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(x_interior, x_boundary):
        """
        Single training step for the PINN:
          1) Evaluate the PDE + boundary loss via loss_fn().
          2) Compute gradients w.r.t. model parameters.
          3) Return (loss, grads) for the optimizer to update.

        Returns:
            (mx.array, dict): (loss, grads)
        """
        # We extract states from "state" (though MLX may also handle them implicitly).
        (mlp_model_state, optimizer.state, mx.random.state, x_interior, x_boundary) = state

        loss_and_grad_fn = nn.value_and_grad(mlp_model, loss_fn)
        loss_val, grads = loss_and_grad_fn(x_interior, x_boundary)
        return loss_val, grads

    print("Starting PINN training...")

    for epoch in range(num_epochs):
        loss_val, grads = train_step(x_interior, x_boundary)
        # The optimizer updates model parameters
        optimizer.update(mlp_model, grads)

        # Evaluate new param states
        mx.eval(mlp_model.parameters(), optimizer.state)

        # Optional learning-rate schedule
        if epoch == 2000:
            optimizer.learning_rate = 1e-4

        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch + 1}, Loss = {float(loss_val):.6e}")


    # ------------------------------------------------------------------------------
    # 7) Evaluate PINN on a uniform grid
    # ------------------------------------------------------------------------------
    x_test = mx.linspace(0, 1, 101).reshape((-1, 1))  # shape (101,1)
    T_pinn = mx.vmap(T_fn_scalar)(x_test)            # shape (101,)
    mx.eval(T_pinn)
    return T_pinn

# ------------------------------------------------------------------------------
# 8) Solve the same PDE with an FD approach for comparison
# ------------------------------------------------------------------------------
@mx.compile
def calculate_spatial_derivative_1D_initial(T: mx.array, dx: float) -> mx.array:
    """
    Vectorized second-order spatial derivative (d²T/dx²) in 1D using central diffs.

    T''(x_i) ~ (T[i+1] - 2*T[i] + T[i-1]) / dx² for i in [1..nx-2]

    Args:
        T (mx.array): shape (nx,), temperature at each of nx points
        dx (float): spacing between consecutive points in [0,1]

    Returns:
        mx.array: shape (nx-2,), second derivative at the interior points
    """
    # Indices: 1..nx-2 are interior
    d2T_dx2 = (T[2:] - 2 * T[1:-1] + T[:-2]) / (dx ** 2)
    return d2T_dx2


@mx.compile
def solve_heat_conduction_fd_1D(nx: int, num_steps: int, alpha: float) -> mx.array:
    """
    Solve T''(x)=0 on x in [0,1] with T(0)=0, T(1)=1 using an explicit Euler step.

    We treat:
       ∂T/∂t = alpha * ∂²T/∂x²
    and let it converge to steady state. The final T(x) should match T(x)=x.

    Args:
        nx (int): # grid points
        num_steps (int): # of time steps
        alpha (float): "diffusivity" controlling how fast T evolves

    Returns:
        mx.array: shape (nx,) final T distribution after num_steps
    """
    dx = 1.0 / (nx - 1)
    dt = 0.25 * dx**2 / alpha  # explicit Euler time step for stability

    # Initialize T=0 except for boundary points
    T = mx.zeros((nx,))
    T[0]   = 0.0  # left boundary
    T[-1]  = 1.0  # right boundary

    for step in range(num_steps):
        d2T_dx2 = calculate_spatial_derivative_1D_initial(T, dx)
        T_new   = mx.array(T)
        mx.eval(T_new)  # ensure T_new is "materialized" in MLX

        # Update interior region [1..nx-2]
        T_new[1:-1] = T[1:-1] + alpha * dt * (d2T_dx2+1)

        # Re-impose boundaries
        T_new[0]  = 0.0
        T_new[-1] = 1.0

        T = T_new

        if divmod(step, 500)[1] == 0:
            print(f"FD step = {step}")

    return T

def none(x):
    return x

# ------------------------------------------------------------------------------
# 9) Compare FD solution, PINN solution, and exact T(x)=x
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    nx = 101
    num_steps = 30000
    alpha = 0.1

    # FD solve
    fd_solution = solve_heat_conduction_fd_1D(nx, num_steps, alpha)

    # Convert T_pinn to np array for plotting
    x_grid = np.linspace(0, 1, nx)

    colors = ['r', 'g', 'b', 'c', 'm']
    none.__name__="No"
    index = 0
    for activation in [nn.silu, nn.tanh, nn.sigmoid, nn.relu, none]:
        plt.plot(x_grid, solve_with_pinn(activation), label= activation.__name__+' activation', color = colors[index])
        index+=1
    plt.plot(x_grid, fd_solution, label='FD', color = 'y')

    # Compare with exact T(x)=x
    T_exact = (-1/2)*x_grid**2+(3/2)*x_grid
    plt.plot(x_grid, T_exact, 'k:', label='Exact: x')

    plt.xlabel('x')
    plt.ylabel('T(x)')
    plt.title("1D Heat Conduction: PINN vs. FD vs. Exact")
    plt.legend()
    plt.grid(True)
    plt.show()