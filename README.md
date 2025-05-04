# MMK419: PINN Heat Conduction Mini-Project

This repository contains the full implementation and analysis of a two-part mini-project for the course **MMK419 - Modern Tools of Computational Engineering** at the University of Cyprus. The goal is to explore **Physics-Informed Neural Networks (PINNs)** for solving heat conduction problems in one and two dimensions using the **MLX** machine learning library.

## 🔍 Project Purpose

The aim of this project is to implement and compare the performance of PINNs on steady-state heat conduction problems in 1D and 2D domains. The focus is on understanding how neural networks can be trained not just from data, but directly from governing physical laws, i.e., partial differential equations (PDEs). A secondary goal is to compare the performance of different optimizers, including a novel optimizer designed specifically for PDE problems.

---

## 📌 Problems Solved

### 📘 Part 1: 1D Heat Conduction with a Heat Source

We solve the PDE:

$$
\frac{d^2T}{dx^2} = -1
$$

with Dirichlet boundary conditions:

$$
T(0) = 0, \quad T(1) = 1
$$

This is a non-linear problem, and the PINN is trained with various activation functions (`ReLU`, `tanh`, `sigmoid`, `SiLU`, and `none`) to analyze their ability to capture non-linearity. The result is compared against a finite difference (FD) solution and the exact analytical solution:

$$
T(x) = -\frac{1}{2}x^2 + \frac{3}{2}x
$$

### 📗 Part 2: 2D Steady-State Heat Conduction

We solve the 2D Laplace equation:

$$
\nabla^2 T(x, y) = 0 \quad \text{in} \quad [0,1] \times [0,1]
$$

with the following Dirichlet boundary conditions:

- \( T = 0 \) on \( x = 0 \) and \( y = 0 \)
- \( T = 1 \) on \( x = 1 \) and \( y = 1 \)

The solution is learned using a fully connected neural network and validated against a reference finite difference solution. Contour plots visualize the PINN solution, the FD solution, and the absolute error between them.

---

## 🧰 Tools and Libraries Used

- [**MLX**](https://ml-explore.github.io/mlx): Apple’s machine learning library with built-in auto-differentiation and GPU acceleration.
- **Matplotlib**: For generating plots and visualizations.
- **NumPy**: For numerical operations and reference solution generation.
- Custom Python classes and functions for:
  - Neural network modeling
  - Loss function definition
  - Training loops with optimizers
  - Finite difference solvers for validation

---

## ⚙️ Methodology

- **PINN Architecture**:
  - A fully connected MLP with 4 layers and activation functions tested per case.
  - The network is trained to minimize the residuals of the PDE and boundary conditions using MLX's autodiff.
  
- **Training Strategy**:
  - For both 1D and 2D cases, the model is trained using the Adam optimizer or a custom optimizer.
  - In 1D, experiments are performed with various activation functions to observe their impact.
  
- **Validation**:
  - Finite difference methods (FD) are implemented to compute numerical reference solutions.
  - Comparison metrics include visual plots and absolute error maps.

---

## 📊 Results

- Non-linear activations like `tanh` and `SiLU` significantly outperform linear or no activations in capturing curved profiles in 1D.
- In 2D, the PINN solution closely matches the finite difference result, with smooth contours and low error across the domain.
- Visualizations highlight the strengths and limitations of the PINN approach in low-data or high-constraint PDE settings.

---

## 📚 Citation

This project uses the **Kourkoutas optimizer**, a custom PDE-aware optimizer developed specifically for this course:

> **Kourkoutas Optimizer**  
> Created by **Dr. Stavros Kassinos**  
> Department of Mechanical and Manufacturing Engineering, University of Cyprus  
> April 2025 — Version b.0.0.1  
> A Bayesian-inspired optimizer for PDEs, incorporating stochastic noise, momentum, and tunable exploration dynamics.

---

## 👥 Authors

- Christodoulos Negkoglou  
- Eleftheria Christoforou  
- Andreas Masri Alexandrou
