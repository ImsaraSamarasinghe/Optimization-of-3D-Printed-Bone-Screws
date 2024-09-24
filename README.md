# Bone Screw Topology Optimization

This project implements a topology optimization routine using the Finite Element Method (FEM) for a bone screw. The code leverages the Firedrake and IPOPT (via `cyipopt`) libraries to solve for the optimal material distribution of the screw material under a load, based on the Solid Isotropic Material with Penalization (SIMP) model. The output is the optimized material distribution and an animation of the optimization process.

![](https://github.com/ImsaraSamarasinghe/Optimization-of-3D-Printed-Bone-Screws/blob/main/P%3D1500.png)

## Features

- **Firedrake** FEM-based simulation.
- **SIMP Model**: Used to interpolate between void and material to optimize structure design.
- **IPOPT Optimization**: Uses IPOPT to solve the nonlinear optimization problem.
- **Bone screw Simulation**: Optimizes material layout to support the given load.
- **Visualization**: Outputs animation and final visual of the optimized structure.

## Dependencies

Ensure the following Python libraries are installed:

- **Firedrake**: Install via the [Firedrake website](https://www.firedrakeproject.org/)
- **cyipopt**: Install using pip

    ```bash
    pip install cyipopt
    ```

- **NumPy**: Install via pip

    ```bash
    pip install numpy
    ```

- **Matplotlib**: Install for plotting

    ```bash
    pip install matplotlib
    ```

## Problem Description

The goal is to optimize the layout of material in a cantilever beam to minimize its compliance (maximize stiffness) under a given load. The SIMP interpolation method is used to determine the relationship between material density and stiffness. The problem includes:

- **Elasticity Problem**: Modeled using linear elasticity equations.
- **Objective Function**: Compliance (stiffness) of the structure.
- **Constraints**: Volume constraint limiting the amount of material used.
- **Density Filter**: Smooths the design to avoid checkerboard patterns.

## Code Overview

### `cantilever` Class

The `cantilever` class implements the following methods:
- `HH_filter()`: Applies a filter to smooth the material density field.
- `sigma()`: Computes the stress tensor using Young's modulus and Poisson's ratio.
- `epsilon()`: Computes the strain tensor.
- `objective()`: Computes the objective function (compliance) of the structure.
- `gradient()`: Computes the gradient of the objective function with respect to the material density.
- `constraints()`: Volume constraint for the optimization problem.
- `jacobian()`: Computes the Jacobian matrix of the constraints.

### `main()`

The `main` function sets up the problem, defines the boundary conditions, and creates the IPOPT optimization problem. After solving, the optimized design is visualized and saved.

## How to Run

1. **Install dependencies** as described above.
   
2. **Run the code** by executing the following command in your terminal:

   ```bash
   python cantilever_optimization.py
   ```

3. **View results**: After the optimization completes, the final material distribution is saved as an image and written to a `.pvd` file for visualization in Paraview.

## Parameters

- **Domain Size**: Length (L) and width (W) of the cantilever beam.
- **Mesh Size**: `nx` and `ny` define the number of elements in the mesh.
- **Material Properties**: `E_max` (Young's modulus of the material), `nu` (Poisson's ratio), and `p` (penalization factor for the SIMP model).
- **Volume Fraction**: The fraction of the design domain allowed to contain material.
- **Filter Radius (`r_min`)**: Radius for density filtering to smooth the design.

## Visualization

After solving, the optimized cantilever beam design is displayed as a grayscale plot, where the intensity of each region corresponds to the material density.

- **Animation**: During optimization, the intermediate solutions are written to a `.pvd` file, which can be visualized using Paraview.
- **Final Design**: The final material distribution is saved as a PNG image and can also be visualized using Paraview.

## Example Output

The output files include:
- **Optimized Beam Image**: A PNG image showing the optimized material layout.
- **Animation File**: A `.pvd` file that can be opened in Paraview to view the optimization process.

## Future Improvements

- Extend the code to handle 3D structures.
- Include multiple load cases.
- Implement more advanced filtering techniques to further improve the solution.
