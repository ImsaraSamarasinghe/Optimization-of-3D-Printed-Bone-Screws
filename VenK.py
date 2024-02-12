import matplotlib.pyplot as plt
from firedrake import *
from firedrake.adjoint import *
import numpy as np
continue_annotation()

mesh = UnitSquareMesh(50,50)


# function space
V = VectorFunctionSpace(mesh,'CG',1)
u_trial = TrialFunction(V)
u_test = TestFunction(V)
u = Function(V)
B = Constant([0,0]) # body force
G = Constant([5,0]) # traction

# material
E, nu = 10, 0.3
lambda_, mu = E*nu/((1-nu)*(1-2*nu)), 0.5*E/(1+nu)

# params
F = Identity(2)+grad(u)
C = F.T*F
J = det(F)
Ic = tr(C)


# dirichlet
bc1 = DirichletBC(V,Constant([0,0]),3)
bc2 = DirichletBC(V,Constant([0,0]),4)

# SED
psi = (mu/2)*(Ic-3)-mu*ln(J)+(lambda_/2)*(ln(J))**2
# Potential Energy
PI = psi*dx - dot(B,u)*dx - dot(G,u)*ds(2)

F = derivative(PI,u,u_test)
J = derivative(F,u,u_trial)

solve(F == 0, u, bcs=[bc1,bc2], J=J)

# --- Plot ---
fig, axes = plt.subplots()
collection = tripcolor(u, axes=axes, cmap='Greys')
fig.colorbar(collection);
plt.savefig("Optimised Beam.png")
plt.show()