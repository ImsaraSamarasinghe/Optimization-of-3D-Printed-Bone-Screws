import matplotlib.pyplot as plt
from firedrake import *
from firedrake.adjoint import *
import numpy as np
continue_annotation()

mesh = UnitSquareMesh(20,20)

E = 20.0e6
nu = 0.3
lambda_ = (E*nu)/((1-nu)*(1-2*nu))
mu = 0.5*E/(1+nu)
rho_0 = 200.0

#load
G_val = 1.0e5# Load on Neumann boundary
B_val = -10# Body force

V = VectorFunctionSpace(mesh,'CG',1)
u_tr = TrialFunction(V)
u_test = TestFunction(V)
u = Function(V)
g = Constant([0.0,G_val])
b = Constant([0.0,B_val])
N = Constant([0,1])

bc1 = DirichletBC(V,Constant([0,0]),3)
aa, bb, cc, dd, ee = 0.5*mu, 0.0, 0.0, mu, -1.5*mu

# Weak form
I = Identity(2)
F = I + grad(u)
C = F.T*F
J = det(F)

n = dot(cofac(F),N)
surface_def = sqrt(inner(n,n))
psi = (aa*inner(F,F)+ee-dd*ln(J))*dx-rho_0*J*dot(b,u)*dx+surface_def*inner(g,u)*ds(1)

#solver
Form = derivative(psi,u,u_test)
Jac = derivative(Form,u,u_tr)

problem = NonlinearVariationalProblem(Form,u,bc1,Jac)
solver = NonlinearVariationalSolver(problem)
solver.solve()


# --- Plot ---
fig, axes = plt.subplots()
collection = tripcolor(u, axes=axes, cmap='Greys')
fig.colorbar(collection);
plt.savefig("Optimised Beam.png")
plt.show()


