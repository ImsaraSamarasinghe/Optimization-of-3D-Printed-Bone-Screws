from firedrake import *
from firedrake.adjoint import *
import cyipopt
import numpy as np
import matplotlib.pyplot as plt
continue_annotation() # start tape

class cantilever:
    def __init__(self,E_max,nu,p,E_min,t,BC,v,u,uh,rho,rho_filt,r_min,RHO,outfile):
        self.E_max = E_max
        self.nu = nu
        self.p = p
        self.E_min = E_min
        self.t = t
        self.BC = BC
        self.v = v
        self.u = u
        self.uh = uh
        self.rho = rho
        self.rho_filt = rho_filt
        self.r_min = r_min
        self.RHO = RHO
        self.outfile = outfile

    # class attributes for computations
    def HH_filter(self):
        rhof, w = TrialFunction(self.RHO), TestFunction(self.RHO)
        A = (self.r_min**2)*inner(grad(rhof), grad(w))*dx+rhof*w*dx
        L = self.rho*w*dx
        bc = []
        solve(A==L, self.rho_filt, bcs=bc)
        #print(f"rho_filt 1:\n {self.rho_filt.vector()[:]}")

    def sigma(self,u,lambda_,mu):
        return lambda_ * div(u) * Identity(2) + 2 * mu * self.epsilon(u)

    def epsilon(self,u):
        return 0.5 * (grad(u) + grad(u).T)
    # end of class attributes for computations


    def objective(self,x):
        #print("start objective")
        self.rho.vector()[:] = x # Assign new densities to rho
        self.HH_filter() # filter
        #print(f"rho_filt 2:\n {self.rho_filt.vector()[:]}")
        # forward problem - start
        E = self.E_min + (self.E_max - self.E_min)*self.rho_filt**self.p
        lambda_ = E*self.nu/((1+self.nu)*(1-2*self.nu))
        mu = E/(2*(1+self.nu))

        a = inner(self.sigma(self.u,lambda_,mu),self.epsilon(self.v))*dx
        l = inner(self.t,self.v)*ds(2)

        forward_problem = LinearVariationalProblem(a,l,self.uh,bcs=self.BC)
        forward_solver = LinearVariationalSolver(forward_problem)
        forward_solver.solve()
        # forward problem - end
        J = assemble(inner(self.t,self.uh)*ds(2))
        #print("End objective")
        
        # create animation
        self.outfile.write(self.rho_filt)
        # end create animation
        
        return J

    def gradient(self,x):
        #print("start gradient")
        self.rho.vector()[:] = x # assign new rho
        self.HH_filter() # filter rho

        # forward problem - start
        E = self.E_min + (self.E_max - self.E_min)*self.rho_filt**self.p
        lambda_ = E*self.nu/((1+self.nu)*(1-2*self.nu))
        mu = E/(2*(1+self.nu))

        a = inner(self.sigma(self.u,lambda_,mu),self.epsilon(self.v))*dx
        l = inner(self.t,self.v)*ds(2)

        forward_problem = LinearVariationalProblem(a,l,self.uh,bcs=self.BC)
        forward_solver = LinearVariationalSolver(forward_problem)
        forward_solver.solve()
        # forward problem - end
        J = assemble(inner(self.t,self.uh)*ds(2))
        c = Control(self.rho)
        dJdRho = compute_gradient(J,c)
        #print("end gradient")
        return dJdRho.dat.data

    def constraints(self,x):
        #print('start Constraints')
        self.rho.vector()[:] = x
        self.HH_filter()
        Volume = assemble(self.rho_filt*dx)
        #print('end constraint')
        return Volume

    def jacobian(self,x):
        #print('start Jacobian')
        self.rho.vector()[:] = x
        self.HH_filter()
        Volume = assemble(self.rho_filt*dx)
        c = Control(self.rho)
        jac = compute_gradient(Volume,c)
        #print("end constraint")
        return jac.dat.data

def main():
    outfile = File("/home/is420/MEng_project_controlled/outInterim/CantileverAnimated.pvd")
    # problem parameters
    L, W = 5.0, 1.0 # domain size
    nx, ny = 150, 30 # mesh size
    VolFrac = 0.5*L*W # Volume Fraction
    E_max, nu = 1e5, 0.3 # material properties
    p, E_min = 3.0, 1e-3 # SIMP Values
    t = Constant([0,-2000]) # load

    # setup BC, mesh, and function spaces
    mesh = RectangleMesh(nx,ny,L,W)
    V = VectorFunctionSpace(mesh,'CG',1)
    RHO = FunctionSpace(mesh,'CG',1)
    BC = DirichletBC(V,Constant([0,0]),1)

    # radius for hh HH_filter
    r_min = 0.02 # currently working and tested value is 0.02

    # setup functions
    v = TestFunction(V)
    u = TrialFunction(V)
    uh = Function(V)
    rho = Function(RHO)
    rho_init = Function(RHO)
    rho_filt = Function(RHO)

    # create optimiser
    rho_init.assign(0.5)
    x0 = rho_init.vector()[:].tolist() # Initial guess (rho initial)
    ub = np.ones(rho_init.vector()[:].shape).tolist() # upper bound of rho
    lb = np.zeros(rho_init.vector()[:].shape).tolist() # lower bound of rho

    cl = [0] # lower bound of the volume constraint
    cu = [VolFrac] # upper bound of the volume constraint

    obj_cantilever = cantilever(E_max,nu,p,E_min,t,BC,v,u,uh,rho,rho_filt,r_min,RHO,outfile)

    TopOpt_problem = cyipopt.Problem(
        n = len(x0),
        m = len(cl),
        problem_obj=obj_cantilever,
        lb = lb,
        ub = ub,
        cl = cl,
        cu = cu
    )

    TopOpt_problem.add_option('linear_solver', 'ma57')
    TopOpt_problem.add_option('max_iter', 50) # max 220
    TopOpt_problem.add_option('accept_after_max_steps', 10) # was 10
    TopOpt_problem.add_option('hessian_approximation', 'limited-memory')
    TopOpt_problem.add_option('mu_strategy', 'adaptive')
    TopOpt_problem.add_option('mu_oracle', 'probing')
    TopOpt_problem.add_option('tol', 1e-5)

    rho_opt, info = TopOpt_problem.solve(x0)

    rho.vector()[:] = np.array(rho_opt)
    
    # --- Plot ---
    plt.style.use("dark_background")
    fig, axes = plt.subplots()
    collection = tripcolor(rho, axes=axes, cmap='Greys')
    colorbar = fig.colorbar(collection);
    colorbar.set_label(r'$\rho$',fontsize=14,rotation=90)
    plt.gca().set_aspect(1)
    plt.savefig("/home/is420/MEng_project_controlled/outInterim/Optimised Beam.png")
    plt.show()
    
    File("/home/is420/MEng_project_controlled/outInterim/CantileverBeam.pvd").write(rho)

if __name__ == '__main__':
    main()
