# ------- Import libraries ------
from firedrake import *
from firedrake.adjoint import *
import matplotlib.pyplot as plt
import cyipopt
import numpy as np
continue_annotation() # start tape

# ------- setup class --------
class cantilever:
    def __init__(self,E_max,nu,p,E_min,t,BC1,BC2,BC3,v,u,uh,rho,rho_filt,r_min,RHO,find_area): # create class variables
        self.E_max = E_max
        self.nu = nu
        self.p = p
        self.E_min = E_min
        self.t = t
        self.BC1 = BC1
        self.BC2 = BC2
        self.BC3 = BC3
        self.v = v
        self.u = u
        self.uh = uh
        self.rho = rho
        self.rho_filt = rho_filt
        self.r_min = r_min
        self.RHO = RHO
        self.find_area = find_area

    #---------- class attributes for computations ---------------
    def HH_filter(self): # Helmholtz filter for densities
        rhof, w = TrialFunction(self.RHO), TestFunction(self.RHO)
        A = (self.r_min**2)*inner(grad(rhof), grad(w))*dx+rhof*w*dx
        L = self.rho*w*dx
        bc = []
        solve(A==L, self.rho_filt, bcs=bc)

    def sigma(self,u,lambda_,mu): # stress Tensor
        return lambda_ * div(u) * Identity(2) + 2 * mu * self.epsilon(u)

    def epsilon(self,u): # Strain Tensor
        return 0.5 * (grad(u) + grad(u).T)
    #------ end of class attributes for computations -----------

    # Function for the creation of the objective function
    def objective(self,x):
        self.rho.vector()[:] = x # Assign new densities to rho
        self.HH_filter() # filter densities
        
        E = self.E_min + (self.E_max - self.E_min)*self.rho_filt**self.p # SIMP Equation
        
        # ------- Forward Problem -START -------------
        lambda_ = E*self.nu/((1+self.nu)*(1-2*self.nu))
        mu = E/(2*(1+self.nu))

        a = inner(self.sigma(self.u,lambda_,mu),self.epsilon(self.v))*dx
        l = inner(self.t,self.v)*ds(2)

        forward_problem = LinearVariationalProblem(a,l,self.uh,bcs=[self.BC2,self.BC3])##NOTE:::BC1 REMOVED
        forward_solver = LinearVariationalSolver(forward_problem)
        forward_solver.solve()
        # ------ Forward Problem -END ----------------
        
        # Find the new objective
        s = self.sigma(self.uh,lambda_,mu)
        Force_upper = assemble(s[0,1]*ds(4))
        Force_lower = assemble(s[0,1]*ds(3))
        area_upper = assemble(self.find_area*ds(4))
        area_lower = assemble(self.find_area*ds(3))
        avg_ss_upper = Force_upper/area_upper
        avg_ss_lower = Force_lower/area_lower
        J = assemble((s[0,1]-avg_ss_upper)**2*ds(4)+(s[0,1]-avg_ss_lower)**2*ds(3))
        
        return J
        
    # Function for the creation of the gradient
    def gradient(self,x):
        self.rho.vector()[:] = x # assign new densities to rho
        self.HH_filter() # filter rho

        # ----------- forward problem - start ------------
        E = self.E_min + (self.E_max - self.E_min)*self.rho_filt**self.p
        lambda_ = E*self.nu/((1+self.nu)*(1-2*self.nu))
        mu = E/(2*(1+self.nu))

        a = inner(self.sigma(self.u,lambda_,mu),self.epsilon(self.v))*dx
        l = inner(self.t,self.v)*ds(2)

        forward_problem = LinearVariationalProblem(a,l,self.uh,bcs=[self.BC2,self.BC3])##NOTE:::BC1 REMOVED
        forward_solver = LinearVariationalSolver(forward_problem)
        forward_solver.solve()
        # --------- forward problem - end -----------------
        
        # --- find gradient ------
        s = self.sigma(self.uh,lambda_,mu)
        Force_upper = assemble(s[0,1]*ds(4))
        Force_lower = assemble(s[0,1]*ds(3))
        area_upper = assemble(self.find_area*ds(4))
        area_lower = assemble(self.find_area*ds(3))
        avg_ss_upper = Force_upper/area_upper
        avg_ss_lower = Force_lower/area_lower
        J = assemble((s[0,1]-avg_ss_upper)**2*ds(4)+(s[0,1]-avg_ss_lower)**2*ds(3))
        c = Control(self.rho)
        dJdRho = compute_gradient(J,c)
        
        return dJdRho.dat.data
        
    # Function to evaluate constraints
    def constraints(self,x):
        # Volume constraint
        self.rho.vector()[:] = x
        self.HH_filter()
        Volume = assemble(self.rho_filt*dx)
        
        return Volume
    
    
    # function to find jacobian
    def jacobian(self,x):
        # gradient of the volume
        self.rho.vector()[:] = x
        self.HH_filter()
        Volume = assemble(self.rho_filt*dx)
        c = Control(self.rho)
        jac = compute_gradient(Volume,c)
        
        return jac.dat.data

def main():
    # ------ problem parameters ------------
    L, W = 5.0, 1.0 # domain size
    nx, ny = 150, 30 # mesh size
    VolFrac = 0.5*L*W # Volume Fraction
    E_max, nu = 1e5, 0.3 # material properties
    p, E_min = 3.0, 1e-3 # SIMP Values
    t = Constant([2000,0]) # load

    # ----- setup BC, mesh, and function spaces ----
    mesh = RectangleMesh(nx,ny,L,W)
    V = VectorFunctionSpace(mesh,'CG',1)
    RHO = FunctionSpace(mesh,'CG',1)
    BC1 = DirichletBC(V,Constant([0,0]),1)
    BC2 = DirichletBC(V,Constant([0,0]),3)
    BC3 = DirichletBC(V,Constant([0,0]),4)
    

    # radius for hh HH_filter
    r_min = 0.02 # currently working and tested value is 0.02

    # ------ setup functions -----
    v = TestFunction(V)
    u = TrialFunction(V)
    uh = Function(V)
    rho = Function(RHO)
    rho_init = Function(RHO)
    rho_filt = Function(RHO)
    find_area = Function(RHO).assign(Constant(1))

    # ------ create optimiser -----
    rho_init.assign(0.5)
    x0 = rho_init.vector()[:].tolist() # Initial guess (rho initial)
    ub = np.ones(rho_init.vector()[:].shape).tolist() # upper bound of rho
    lb = np.zeros(rho_init.vector()[:].shape).tolist() # lower bound of rho
    
    Volume_Lower = 0
    Volume_Upper = VolFrac


    cl = [Volume_Lower] # lower bound of the constraints
    cu = [Volume_Upper] # upper bound of the constraints

    obj = cantilever(E_max,nu,p,E_min,t,BC1,BC2,BC3,v,u,uh,rho,rho_filt,r_min,RHO,find_area)

    TopOpt_problem = cyipopt.Problem(
        n = len(x0),
        m = len(cl),
        problem_obj=obj,
        lb = lb,
        ub = ub,
        cl = cl,
        cu = cu
    )

    TopOpt_problem.add_option('linear_solver', 'ma57')
    TopOpt_problem.add_option('max_iter', 200) # max 300 so far tested
    TopOpt_problem.add_option('accept_after_max_steps', 10) # was 10
    TopOpt_problem.add_option('hessian_approximation', 'limited-memory')
    TopOpt_problem.add_option('mu_strategy', 'adaptive')
    TopOpt_problem.add_option('mu_oracle', 'probing')
    TopOpt_problem.add_option('tol', 1e-5)
    
    # ------- solve -------
    rho_opt, info = TopOpt_problem.solve(x0)

    # ------ write to .pvd file -------
    rho.vector()[:] = np.array(rho_opt)
    
    # --- PLOT ---
    fig, axes = plt.subplots()
    collection = tripcolor(rho, axes=axes, cmap='Greys')
    fig.colorbar(collection);
    plt.savefig("Grey Area.png")
    plt.show()
    # File("Grey_confirmation.pvd").write(rho)

if __name__ == '__main__':
    main()
