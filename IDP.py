# ------- Import libraries ------
from firedrake import *
from firedrake.adjoint import *
import cyipopt
import numpy as np
import matplotlib.pyplot as plt
import random
continue_annotation() # start tape

# ------- setup class --------
class cantilever:
    def __init__(self,E_max,nu,p,E_min,t,BC1,BC2,BC3,v,u,uh,rho,rho_filt,rho_filt2,r_min,RHO,find_area,alpha,beta,i): # create class variables
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
        self.rho_filt2 = rho_filt2
        self.r_min = r_min
        self.RHO = RHO
        self.find_area = find_area
        self.alpha = alpha
        self.IDP = None
        self.beta = beta
        self.i = i
        self.outfile1 = File(f"Design_Variable_1.pvd")
        self.outfile2 = File(f"HelmHoltzFilter_1.pvd")
        self.outfile3 = File(f"ProjectionFilter_1.pvd")

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
        
    def get_IDP(self): # function to access IDP outside the class
        return self.IDP
    
    def tanh_filter(self,eta):
        numerator = np.tanh(self.beta * eta) + np.tanh(self.beta * (self.rho_filt.dat.data - eta))
        denominator = np.tanh(self.beta * eta) + np.tanh(self.beta * (1 - eta))
        self.rho_filt2.vector()[:] = numerator/denominator
        
    #------ end of class attributes for computations -----------

    # Function for the creation of the objective function
    def objective(self,x):
        self.rho.vector()[:] = x # Assign new densities to rho
        self.outfile1.write(self.rho)
        
        self.HH_filter() # filter densities
        self.outfile2.write(self.rho_filt)
        
        self.tanh_filter(0.5) # filter using tanh filter (eta = 0.5)
        self.outfile3.write(self.rho_filt)
        
        E = self.E_min + (self.E_max - self.E_min)*self.rho_filt2**self.p # SIMP Equation
        
        # ------- Forward Problem -START -------------
        lambda_ = E*self.nu/((1-self.nu)*(1-2*self.nu))
        mu = E/(2*(1+self.nu))

        a = inner(self.sigma(self.u,lambda_,mu),self.epsilon(self.v))*dx
        l = inner(self.t,self.v)*ds(2)

        forward_problem = LinearVariationalProblem(a,l,self.uh,bcs=[self.BC1,self.BC2,self.BC3])##NOTE:::BC1 REMOVED
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
        self.tanh_filter(0.5) # filter using tanh filter (eta = 0.5)

        # ----------- forward problem - start ------------
        E = self.E_min + (self.E_max - self.E_min)*self.rho_filt2**self.p
        lambda_ = E*self.nu/((1-self.nu)*(1-2*self.nu))
        mu = E/(2*(1+self.nu))

        a = inner(self.sigma(self.u,lambda_,mu),self.epsilon(self.v))*dx
        l = inner(self.t,self.v)*ds(2)

        forward_problem = LinearVariationalProblem(a,l,self.uh,bcs=[self.BC1,self.BC2,self.BC3])##NOTE:::BC1 REMOVED
        forward_solver = LinearVariationalSolver(forward_problem)
        forward_solver.solve()
        # --------- forward problem - end -----------------
        
        # --- find gradient ------
        # Find the new objective
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
        self.tanh_filter(0.5)
        
        Volume = assemble(self.rho_filt2*dx)
        
        # Intermediate Density Penalisation
        self.IDP = assemble(((4.*self.rho_filt2*(1.-self.rho_filt2))**(1-self.alpha))*dx)

        return np.array((Volume,self.IDP))
    
    
    # function to find jacobian
    def jacobian(self,x):
        # gradient of the volume
        self.rho.vector()[:] = x
        self.HH_filter()
        self.tanh_filter(0.5)
        
        Volume = assemble(self.rho_filt2*dx)
        c = Control(self.rho)
        jac1 = compute_gradient(Volume,c)
        
        # gradient of the IDP Function
        self.IDP = assemble(((4.*self.rho_filt2*(1.-self.rho_filt2))**(1-self.alpha))*dx)
        jac2 = compute_gradient(self.IDP,c)
        
        return np.concatenate((jac1.dat.data,jac2.dat.data))

def main():
    # Times
    t1 = 0
    # ------ problem parameters ------------
    L, W = 5.0, 1.0 # domain size
    nx, ny = 150, 30 # mesh size
    VolFrac = 0.5*L*W # Volume Fraction
    E_max, nu = 1, 0.3 # material properties # E_max = 1e5
    p, E_min = 3.0, 1e-3 # SIMP Values
    t = Constant([1,0]) # load # t = 2000

    # ----- setup BC, mesh, and function spaces ----
    mesh = RectangleMesh(nx,ny,L,W)
    V = VectorFunctionSpace(mesh,'CG',1)
    RHO = FunctionSpace(mesh,'CG',1)
    BC1 = DirichletBC(V,Constant([0,0]),1)
    BC2 = DirichletBC(V,Constant([0,0]),3)
    BC3 = DirichletBC(V,Constant([0,0]),4)
    
    # radius for hh HH_filter
    r_min = 0.08

    # ------ setup functions -----
    v = TestFunction(V)
    u = TrialFunction(V)
    uh = Function(V)
    rho = Function(RHO)
    rho_init = Function(RHO)
    rho_filt = Function(RHO)
    rho_filt2 = Function(RHO)
    find_area = Function(RHO).assign(Constant(1))

    # ------ create optimiser -----
    rho_init.assign(0.5) # Assign starting initialisation for the rho field = 0.5
    x0 = rho_init.vector()[:].tolist() # Initial guess (rho initial)
    ub = np.ones(rho_init.vector()[:].shape).tolist() # upper bound of rho
    lb = np.zeros(rho_init.vector()[:].shape).tolist() # lower bound of rho
    
    Volume_Lower = 0
    Volume_Upper = VolFrac
    phi_max = 50
    phi_min = 0

    cl = [Volume_Lower,phi_min] # lower bound of the constraints
    alpha = 0.0000001
    beta = 2
    
    # ------- solve with sub-iterations -------
    for i in range(1,4):
        cu = [Volume_Upper,phi_max] #Update the constraints 
        obj = cantilever(E_max,nu,p,E_min,t,BC1,BC2,BC3,v,u,uh,rho,rho_filt,rho_filt2,r_min,RHO,find_area,alpha,beta,i) # create object class
        
        # Setup problem
        TopOpt_problem = cyipopt.Problem(
            n = len(x0),
            m = len(cl),
            problem_obj=obj,
            lb = lb,
            ub = ub,
            cl = cl,
            cu = cu
        )
        
        # ------ Solver Settings ----
        if (i==1):
            max_iter = 60
        else:
            max_iter = 25
        
        
        TopOpt_problem.add_option('linear_solver', 'ma57')
        TopOpt_problem.add_option('max_iter', max_iter) 
        TopOpt_problem.add_option('accept_after_max_steps', 10)
        TopOpt_problem.add_option('hessian_approximation', 'limited-memory')
        TopOpt_problem.add_option('mu_strategy', 'adaptive')
        TopOpt_problem.add_option('mu_oracle', 'probing')
        TopOpt_problem.add_option('tol', 1e-5)
        
        rho_opt, info = TopOpt_problem.solve(x0) # ---- SOLVE -----
        rho_init.vector()[:] = np.array(rho_opt) # Assign optimised rho to rho_init for next iteration
        x0 = rho_init.vector()[:].tolist() # Warm start the next iteration using the last iteration
        
        # phi_max according to paper
        if (i==7):
            phi_max = 0.075*100
        else:
            phi_max = 0.35*100
            
        alpha = 0.18*i-0.13 # Update alpha linearly according to paper
        beta = 4*i # Update beta according to paper
        
        # write new file with each iteration
        filename = f"iteration_{i}.pvd"
        File(filename).write(rho_init)
    

if __name__ == '__main__':
    main()