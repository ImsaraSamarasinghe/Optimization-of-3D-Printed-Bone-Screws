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
    def __init__(self,E_max,nu,p,E_min,t,BC1,BC2,BC3,v,u,uh,rho,rho_filt,r_min,RHO,find_area,alpha,beta,file): # create class variables
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
        self.alpha = alpha
        self.IDP = None
        self.beta = beta
        self.lambda_ = None
        self.mu = None
        self.file = file

    #---------- class attributes for computations ---------------
    def HH_filter(self): # Helmholtz filter for densities
        rhof, w = TrialFunction(self.RHO), TestFunction(self.RHO)
        A = (self.r_min**2)*inner(grad(rhof), grad(w))*dx+rhof*w*dx
        L = self.rho*w*dx
        bc = []
        solve(A==L, self.rho_filt, bcs=bc)

    def sigma(self,u): # stress Tensor
        return self.lambda_ * div(u) * Identity(2) + 2 * self.mu * self.epsilon(u)

    def epsilon(self,u): # Strain Tensor
        return 0.5 * (grad(u) + grad(u).T)
        
    def get_IDP(self): # function to access IDP outside the class
        return self.IDP
    
    def tanh_filter(self,eta):
        numerator = np.tanh(self.beta * eta) + np.tanh(self.beta * (self.rho_filt.dat.data - eta))
        denominator = np.tanh(self.beta * eta) + np.tanh(self.beta * (1 - eta))
        self.rho_filt.vector()[:] = numerator/denominator
    
    def forward(self):
        E = self.E_min + (self.E_max - self.E_min)*self.rho_filt**self.p # SIMP Equation
        self.lambda_ = E*self.nu/((1-self.nu)*(1-2*self.nu))
        self.mu = E/(2*(1+self.nu))
        a = inner(self.sigma(self.u),self.epsilon(self.v))*dx
        l = inner(self.t,self.v)*ds(2)
        forward_problem = LinearVariationalProblem(a,l,self.uh,bcs=[self.BC1,self.BC2,self.BC3])##NOTE:::BC1 REMOVED
        forward_solver = LinearVariationalSolver(forward_problem)
        forward_solver.solve()
    #------ end of class attributes for computations -----------

    # Function for the creation of the objective function
    def objective(self,x):
        self.rho.vector()[:] = x # Assign new densities to rho
        self.HH_filter() # filter densities
        self.tanh_filter(0.5) # filter using tanh filter (eta = 0.5)
        self.forward() # forward problem
        self.file.write(self.uh)
        # Find the new objective
        s = self.sigma(self.uh)
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
        self.forward() # forward problem
        # --- find gradient ------
        s = self.sigma(self.uh)
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
        
        Volume = assemble(self.rho_filt*dx)
        
        # Intermediate Density Penalisation
        self.IDP = assemble(((4.*self.rho_filt*(1.-self.rho_filt))**(1-self.alpha))*dx)
        
        # Magnitude constraint
        self.forward()
        mag = assemble(inner(self.uh,self.uh)**(0.5)*ds(2))

        return np.array((Volume,self.IDP,mag))
    
    
    # function to find jacobian
    def jacobian(self,x):
        # gradient of the volume
        self.rho.vector()[:] = x
        self.HH_filter()
        self.tanh_filter(0.5)
        
        Volume = assemble(self.rho_filt*dx)
        c = Control(self.rho)
        jac1 = compute_gradient(Volume,c)
        
        # gradient of the IDP Function
        self.IDP = assemble(((4.*self.rho_filt*(1.-self.rho_filt))**(1-self.alpha))*dx)
        jac2 = compute_gradient(self.IDP,c)
        
        # gradient of the mangitude
        self.forward()
        mag = assemble(inner(self.uh,self.uh)**(0.5)*ds(2))
        jac3 = compute_gradient(mag,c)
        
        return np.concatenate((jac1.dat.data,jac2.dat.data,jac3.dat.data))

def main():
    # plotting settings
    file = File(f"/home/is420/MEng_project_controlled/newIDPresults/vtu/uh.pvd")
    plt.style.use("dark_background")
    # Times
    t1 = 0
    # ------ problem parameters ------------
    L, W = 5.0, 1.0 # domain size
    nx, ny = 150, 30 # mesh size
    VolFrac = 0.5*L*W # Volume Fraction
    E_max, nu = 110e9, 0.3 # material properties # code tested at 1 kinda worked # new youngs modulus titanium alloy 
    p, E_min = 3.0, 1e-3 # SIMP Values
    t = Constant([2000,0]) # load # t = 2000

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
    find_area = Function(RHO).assign(Constant(1))

    # ------ create optimiser -----
    rho_init.assign(0.5) # Assign starting initialisation for the rho field = 0.5
    x0 = rho_init.vector()[:].tolist() # Initial guess (rho initial)
    ub = np.ones(rho_init.vector()[:].shape).tolist() # upper bound of rho
    lb = np.zeros(rho_init.vector()[:].shape).tolist() # lower bound of rho
    
    # --- constraints ---
    Volume_Lower = 0
    Volume_Upper = VolFrac
    phi_max = 100
    phi_min = 0
    u_min = 0
    u_max = 2*9e-9 # Value seen with full titanium block pulled out.

    cl = [Volume_Lower,phi_min,u_min] # lower bound of the constraints
    alpha = 0.0000001 # value of alpha
    beta = 2 # value of beta
    
    # ------- solve with sub-iterations -------
    for i in range(1,5):
        cu = [Volume_Upper,phi_max,u_max] #Update the constraints 
        obj = cantilever(E_max,nu,p,E_min,t,BC1,BC2,BC3,v,u,uh,rho,rho_filt,r_min,RHO,find_area,alpha,beta,file) # create object class
        
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
            max_iter = 90
        else:
            max_iter = 30
        
        
        TopOpt_problem.add_option('linear_solver', 'ma57')
        TopOpt_problem.add_option('max_iter', max_iter) 
        TopOpt_problem.add_option('accept_after_max_steps', 10)
        TopOpt_problem.add_option('hessian_approximation', 'limited-memory')
        TopOpt_problem.add_option('mu_strategy', 'adaptive')
        TopOpt_problem.add_option('mu_oracle', 'probing')
        TopOpt_problem.add_option('tol', 1e-5)
        
        print(f" ##### starting sub-it: {i}, alpha: {alpha}, beta: {beta}, phi_max: {phi_max}, max_iter: {max_iter} ###### ")
        rho_opt, info = TopOpt_problem.solve(x0) # ---- SOLVE -----
        print(f" ##### ending sub-it: {i} #####")
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
        filename = f"/home/is420/MEng_project_controlled/newIDPresults/iteration_realVal{i}.pvd"
        File(filename).write(rho_init)
        
        # write png files
        fig, axes = plt.subplots()
        collection = tripcolor(rho, axes=axes, cmap='Greys')
        colorbar = fig.colorbar(collection);
        colorbar.set_label(r'$\rho$',fontsize=14,rotation=90)
        plt.gca().set_aspect(1)
        plt.savefig(f"/home/is420/MEng_project_controlled/newIDPresults/iteration_realVal{i}.png")

if __name__ == '__main__':
    main()