# ------- Import libraries ------
from firedrake import *
from firedrake.adjoint import *
import cyipopt
import numpy as np
import matplotlib.pyplot as plt
import pickle
continue_annotation() # start tape

# ------- setup class --------
class cantilever:
    def __init__(self,E_max,nu,p,E_min,t,BC1,BC2,BC3,v,u,uh,rho,rho_filt,r_min,RHO,find_area,alpha,beta,STRESS,x,iter,mesh): # create class variables
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
        self.STRESS = STRESS
        self.x=x # x coordinates for the forcing function
        self.iter = iter # iteration count
        self.d_file = File(f"newIDPresults/vtu/uh_iter_{self.iter}.pvd")
        self.rho_filt2 = Function(self.RHO) # new function to store after TANH filter
        self.png_count = 0
        # -- lists for storing constraints -- 
        self.function_constraint = []
        self.IDP_constraint = []
        self.Volume_constraint = []
        self.u_constraint = []
        self.uf_constraint = [] # upper force constraint history
        self.lf_constraint = [] # lower force constraint history
        self.eq_constraint = [] # equillibrium constraint history
        # --- Lists for storing objective history ---
        self.objective_history = []
        # --- Store rho ----
        self.rho_file = File(f"PNG_rho/rho_iter_{self.iter}.pvd")
        # ---- store force histories ----
        self.upper_force = []
        self.lower_force = []
        # ---  mesh --- 
        self.mesh = mesh
        # --- Facet Normals ---
        self.n = FacetNormal(self.mesh)
        # --- Forcing Function ----
        self.ForcingFunction = project((tanh(asin(sin(15*self.x))*100)+1)/2,self.RHO) # Added a forcing function for edges.
        # --- record shear stress for objective ---
        self.obj_shear_upper = []
        self.obj_shear_lower = []
        
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
    
    def tanh_filter(self,eta): # Projection filter
        numerator = tanh(self.beta * eta) + tanh(self.beta * (self.rho_filt - eta))
        denominator = tanh(self.beta * eta) + tanh(self.beta * (1 - eta))
        self.rho_filt2.interpolate(numerator/denominator)
    
    def forward(self):
        E = self.E_min + (self.E_max - self.E_min)*self.rho_filt2**self.p # SIMP Equation
        self.lambda_ = E*self.nu/((1+self.nu)*(1-2*self.nu))
        self.mu = E/(2*(1+self.nu))
        a = inner(self.sigma(self.u),self.epsilon(self.v))*dx
        l = inner(self.t,self.v)*ds(2)
        forward_problem = LinearVariationalProblem(a,l,self.uh,bcs=[self.BC2,self.BC3])
        forward_solver = LinearVariationalSolver(forward_problem)
        forward_solver.solve()
    
    # function to fill in the voight stress vector
    def voigt_vector(self):
        s = self.sigma(self.uh)
        return as_vector([s[0,0],s[1,1]])

    ####
    def rec_stress(self):
        s = self.sigma(self.uh)
        shear = s[0,1]
        ss = project(shear,self.STRESS)
        
        fig, axes = plt.subplots()
        collection = tripcolor(ss, axes=axes, cmap='viridis')
        colorbar = fig.colorbar(collection);
        colorbar.set_label(r'$\sigma_{xy}$',fontsize=14,rotation=90)
        plt.gca().set_aspect(1)
        plt.savefig(f"PNG_shear/shear_{self.png_count}_{self.iter}.png")
        self.png_count = self.png_count + 1
        plt.close('all')
    ####
    def rec_constraints(self,vol,IDP,u,func,uf,lf,equillibrium):
        self.function_constraint.append(func)
        self.IDP_constraint.append(IDP)
        self.Volume_constraint.append(vol)
        self.u_constraint.append(u)
        self.uf_constraint.append(uf)
        self.lf_constraint.append(lf)
        self.eq_constraint.append(equillibrium)
        # pickle lists
        with open('PickleFiles/constraints.pkl','wb') as file:
            pickle.dump((self.function_constraint,self.IDP_constraint,self.Volume_constraint
                        ,self.u_constraint,self.uf_constraint,self.lf_constraint,self.eq_constraint),file)
    
    def rec_objective(self,J):
        self.objective_history.append(J)
        # create pickle here
        with open('PickleFiles/objective.pkl','wb') as file:
            pickle.dump(self.objective_history, file)
    
    def rec_forces(self,s):
        self.upper_force.append(assemble(s[1,1]*ds(4)))
        self.lower_force.append(assemble(s[1,1]*ds(3)))
        # pickle lists
        with open('PickleFiles/forces.pkl','wb') as file:
            pickle.dump((self.upper_force, self.lower_force), file)
        # create pickle here
    #------ end of class attributes for computations -----------

    # Function for the creation of the objective function
    def objective(self,x):
        self.rho.vector()[:] = x # Assign new densities to rho
        self.rho_file.write(self.rho) # write to rho_file
        self.HH_filter() # filter densities
        self.tanh_filter(0.5) # filter using tanh filter (eta = 0.5)
        self.forward() # forward problem
        self.d_file.write(self.uh) # write displacement to file
        self.rec_stress()
        # Find the new objective
        s = self.sigma(self.uh)
        Force_upper = assemble(s[0,1]*ds(4))
        Force_lower = assemble(s[0,1]*ds(3))
        area_upper = assemble(self.find_area*ds(4))
        area_lower = assemble(self.find_area*ds(3))
        avg_ss_upper = Force_upper/area_upper
        avg_ss_lower = Force_lower/area_lower
        J = assemble((s[0,1]-avg_ss_upper)**2*ds(4)+(s[0,1]-avg_ss_lower)**2*ds(3))
        
        """ Record for testing only """
        with open('PickleFiles/RecordFiles.pkl','wb') as file:
            self.obj_shear_upper.append(Force_upper)
            self.obj_shear_lower.append(Force_lower)
            pickle.dump((self.obj_shear_upper,self.obj_shear_lower), file)
        """ Record for testing only """

        self.rec_objective(J)
        self.rec_forces(s)
        
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
        # Initialisation
        self.rho.vector()[:] = x
        self.HH_filter()
        self.tanh_filter(0.5)
        
        # Volume Constraint
        Volume = assemble(self.rho_filt2*dx)
        
        # Intermediate Density Penalisation
        IDP = assemble(((4.*self.rho_filt2*(1.-self.rho_filt2))**(1-self.alpha))*dx)
        
        # Magnitude constraint on the RHS boundary
        self.forward()
        mag = assemble(dot(self.uh,self.n)*ds(2))
        
        # Forcing Function
        mag2 = assemble((self.rho_filt2-self.ForcingFunction)**2*ds(3)+(self.rho_filt2-self.ForcingFunction)**2*ds(4))
                
        # expansion forces
        # force constraint 1
        voigt_stress = self.voigt_vector()
        upper_force_constraint = assemble(dot(voigt_stress,self.n)*ds(4))
        lower_force_constraint = assemble(dot(voigt_stress,self.n)*ds(3))
        equillibrium_constraint = assemble(dot(voigt_stress,self.n)*ds(4)-dot(voigt_stress,self.n)*ds(3))

        # shear constraint (top and bottom shear must be equivalent)
        s = self.sigma(self.uh)
        shear_constraint = assemble(s[0,1]*ds(4)+s[0,1]*ds(3))

        # shear must add upto total force
        force_xx = assemble(s[0,0]*ds(2)-s[0,1]*ds(3)+s[0,1]*ds(4))
        
        self.rec_constraints(Volume,IDP,mag,mag2,upper_force_constraint,lower_force_constraint,equillibrium_constraint) # record all constraints for history
        
        return np.array((Volume,IDP,mag,mag2,upper_force_constraint,lower_force_constraint,equillibrium_constraint,shear_constraint,force_xx))
    
    
    # function to find jacobian
    def jacobian(self,x):
        # Initialisation
        self.rho.vector()[:] = x
        self.HH_filter()
        self.tanh_filter(0.5)
        c = Control(self.rho)
        # gradient of the colume constraint
        Volume = assemble(self.rho_filt2*dx)
        jac1 = compute_gradient(Volume,c)
        
        # gradient of the IDP Function
        IDP = assemble(((4.*self.rho_filt2*(1.-self.rho_filt2))**(1-self.alpha))*dx) ##### changed to rho_filt
        jac2 = compute_gradient(IDP,c)
        
        # gradient of the magnitude on the RHS boundary
        self.forward()
        mag = assemble(dot(self.uh,self.n)*ds(2))
        jac3 = compute_gradient(mag,c)
        
        # gradient of the forcing function
        mag2 = assemble((self.rho_filt2-self.ForcingFunction)**2*ds(3)+(self.rho_filt2-self.ForcingFunction)**2*ds(4))
        jac4 = compute_gradient(mag2,c)
        
        # expansion forces - gradients
        voigt_stress = self.voigt_vector()
        upper_force_constraint = assemble(dot(voigt_stress,self.n)*ds(4))
        lower_force_constraint = assemble(dot(voigt_stress,self.n)*ds(3))
        equillibrium_constraint = assemble(dot(voigt_stress,self.n)*ds(4)-dot(voigt_stress,self.n)*ds(3))
        
        jac5 = compute_gradient(upper_force_constraint,c)
        jac6 = compute_gradient(lower_force_constraint,c)
        jac7 = compute_gradient(equillibrium_constraint,c)

        # shear constraint
        s = self.sigma(self.uh)
        shear_constraint = assemble(s[0,1]*ds(4)+s[0,1]*ds(3))
        jac8 = compute_gradient(shear_constraint,c)

        # shear must add upto total force
        force_xx = assemble(s[0,0]*ds(2)-s[0,1]*ds(3)+s[0,1]*ds(4))
        jac9 = compute_gradient(force_xx,c)
        
        return np.concatenate((jac1.dat.data,jac2.dat.data,jac3.dat.data,jac4.dat.data,jac5.dat.data,jac6.dat.data,jac7.dat.data,jac8.dat.data,jac9.dat.data))


def function_plot(function,str,sub_iter):
    fig, axes = plt.subplots()
    collection = tripcolor(function, axes=axes, cmap='viridis')
    colorbar = fig.colorbar(collection);
    colorbar.set_label(r'$\rho$',fontsize=14,rotation=90)
    plt.title(f"sub_iteration: {sub_iter}")
    plt.gca().set_aspect(1)
    plt.savefig(f"newIDPresults/{str}_sub-iter{sub_iter}.png")
    plt.close('all')

    
def main():
    # plotting settings
    plt.style.use("default")
    # Times
    t1 = 0
    # ------ problem parameters ------------
    L, W = 5.0, 1.0 # domain size # original 5.0,1.0
    nx, ny = 180, 90 # mesh size 180, 90
    VolFrac = 0.7*L*W # Volume Fraction
    E_max, nu = 110e9, 0.3 # material properties #try changing the poissons ratio 
    p, E_min = 3.0, 1e-3 # SIMP Values
    t = Constant([1400,0]) # load # t = 2000

    # ----- setup BC, mesh, and function spaces ----
    mesh = RectangleMesh(nx,ny,L,W)
    x, y = SpatialCoordinate(mesh)
    V = VectorFunctionSpace(mesh,'CG',1)
    RHO = FunctionSpace(mesh,'CG',1)
    STRESS = FunctionSpace(mesh,'CG',1) ## stress function space
    # BC1 = DirichletBC(V,Constant([0,0]),1)
    BC1 = DirichletBC(V.sub(1),Constant(0),2) # set y component of Neumann boundary to zero
    BC2 = DirichletBC(V,Constant([0,0]),3)
    BC3 = DirichletBC(V,Constant([0,0]),4)
    
    # radius for hh HH_filter
    r_min = 1.6*L/nx # working is only at 1.6 and with IDP using rho_filt2

    # ------ setup functions -----
    v = TestFunction(V)
    u = TrialFunction(V)
    uh = Function(V)
    rho = Function(RHO)
    rho_init = Function(RHO)
    rho_temp = Function(RHO) # used for initiliase function
    rho_filt = Function(RHO)
    find_area = Function(RHO).assign(Constant(1))

    # ------ create optimiser -----
    rho_init.vector()[:] = 0.5
    function_plot(rho_init,"initialisation_rho_","inital") # plot the initialised rho domain
    x0 = rho_init.vector()[:].tolist() # Initial guess (rho initial)
    ub = np.ones(rho_init.vector()[:].shape).tolist() # upper bound of rho
    lb = np.ones(rho_init.vector()[:].shape)*0.00001 # lower bound of rho
    lb = lb.tolist()
    
    # --- constraints (max & min)---
    Volume_Lower = 0
    Volume_Upper = VolFrac
    phi_max = 10 # currently set at 10
    phi_min = 0
    u_min = 0
    # u_max = 20*6.477e-9# Value seen with full titanium block pulled out. (multiplied)
    u_max = 1000*5.26e-09 # 15*1.5e-8
    force_func_max = 1
    force_func_min = -1
    # force constraints
    # Both force constraints together
    upper_force_min = 200
    upper_force_max = 10000
    lower_force_min = 200
    lower_force_max = 10000
    equillibrium_min = -1
    equillibrium_max = 1
    # shear
    max_shear_con = 1
    min_shear_con = -1
    # force equillibrium in xx
    max_force_xx = 1
    min_force_xx = -1
    # ------------------------------

    cl = [Volume_Lower,phi_min,u_min,force_func_min,upper_force_min,lower_force_min,equillibrium_min,min_shear_con,min_force_xx] # lower bound of the constraints
    alpha = 0.05 # value of alpha , ORIGINAL = 0.0000001 try with alpha=0.05
    beta = 2 # value of beta , ORIGINAL = 2 current 3
    
    # ------- solve with sub-iterations -------
    for i in range(1,5): # set for only sub-iteration
        cu = [Volume_Upper,phi_max,u_max,force_func_max,upper_force_max,lower_force_max,equillibrium_max,max_shear_con,max_force_xx] #Update the constraints 
        obj = cantilever(E_max,nu,p,E_min,t,BC1,BC2,BC3,v,u,uh,rho,rho_filt,r_min,RHO,find_area,alpha,beta,STRESS,x,i,mesh) # create object class
        
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
            max_iter = 50 ## currently stopping at 58
        else:
            max_iter = 20 ## currently stopping at 10
        
        
        TopOpt_problem.add_option('linear_solver', 'ma57')
        TopOpt_problem.add_option('max_iter', max_iter) 
        TopOpt_problem.add_option('accept_after_max_steps', 10)
        TopOpt_problem.add_option('hessian_approximation', 'limited-memory')
        TopOpt_problem.add_option('mu_strategy', 'adaptive')
        TopOpt_problem.add_option('mu_oracle', 'probing')
        TopOpt_problem.add_option('tol', 1e-5)
        TopOpt_problem.add_option('max_cpu_time', 700.0)
        # TopOpt_problem.add_option('max_wall_time', 360.0)

        print(f" ##### starting sub-it: {i}, alpha: {alpha}, beta: {beta}, phi_max: {phi_max}, max_iter: {max_iter} ###### ")
        rho_opt, info = TopOpt_problem.solve(x0) # ---- SOLVE -----
        print(f" ##### ending sub-it: {i} #####")
        
        rho_init.vector()[:] = np.array(rho_opt) # Assign optimised rho to rho_init for next iteration
        x0 = rho_init.vector()[:].tolist() # Warm start the next iteration using the last iteration
        
        # phi_max according to paper
        phi_max = obj.IDP_constraint[-1] # use the last reached value of IDP_constraint
            
        alpha = 0.18*i-0.13+0.05 # Update alpha linearly according to paper
        beta = 4*i # Update beta according to paper
        
        # write png files of final rho distribution
        function_plot(rho_init,'final rho distribution',i)
        
        # ----- Final boundary Forces ----
        print(f"##################################")
        print(f"Upper force: {obj.upper_force[-1]}")
        print(f"Lower force: {obj.lower_force[-1]}")
        print(f"##################################")

if __name__ == '__main__':
    main()