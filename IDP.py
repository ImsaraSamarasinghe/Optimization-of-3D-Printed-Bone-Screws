# ------- Import libraries ------
from firedrake import *
from firedrake.adjoint import *
import cyipopt
import numpy as np
import matplotlib.pyplot as plt
import random
import noise
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
        self.ForcingFunction=0
        self.iter = iter # iteration count
        self.d_file = File(f"/home/is420/MEng_project_controlled/newIDPresults/vtu/uh_iter_{self.iter}.pvd")
        self.s_file = File(f"/home/is420/MEng_project_controlled/newIDPresults/stress_folder/stresses_iter.pvd")
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
        self.rho_file = File(f"/home/is420/MEng_project_controlled/PNG_rho/rho_iter_{self.iter}.pvd")
        # ---- store force histories ----
        self.upper_force = []
        self.lower_force = []
        # ---  mesh --- 
        self.mesh = mesh
        # --- Facet Normals ---
        self.n = FacetNormal(self.mesh)
        # --- Voight space and Functions ---
        
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
        forward_problem = LinearVariationalProblem(a,l,self.uh,bcs=[self.BC1,self.BC2,self.BC3])
        forward_solver = LinearVariationalSolver(forward_problem)
        forward_solver.solve()
    
    def function(self):
        self.ForcingFunction = project((tanh(asin(sin(15*self.x))*100)+1)/2,self.RHO) # Added a forcing function for edges.
    
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
        plt.savefig(f"/home/is420/MEng_project_controlled/PNG_shear/shear_{self.png_count}_{self.iter}.png")
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
    
    def rec_objective(self,J):
        self.objective_history.append(J)
    
    def rec_forces(self,s):
        self.upper_force.append(assemble(s[1,1]*ds(4)))
        self.lower_force.append(assemble(s[1,1]*ds(3)))
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
        mag = assemble(inner(self.uh,self.uh)**(0.5)*ds(2))
        
        # Forcing Function
        self.function()
        mag2 = assemble((self.rho_filt2-self.ForcingFunction)**2*ds(3)+(self.rho_filt2-self.ForcingFunction)**2*ds(4))
                
        # expansion forces
        # force constraint 1
        voigt_stress = self.voigt_vector()
        upper_force_constraint = assemble(dot(voigt_stress,self.n)*ds(4))
        lower_force_constraint = assemble(dot(voigt_stress,self.n)*ds(3))
        equillibrium_constraint = assemble(dot(voigt_stress,self.n)*ds(4)-dot(voigt_stress,self.n)*ds(3))
        
        self.rec_constraints(Volume,IDP,mag,mag2,upper_force_constraint,lower_force_constraint,equillibrium_constraint) # record all constraints for history

        
        return np.array((Volume,IDP,mag,mag2,upper_force_constraint,lower_force_constraint,equillibrium_constraint))
    
    
    # function to find jacobian
    def jacobian(self,x):
        # Initialisation
        self.rho.vector()[:] = x
        self.HH_filter()
        self.tanh_filter(0.5)
        
        # gradient of the colume constraint
        Volume = assemble(self.rho_filt2*dx)
        c = Control(self.rho)
        jac1 = compute_gradient(Volume,c)
        
        # gradient of the IDP Function
        IDP = assemble(((4.*self.rho_filt2*(1.-self.rho_filt2))**(1-self.alpha))*dx)
        jac2 = compute_gradient(IDP,c)
        
        # gradient of the magnitude on the RHS boundary
        self.forward()
        mag = assemble(inner(self.uh,self.uh)**(0.5)*ds(2))
        jac3 = compute_gradient(mag,c)
        
        # gradient of the forcing function
        self.function()
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
        
        return np.concatenate((jac1.dat.data,jac2.dat.data,jac3.dat.data,jac4.dat.data,jac5.dat.data,jac6.dat.data,jac7.dat.data))

def constraint_history(constraint,str,sub_iter):
    x = []
    for i in range(1,len(constraint)+1):
        x.append(i)
    fig, axes = plt.subplots()
    axes.plot(x,constraint)
    plt.xlabel('iteration')
    plt.ylabel('value')
    plt.title(str)
    plt.savefig(f"/home/is420/MEng_project_controlled/constraint_history/{str}_sub-iter_{sub_iter}.png")
    plt.close('all')

def function_plot(function,str,sub_iter):
    fig, axes = plt.subplots()
    collection = tripcolor(function, axes=axes, cmap='viridis')
    colorbar = fig.colorbar(collection);
    colorbar.set_label(r'$\rho$',fontsize=14,rotation=90)
    plt.title(f"sub_iteration: {sub_iter}")
    plt.gca().set_aspect(1)
    plt.savefig(f"/home/is420/MEng_project_controlled/newIDPresults/{str}_sub-iter{sub_iter}.png")
    plt.close('all')

def force_history(force_array,str,sub_iter):
    x = []
    for i in range(1,len(force_array)+1):
        x.append(i)
    fig, axes = plt.subplots()
    axes.plot(x,force_array)
    plt.xlabel('iteration')
    plt.ylabel('Force (N)')
    plt.title(str)
    plt.savefig(f"/home/is420/MEng_project_controlled/forces/{str}_sub-iter_{sub_iter}.png")
    plt.close('all')
    
def main():
    # plotting settings
    plt.style.use("default")
    # Times
    t1 = 0
    # ------ problem parameters ------------
    L, W = 5.0, 1.0 # domain size # original 5.0,1.0
    nx, ny = 180, 90 # mesh size 150, 60
    VolFrac = 0.4*L*W # Volume Fraction
    E_max, nu = 110e9, 0.3 # material properties #try changing the poissons ratio 
    p, E_min = 3.0, 1e-3 # SIMP Values
    t = Constant([2000,0]) # load # t = 2000

    # ----- setup BC, mesh, and function spaces ----
    mesh = RectangleMesh(nx,ny,L,W)
    x, y = SpatialCoordinate(mesh)
    V = VectorFunctionSpace(mesh,'CG',1)
    RHO = FunctionSpace(mesh,'CG',1)
    STRESS = FunctionSpace(mesh,'CG',1) ## stress function space
    BC1 = DirichletBC(V,Constant([0,0]),1)
    BC2 = DirichletBC(V,Constant([0,0]),3)
    BC3 = DirichletBC(V,Constant([0,0]),4)
    
    # radius for hh HH_filter
    r_min = 1.5*L/nx

    # ------ setup functions -----
    v = TestFunction(V)
    u = TrialFunction(V)
    uh = Function(V)
    rho = Function(RHO)
    rho_init = Function(RHO)
    rho_temp = Function(RHO) # used for initiliase function
    rho_filt = Function(RHO)
    find_area = Function(RHO).assign(Constant(1))
    
    # --- Function for random intialisation ---
    def Initialise_rho():
        # Define the size and spacing of the diamonds
        spacing_x = 0.1
        spacing_y = 0.1

        # Evaluate function to create a repeated diamond pattern
        with rho_init.dat.vec as rho_vec:
            for i, x in enumerate(mesh.coordinates.dat.data):
                # Compute the position within the repeated pattern
                pattern_x = x[0] % (2 * spacing_x)
                pattern_y = x[1] % (2 * spacing_y)
                
                # Compute the distance from the center of the current pattern cell
                distance_x = abs(pattern_x - spacing_x)
                distance_y = abs(pattern_y - spacing_y)
                
                # Assign values to create a diamond pattern
                if distance_x / spacing_x + distance_y / spacing_y <= 1:
                    rho_vec[i] = 1.0
                else:
                    rho_vec[i] = 0.0


        def HH_filter():
            r_min=0.12
            rhof, w = TrialFunction(RHO), TestFunction(RHO)
            A = (r_min**2)*inner(grad(rhof), grad(w))*dx+rhof*w*dx
            L = rho_init*w*dx
            bc = []
            solve(A==L, rho_temp, bcs=bc)
        
        HH_filter()
        
    def striped():
        # Define the angle of the stripes (in radians)
        stripe_angle = np.pi  # 30 degrees

        # Define the spacing between the stripes
        stripe_spacing = 0.2

        # Define the width of the central column
        central_column_width = 0.4

        # Evaluate function to create the combined pattern
        with rho_init.dat.vec as rho_vec:
            for i, x in enumerate(mesh.coordinates.dat.data):
                # Compute the position relative to the centerline of the stripes
                position_x = x[0] * np.cos(stripe_angle) + x[1] * np.sin(stripe_angle)
                
                # Compute the distance from the centerline of the stripes
                distance = abs(position_x - W / 2) % (2 * stripe_spacing)
                
                # Assign values based on the position relative to the central column
                if abs(x[1] - W / 2) <= central_column_width / 2:
                    rho_vec[i] = 1.0
                elif distance <= stripe_spacing:
                    rho_vec[i] = 1.0
                else:
                    rho_vec[i] = 0.0
                    
        def HH_filter():
            r_min=0.08
            rhof, w = TrialFunction(RHO), TestFunction(RHO)
            A = (r_min**2)*inner(grad(rhof), grad(w))*dx+rhof*w*dx
            L = rho_init*w*dx
            bc = []
            solve(A==L, rho_temp, bcs=bc)

        HH_filter()

    # ------ create optimiser -----
    rho_init.vector()[:] = 0.5
    function_plot(rho_init,"initialisation_rho_","inital") # plot the initialised rho domain
    x0 = rho_init.vector()[:].tolist() # Initial guess (rho initial)
    ub = np.ones(rho_init.vector()[:].shape).tolist() # upper bound of rho
    lb = np.zeros(rho_init.vector()[:].shape).tolist() # lower bound of rho
    
    # --- constraints (max & min)---
    Volume_Lower = 0
    Volume_Upper = VolFrac
    phi_max = 100
    phi_min = 0
    u_min = 0
    u_max = 20*1.5371025135048747e-08# Value seen with full titanium block pulled out. (multiplied)
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
    # ------------------------------

    cl = [Volume_Lower,phi_min,u_min,force_func_min,upper_force_min,lower_force_min,equillibrium_min] # lower bound of the constraints
    alpha = 0.0000001 # value of alpha
    beta = 2 # value of beta
    
    # ------- solve with sub-iterations -------
    for i in range(1,2): # set for only sub-iteration
        cu = [Volume_Upper,phi_max,u_max,force_func_max,upper_force_max,lower_force_max,equillibrium_max] #Update the constraints 
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
            max_iter = 95 ##90 - tested - MAX: 160 ---> 180 received alpha errors ;; currently at 150 ;; 140 ;; 138;; 84
        else:
            max_iter = 50 ##30 - tested - MAX: 60 currently at 50
        
        
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
        
        # write png files of final rho distribution
        fig, axes = plt.subplots()
        collection = tripcolor(rho, axes=axes, cmap='viridis')
        colorbar = fig.colorbar(collection);
        colorbar.set_label(r'$\rho$',fontsize=14,rotation=90)
        plt.title(f"sub_iteration: {i}")
        plt.gca().set_aspect(1)
        plt.savefig(f"/home/is420/MEng_project_controlled/newIDPresults/iteration_realVal{i}.png")
        plt.close("all")
        
        # create plots for the constraints
        constraint_history(obj.function_constraint,"function_constraint",i)
        constraint_history(obj.IDP_constraint,"IDP_constraint",i)
        constraint_history(obj.Volume_constraint,"Volume_constraint",i)
        constraint_history(obj.u_constraint,"u_constraint",i)
        constraint_history(obj.uf_constraint,"Upper_Force_Constraint",i)
        constraint_history(obj.lf_constraint,"Lower_Force_Constraint",i)
        constraint_history(obj.eq_constraint,"Equillibrium_Constraint",i)
        
        # create plot of the objective
        constraint_history(obj.objective_history,"Objective_History",i)
        
        # plot functions
        function_plot(obj.ForcingFunction,"ForcingFunctionDomain",i)
        
        # plot forces
        force_history(obj.upper_force,"Upper Force",i)
        force_history(obj.lower_force,"Lower Force",i)
        
        # ----- Final boundary Forces ----
        print(f"##################################")
        print(f"Upper force: {obj.upper_force[-1]}")
        print(f"Lower force: {obj.lower_force[-1]}")
        print(f"##################################")

if __name__ == '__main__':
    main()