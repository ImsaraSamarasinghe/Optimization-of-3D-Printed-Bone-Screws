# ------- Import libraries ------
from firedrake import *
from firedrake.adjoint import *
import cyipopt
import numpy as np
import matplotlib.pyplot as plt
import pickle
continue_annotation() # start tape

# ------- setup class --------
class screw3D:
    def __init__(self,E_max,nu,p,E_min,t,BC1,BC2,v,u,uh,rho,rho_filt,r_min,RHO,find_area,alpha,beta,STRESS,z,iter,mesh): # create class variables
        self.E_max = E_max
        self.nu = nu
        self.p = p
        self.E_min = E_min
        self.t = t
        self.BC1 = BC1
        self.BC2 = BC2
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
        self.z=z # x coordinates for the forcing function
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
        # --- Lists for storing objective history ---
        self.objective_history = []
        # --- Store rho ----
        self.rho_file = File(f"/home/is420/MEng_project_controlled/PNG_rho/rho_iter_{self.iter}.pvd")
        # ---- store force history ----
        self.expansion_force = []
        # ---  mesh --- 
        self.mesh = mesh
        # --- Facet Normals ---
        self.n = FacetNormal(self.mesh)
        # --- Forcing Function ---
        self.ForcingFunction = project((tanh(asin(sin(15*self.z))*100)+1)/2,self.RHO) # Added a forcing function for edges.
        
    #---------- class attributes for computations ---------------
    def HH_filter(self): # Helmholtz filter for densities
        rhof, w = TrialFunction(self.RHO), TestFunction(self.RHO)
        A = (self.r_min**2)*inner(grad(rhof), grad(w))*dx(2)+rhof*w*dx(2)
        L = self.rho*w*dx(2)
        bc = []
        solve(A==L, self.rho_filt, bcs=bc)

    def sigma(self,u): # stress Tensor
        return self.lambda_ * div(u) * Identity(3) + 2 * self.mu * self.epsilon(u)

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
        a = inner(self.sigma(self.u),self.epsilon(self.v))*dx(2)
        l = inner(self.t,self.v)*ds(4)
        forward_problem = LinearVariationalProblem(a,l,self.uh,bcs=[self.BC1,self.BC2])
        forward_solver = LinearVariationalSolver(forward_problem)
        forward_solver.solve()
    
    # function to fill in the voight stress vector
    def voigt_vector(self):
        s = self.sigma(self.uh)
        return as_vector([s[0,0],s[1,1],s[2,2]])
    
    # function to record constraints
    def rec_constraints(self,vol,IDP,u,func):
        self.function_constraint.append(func)
        self.IDP_constraint.append(IDP)
        self.Volume_constraint.append(vol)
        self.u_constraint.append(u)
    
    def rec_objective(self,J):
        self.objective_history.append(J)
        # pickle list
        with open('PickleFiles/objective.pkl', 'wb') as f:
            pickle.dump(self.objective_history, f)
    
    def rec_forces(self,force):
        self.expansion_force.append(force)
        # pickle list
        with open('PickleFiles/radial_force.pkl','wb') as f:
            pickle.dump(self.expansion_force, f)
    #------ end of class attributes for computations -----------

    # Function for the creation of the objective function
    def objective(self,x):
        self.rho.vector()[:] = x # Assign new densities to rho
        self.rho_file.write(self.rho) # write to rho_file
        self.HH_filter() # filter densities
        self.tanh_filter(0.5) # filter using tanh filter (eta = 0.5)
        self.forward() # forward problem
        self.d_file.write(self.uh) # write displacement to file
        # self.rec_stress() # removed does not work in 3D

        # ---- Find the new objective ----
        area = assemble(self.find_area*ds(6)) # find the area of the Wall
        s = self.sigma(self.uh) # stress tensor
        Force_xy = assemble(s[0,1]*ds(6))
        Force_xz = assemble(s[0,2]*ds(6))
        Force_yz = assemble(s[1,2]*ds(6))
        J = assemble((s[0,1]-Force_xy/area)**2*ds(6) + (s[0,2]-Force_xz/area)**2*ds(6) + (s[1,2]-Force_yz/area)**2*ds(6))
        
        self.rec_objective(J) # record objective
        
        return J
        
    # Function for the creation of the gradient
    def gradient(self,x):
        self.rho.vector()[:] = x # assign new densities to rho
        self.HH_filter() # filter rho
        self.tanh_filter(0.5) # filter using tanh filter (eta = 0.5)
        self.forward() # forward problem
        
        # --- find gradient ------
        area = assemble(self.find_area*ds(6)) # find the area of the Wall
        s = self.sigma(self.uh) # stress tensor
        Force_xy = assemble(s[0,1]*ds(6))
        Force_xz = assemble(s[0,2]*ds(6))
        Force_yz = assemble(s[1,2]*ds(6))
        J = assemble((s[0,1]-Force_xy/area)**2*ds(6) + (s[0,2]-Force_xz/area)**2*ds(6) + (s[1,2]-Force_yz/area)**2*ds(6))
        
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
        Volume = assemble(self.rho_filt2*dx(2))
        
        # Intermediate Density Penalisation
        IDP = assemble(((4.*self.rho_filt2*(1.-self.rho_filt2))**(1-self.alpha))*dx(2))
        
        # Magnitude constraint on the RHS boundary
        self.forward()
        mag = assemble(inner(self.uh,self.uh)**(0.5)*ds(4))
        
        # Forcing Function
        mag2 = assemble((self.rho_filt2-self.ForcingFunction)**2*ds(6))
                
        # expansion forces
        # force constraint 1
        voigt_stress = self.voigt_vector()
        force_constraint = assemble(dot(voigt_stress,self.n)*ds(6)) # radial force

        self.rec_forces(force_constraint) # record radial force
        self.rec_constraints(Volume,IDP,mag,mag2) # record all constraints for history
        
        return np.array((Volume,IDP,mag,mag2,force_constraint))
    
    
    # function to find jacobian
    def jacobian(self,x):
        # Initialisation
        self.rho.vector()[:] = x
        self.HH_filter()
        self.tanh_filter(0.5)
        
        # gradient of the colume constraint
        Volume = assemble(self.rho_filt2*dx(2))
        c = Control(self.rho)
        jac1 = compute_gradient(Volume,c)
        
        # gradient of the IDP Function
        IDP = assemble(((4.*self.rho_filt2*(1.-self.rho_filt2))**(1-self.alpha))*dx(2))
        jac2 = compute_gradient(IDP,c)
        
        # gradient of the magnitude on the RHS boundary
        self.forward()
        mag = assemble(inner(self.uh,self.uh)**(0.5)*ds(4))
        jac3 = compute_gradient(mag,c)
        
        # gradient of the forcing function
        mag2 = assemble((self.rho_filt2-self.ForcingFunction)**2*ds(6))
        jac4 = compute_gradient(mag2,c)
        
        # expansion force - gradient
        voigt_stress = self.voigt_vector()
        force_constraint = assemble(dot(voigt_stress,self.n)*ds(6))
        jac5 = compute_gradient(force_constraint,c)
        
        return np.concatenate((jac1.dat.data,jac2.dat.data,jac3.dat.data,jac4.dat.data,jac5.dat.data))
    
def main():
    # plotting settings
    plt.style.use("default")
    # Times
    t1 = 0
    # ----- MESH & coordinates-------
    mesh = Mesh('screw_less_fine.msh') # import mesh
    x, y, z = SpatialCoordinate(mesh) # coordinate system
    # ----- Function spaces -----
    V = VectorFunctionSpace(mesh,'CG',1) # main function space (VECTOR)
    RHO = FunctionSpace(mesh,'CG',1) # density varibales function space (SCALAR)
    STRESS = FunctionSpace(mesh,'CG',1) # stress function space (SCALAR)
    # ---- Dirichlet Boundary Conditions -----
    BC1 = DirichletBC(V,Constant([0,0,0]),5) # End
    BC2 = DirichletBC(V,Constant([0,0,0]),6) # Wall
    # ---- Filter Radius -----
    r_min = 1.5*5/164 # WILL NEED CHANGE
    # ------ Functions -----
    v = TestFunction(V)
    u = TrialFunction(V)
    uh = Function(V)
    rho = Function(RHO)
    rho_init = Function(RHO)
    rho_temp = Function(RHO) # used for initiliase function
    rho_filt = Function(RHO)
    find_area = Function(RHO).assign(Constant(1)) # function space used for evaluating areas and volumes
    # ------ problem parameters ------------
    VolFrac = assemble(0.4*find_area*dx(2)) # volume fraction
    E_max, nu = 110e9, 0.3 # material properties #try changing the poissons ratio 
    p, E_min = 3.0, 1e-3 # SIMP Values
    t = Constant([0,0,2000]) # load t=2000 in +z

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
    rho_init.vector()[:] = 0.5 # initialise with 0.5
    x0 = rho_init.vector()[:].tolist() # Initial guess (rho initial)
    ub = np.ones(rho_init.vector()[:].shape).tolist() # upper bound of rho
    lb = np.zeros(rho_init.vector()[:].shape).tolist() # lower bound of rho
    
    # --- constraints (max & min)---
    Volume_Lower = 0
    Volume_Upper = VolFrac
    phi_max = 100
    phi_min = 0
    u_min = 0
    u_max = 20*6.477e-9# Value seen with full titanium block pulled out. (multiplied)
    force_func_max = 1
    force_func_min = -1
    # ----- Radial force constraints -----
    radial_force_min = 200
    radial_force_max = 10000

    cl = [Volume_Lower,phi_min,u_min,force_func_min,radial_force_min] # lower bound of the constraints
    alpha = 0.0000001 # value of alpha
    beta = 2 # value of beta
    
    # ------- solve with sub-iterations -------
    for i in range(1,2): # set for only sub-iteration
        cu = [Volume_Upper,phi_max,u_max,force_func_max,radial_force_max] #Update the constraints 
        obj = screw3D(E_max,nu,p,E_min,t,BC1,BC2,v,u,uh,rho,rho_filt,r_min,RHO,find_area,alpha,beta,STRESS,z,i,mesh) # create object class
        
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
            max_iter = 95 ##90 - tested - MAX: 160 ---> 180 received alpha errors ;; currently at 150 ;; 140 ;; 138;; 84;; 95
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

if __name__ == '__main__':
    main()