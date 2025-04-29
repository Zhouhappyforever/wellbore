#boundary Import necessary libraries
using Gridap                # Main Gridap package for finite element analysis
using Gridap.Geometry       # For mesh and geometry handling
using Gridap.FESpaces       # For finite element spaces
using Gridap.MultiField     # For coupled multi-physics problems
using Gridap.Io             # For input/output operations
using Gridap.Fields         # For field operations
using Gridap.TensorValues   # For tensor operations
using Gridap.ODEs           # For time-dependent problems
using Gridap.CellData       # For cell data operations and projection
using WriteVTK              # For VTK file output (visualization)
using GridapGmsh            # For Gmsh mesh integration

# ============================================================================
# PROBLEM DESCRIPTION
# ============================================================================
# This is a plane strain poroelasticity formulation
# In plane strain, we assume εzz = 0 (no strain in z-direction)
# but σzz ≠ 0 (stress in z-direction can exist)
# This is appropriate for modeling soil/rock layers where the z-dimension is 
# constrained but stresses can develop in that direction

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

# Material properties
E = 20.0e6         # Young's modulus (Pa)
nu = 0.2          # Poisson's ratio
B = 0.8           # Biot coefficient (coupling between fluid pressure and solid stress)
M = 1.0e9         # Biot modulus (Pa) - related to fluid and solid compressibility
k = 1.0e-3        # Permeability (m^2) - how easily fluid flows through the medium
mu = 1.0e-3       # Fluid viscosity (Pa·s)

# Loading conditions
Pb = 31.5e6
p0 = 20.0e6

# Time stepping parameters
T = 0.0005          # Final time (s)
num_steps = 3   # Number of time steps
dt = T / num_steps # Time step size (s)

# ============================================================================
# DERIVED MATERIAL PROPERTIES
# ============================================================================
# Calculate Lamé parameters for plane strain formulation
# For plane strain, we use the same Lamé parameters as in 3D
lambda = E * nu / ((1 + nu) * (1 - 2 * nu))  # First Lamé parameter (Pa)
mu = E / (2 * (1 + nu))                      # Second Lamé parameter (shear modulus) (Pa)
k_mu = k / mu                                # Hydraulic conductivity (permeability/viscosity)

dirichlet_tags = ["top_bottom", "wellbore"]
#####################################
######## ADD YOU CODE HERE ##########
#####################################
# -------------------------------------------------------------------------
# OUTPUT AND MESH SETUP
# -------------------------------------------------------------------------
output_dir = "results"
if !isdir(output_dir)
    mkdir(output_dir)
end

# Load the wellbore mesh
model = GmshDiscreteModel("wellbore.msh")

# Debug: print boundary entity tags
labels = get_face_labeling(model)
println("Entities tagged as top_bottom: ", findall(labels.tag_to_name .== "top_bottom"))
println("Entities tagged as wellbore: ",   findall(labels.tag_to_name .== "wellbore"))

# -------------------------------------------------------------------------
# DOMAIN AND INTEGRATION SETUP
# -------------------------------------------------------------------------
Ω = Triangulation(model)
dΩ = Measure(Ω, 2)            # 2nd order quadrature
gΓ = BoundaryTriangulation(model)
dΓ = Measure(gΓ, 2)

# -------------------------------------------------------------------------
# FINITE ELEMENT SPACES
# -------------------------------------------------------------------------
order_u = 2  # P2 for displacement
order_p = 1  # P1 for pressure

reffe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, order_u)
reffe_p = ReferenceFE(lagrangian, Float64, order_p)

# Displacement test/trial spaces with Dirichlet BC on top_bottom
δu = TestFESpace(model, reffe_u, conformity=:H1, dirichlet_tags=["top_bottom"])
u_trial = TrialFESpace(δu, x -> VectorValue(0.0, 0.0))

# Pressure test/trial spaces with Dirichlet BC on wellbore (constant Pb)
δp = TestFESpace(model, reffe_p, conformity=:H1, dirichlet_tags=["wellbore"])
p_trial = TrialFESpace(δp, x -> Pb)

# Combine into multi-field spaces
Y = MultiFieldFESpace([δu, δp])

# -------------------------------------------------------------------------
# CONSTITUTIVE RELATIONS
# -------------------------------------------------------------------------
function sigma(u)
    ε = symmetric_gradient(u)
    I = TensorValue(1.0,0.0,0.0,1.0)
    return lambda * tr(ε) * I + 2 * mu * ε
end

function sigma_zz(u)
    ε = symmetric_gradient(u)
    return lambda * tr(ε)
end

# -------------------------------------------------------------------------
# INITIAL CONDITIONS
# -------------------------------------------------------------------------
u0 = VectorValue(0.0, 0.0)
p0_val = p0

# -------------------------------------------------------------------------
# TRANSIENT SPACES
# -------------------------------------------------------------------------
u_t = TransientTrialFESpace(δu)
p_t = TransientTrialFESpace(δp)
X_t = MultiFieldFESpace([u_t, p_t])

# -------------------------------------------------------------------------
# WEAK FORMULATION
# -------------------------------------------------------------------------
a(t, (u,p), (δu, δp)) = ∫(
    symmetric_gradient(δu) ⊙ sigma(u) -                           # Solid mechanics
    (B * divergence(δu) * p) +                                   # Biot coupling (solid<-fluid)
    δp * (1/M) * ∂t(p) +                                          # Storage
    ∇(δp) ⋅ (k_mu * ∇(p)) +                                       # Darcy flow
    δp * B * divergence(∂t(u))                                   # Biot coupling (fluid<-solid)
) * dΩ

l(t, (δu, δp)) = ∫(   # No external tractions on wellbore beyond Dirichlet p
    zero(δu)                                                      # zero to fill signature
) * boundary(dΓ)

res(t, up, tup) = a(t, up, tup) - l(t, tup)

# -------------------------------------------------------------------------
# SOLVER SETUP
# -------------------------------------------------------------------------
op = TransientFEOperator(res, X_t, Y)
ls = LUSolver()
nls = NLSolver(ls, method=:newton, iterations=10, show_trace=false)
θ = 1.0                      # Backward Euler
ode_solver = ThetaMethod(nls, dt, θ)

# -------------------------------------------------------------------------
# INITIAL SOLUTION
# -------------------------------------------------------------------------
uh0 = interpolate_everywhere([u0, p0_val], X_t(0.0))

# -------------------------------------------------------------------------
# TIME MARCHING AND OUTPUT
# -------------------------------------------------------------------------
sol = solve(ode_solver, op, 0.0, T, uh0)

createpvd(joinpath(output_dir, "results")) do pvd
    # initial
    disp0, pres0 = uh0\    
    pvd[0.0] = createvtk(Ω, joinpath(output_dir, "results_0.vtu"),
                         cellfields=["displacement"=>disp0, "pressure"=>pres0])
    for (tn, uhn) in sol
        println("Writing results at t = $tn")
        disp_n, pres_n = uhn
        pvd[tn] = createvtk(Ω, joinpath(output_dir, "results_$(tn).vtu"),
                            cellfields=["displacement"=>disp_n, "pressure"=>pres_n])
    end
end

println("Wellbore poroelastic simulation completed! Results in '$output_dir'.")
