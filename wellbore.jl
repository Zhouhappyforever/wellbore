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
# Plane strain poroelasticity: εzz = 0, σzz ≠ 0

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================
# Material properties
E       = 20.0e6     # Young's modulus (Pa)
nu      = 0.2        # Poisson's ratio
B       = 0.8        # Biot coefficient
M       = 1.0e9      # Biot modulus (Pa)
k       = 1.0e-3     # Permeability (m^2)
mu_f    = 1.0e-3     # Fluid viscosity (Pa·s)

# Loading conditions
Pb      = 31.5e6     # Wellbore pressure (Pa)
p0_val  = 20.0e6     # Initial pore pressure (Pa)

# Time stepping parameters
T         = 0.0005       # Final time (s)
num_steps = 100          # Number of time steps (increase for smoother output)
dt        = T / num_steps # Time step size (s)

# ============================================================================
# DERIVED MATERIAL PROPERTIES
# ============================================================================
lambda = E * nu / ((1 + nu) * (1 - 2*nu))  # First Lamé parameter (Pa)
mu_s   = E / (2*(1 + nu))                  # Shear modulus (Pa)
k_mu   = k / mu_f                          # Hydraulic conductivity

dirichlet_tags = ["top_bottom","wellbore"]

# ============================================================================
# OUTPUT & MESH SETUP
# ============================================================================
output_dir = "results"
if !isdir(output_dir)
  mkdir(output_dir)
end

model  = GmshDiscreteModel("wellbore.msh")
labels = get_face_labeling(model)
println("Entities tagged as top_bottom: ", findall(labels.tag_to_name .== "top_bottom"))
println("Entities tagged as wellbore:    ", findall(labels.tag_to_name .== "wellbore"))

# ============================================================================
# DOMAIN & INTEGRATION
# ============================================================================
Ω   = Triangulation(model)
dΩ  = Measure(Ω,2)
Γb  = BoundaryTriangulation(model)
dΓb = Measure(Γb,2)

# ============================================================================
# FINITE ELEMENT SPACES
# ============================================================================
order_u = 2; order_p = 1
reffe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, order_u)
reffe_p = ReferenceFE(lagrangian, Float64, order_p)

δu      = TestFESpace(model, reffe_u, conformity=:H1, dirichlet_tags=["top_bottom"])
u_trial = TrialFESpace(δu, x->VectorValue(0.0, 0.0))
δp      = TestFESpace(model, reffe_p, conformity=:H1, dirichlet_tags=["wellbore"])
p_trial = TrialFESpace(δp, x->Pb)

Y = MultiFieldFESpace([δu, δp])

# ============================================================================
# CONSTITUTIVE RELATIONS
# ============================================================================
function sigma(u)
  ε = symmetric_gradient(u)
  I = TensorValue(1.0, 0.0, 0.0, 1.0)
  return lambda*tr(ε)*I + 2*mu_s*ε
end

function sigma_zz(u)
  return lambda * tr(symmetric_gradient(u))
end

# ============================================================================
# INITIAL & TRANSIENT SPACES
# ============================================================================
u0     = VectorValue(0.0, 0.0)
p0_val = p0_val

u_t = TransientTrialFESpace(δu)
p_t = TransientTrialFESpace(δp)
X_t = MultiFieldFESpace([u_t, p_t])

# ============================================================================
# WEAK FORMULATION
# ============================================================================
a(t, (u, p), (δu, δp)) = ∫(
  symmetric_gradient(δu) ⊙ sigma(u) -         # solid mechanics
  B * divergence(δu) * p +                     # Biot coupling (solid <- fluid)
  δp * (1/M) * ∂t(p) +                         # fluid storage
  ∇(δp) ⋅ (k_mu * ∇(p)) +                       # Darcy flow
  δp * B * divergence(∂t(u))                   # Biot coupling (fluid <- solid)
) * dΩ

l(t, (δu, δp)) = ∫(                                # zero Neumann traction
  δu ⋅ VectorValue(0.0, 0.0)
) * dΓb

res(t, (u, p), (δu, δp)) = a(t, (u, p), (δu, δp)) - l(t, (δu, δp))

# ============================================================================
# TRANSIENT PROBLEM SETUP
# ============================================================================
op = TransientFEOperator(res, X_t, Y)

# ============================================================================
# SOLVER CONFIGURATION
# ============================================================================
ls         = LUSolver()  # direct solver
nls        = NLSolver(ls, method=:newton, iterations=10, show_trace=false)
theta      = 1.0         # backward Euler
ode_solver= ThetaMethod(nls, dt, theta)

# ============================================================================
# INITIAL SOLUTION
# ============================================================================
uh0 = interpolate_everywhere([u0, p0_val], X_t(0.0))

# ============================================================================
# TIME MARCHING & OUTPUT
# ============================================================================
sol = solve(ode_solver, op, 0.0, T, uh0)

createpvd(joinpath(output_dir, "results")) do pvd
  # initial
  disp0, pres0 = uh0
  pvd[0.0] = createvtk(Ω, joinpath(output_dir, "results_0.vtu"),
                       cellfields=["displacement"=>disp0, "pressure"=>pres0])
  for (tn, uhn) in sol
    println("Writing results at t = $tn")
    disp_n, pres_n = uhn
    pvd[tn] = createvtk(Ω, joinpath(output_dir, "results_$(tn).vtu"),
                         cellfields=["displacement"=>disp_n,   "pressure"=>pres_n])
  end
end

println("Wellbore poroelastic simulation completed! Results in '$output_dir'.")

