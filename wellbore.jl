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
E         = 20.0e6     # Young's modulus (Pa)
nu        = 0.2        # Poisson's ratio
B         = 0.8        # Biot coefficient
M         = 1.0e9      # Biot modulus (Pa)
k         = 1.0e-3     # Permeability (m^2)
mu_f      = 1.0e-3     # Fluid viscosity (Pa·s)

Pb        = 31.5e6     # Wellbore pressure (Pa)
p0        = 20.0e6     # Initial pore pressure (Pa)

T         = 0.0005     # Final time (s)
num_steps = 100        # Number of time steps
dt        = T / num_steps # Time step size (s)

# ============================================================================
# DERIVED MATERIAL PROPERTIES
# ============================================================================
lambda    = E * nu / ((1 + nu) * (1 - 2*nu))  # First Lamé parameter (Pa)
mu_s      = E / (2*(1 + nu))                  # Shear modulus (Pa)
k_mu      = k / mu_f                          # Hydraulic conductivity

# ============================================================================
# MESH & OUTPUT SETUP
# ============================================================================
output_dir = "results"
if !isdir(output_dir)
  mkdir(output_dir)
end

model      = GmshDiscreteModel("wellbore.msh")
labels     = get_face_labeling(model)
println("Entities tagged as top_bottom: ", findall(labels.tag_to_name .== "top_bottom"))
println("Entities tagged as wellbore:    ", findall(labels.tag_to_name .== "wellbore"))

# ============================================================================
# DOMAIN & INTEGRATION
# ============================================================================
Ω          = Triangulation(model)
dΩ         = Measure(Ω,2)
Γb         = BoundaryTriangulation(model)
dΓb        = Measure(Γb,2)

# ============================================================================
# FINITE ELEMENT SPACES
# ============================================================================
order_u    = 2  # quadratic displacement
order_p    = 1  # linear pressure

reffe_u    = ReferenceFE(lagrangian, VectorValue{2,Float64}, order_u)
reffe_p    = ReferenceFE(lagrangian, Float64, order_p)

# Test spaces with Dirichlet tags
δu         = TestFESpace(model, reffe_u, conformity=:H1, dirichlet_tags=["top_bottom"])
δp         = TestFESpace(model, reffe_p, conformity=:H1, dirichlet_tags=["wellbore"])

# Trial spaces to enforce Dirichlet BCs
u          = TrialFESpace(δu, x -> VectorValue(0.0, 0.0))  # u_x=u_y=0 on top_bottom
p          = TrialFESpace(δp, x -> Pb)                     # p=Pb on wellbore

# Combine into multi-field spaces
Y          = MultiFieldFESpace([δu, δp])  # Test space

# ============================================================================
# CONSTITUTIVE RELATIONS
# ============================================================================
function sigma(u)
  ε = symmetric_gradient(u)
  I = TensorValue(1.0, 0.0, 0.0, 1.0)
  return lambda * tr(ε) * I + 2*mu_s * ε
end

function sigma_zz(u)
  return lambda * tr(symmetric_gradient(u))
end

# ============================================================================
# TRANSIENT TRIAL SPACES & INITIAL CONDITIONS
# ============================================================================
u0 = VectorValue(0.0, 0.0)
p0_val = 20.0e6  # initial pore pressure (Pa)  # corrected distinct variable

u_t       = TransientTrialFESpace(δu)  # uses u trial for BCs
p_t       = TransientTrialFESpace(δp)  # uses p trial for BCs
X_t       = MultiFieldFESpace([u_t, p_t])
uh0 = interpolate_everywhere([u0, p0_val], X_t(0.0))  # initial condition with correct pressure)  # initial solution

# ============================================================================
# WEAK FORMULATION
# ============================================================================
a(t, (u, p), (δu, δp)) = ∫(
  symmetric_gradient(δu) ⊙ sigma(u) -   # solid mechanics
  B * divergence(δu) * p +               # Biot coupling (solid <- fluid)
  δp * (1/M) * ∂t(p) +                   # fluid storage
  ∇(δp) ⋅ (k_mu * ∇(p)) +                # Darcy flow
  δp * B * divergence(∂t(u))            # Biot coupling (fluid <- solid)
) * dΩ

l(t, (δu, δp)) = ∫(                         # zero Neumann traction
  δu ⋅ VectorValue(0.0, 0.0)
) * dΓb

res(t, up, tup) = a(t, up, tup) - l(t, tup)

# ============================================================================
# TRANSIENT PROBLEM SETUP & SOLVER
# ============================================================================
op          = TransientFEOperator(res, X_t, Y)
ls          = LUSolver()  # direct solver
nls         = NLSolver(ls, method=:newton, iterations=10, show_trace=false)
theta       = 1.0         # backward Euler
ode_solver = ThetaMethod(nls, dt, theta)

# ============================================================================
# TIME MARCHING & OUTPUT
# ============================================================================
sol = solve(ode_solver, op, 0.0, T, uh0)
createpvd(joinpath(output_dir, "results")) do pvd
  # initial state
  disp0, pres0 = uh0
  pvd[0.0] = createvtk(Ω, joinpath(output_dir, "results_0.vtu"),
                         cellfields=["displacement"=>disp0,
                                     "pressure"    =>pres0])
  # time steps
  for (i, (tn, uhn)) in enumerate(sol)
    println("Writing results at t = $tn")
    disp_n, pres_n = uhn
    pvd[tn]     = createvtk(Ω, joinpath(output_dir, "results_$(i).vtu"),
                             cellfields=["displacement"=>disp_n,
                                         "pressure"    =>pres_n])
  end
end

println("Wellbore poroelastic simulation completed! Results in '$output_dir'.")
