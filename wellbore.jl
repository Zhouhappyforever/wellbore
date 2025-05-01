using Gridap                # Main Gridap package
using Gridap.Geometry       # Mesh and geometry handling
using Gridap.FESpaces       # Finite element spaces
using Gridap.MultiField     # Coupled multi-physics problems
using Gridap.Io             # I/O operations
using Gridap.Fields         # Field operations
using Gridap.TensorValues   # Tensor operations
using Gridap.ODEs           # Time-dependent problems
using Gridap.CellData       # Cell data operations and projection
using WriteVTK              # VTK output for visualization
using GridapGmsh            # Gmsh mesh integration

# ============================================================================
# PROBLEM DESCRIPTION
# ============================================================================
# Plane strain poroelasticity around a wellbore subjected to an internal pressure.

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================
E      = 20e6          # Young's modulus (Pa)
nu     = 0.2           # Poisson's ratio
B      = 0.8           # Biot coefficient
M      = 1e9           # Biot modulus (Pa)
k      = 1e-3          # Permeability (m^2)
mu_f   = 1e-3         # Fluid viscosity (Pa·s)

# Boundary pressures
Pb     = 31.5e6       # Wellbore pressure (Pa)
p0     = 20e6         # Initial pore pressure (Pa)

# Time stepping
T       = 5e-4         # Final time (s)
num_steps = 100        # Number of time steps
dt      = T/num_steps # Time step size (s)

# ============================================================================
# DERIVED MATERIAL PROPERTIES
# ============================================================================
lambda = E*nu/((1+nu)*(1-2*nu))   # First Lamé parameter (Pa)
mu     = E/(2*(1+nu))             # Shear modulus (Pa)
k_mu   = k/mu_f                   # Hydraulic conductivity

# ============================================================================
# SETUP OUTPUT AND MESH
# ============================================================================
output_dir = "results"
if !isdir(output_dir)
    mkdir(output_dir)
end

model = GmshDiscreteModel("wellbore.msh")
# Optional: export mesh for quick inspection
writevtk(model, "model_mesh")

# Print boundary tags for verification
labels = get_face_labeling(model)
println("top_bottom tags: ", findall(labels.tag_to_name .== "top_bottom"))
println("wellbore tags:    ", findall(labels.tag_to_name .== "wellbore"))

# ============================================================================
# DOMAIN AND INTEGRATION SETUP
# ============================================================================
degree = 2
Ω      = Triangulation(model)
dΩ     = Measure(Ω, degree)
Γb     = BoundaryTriangulation(model)
dΓb    = Measure(Γb, degree)

# ============================================================================
# FINITE ELEMENT SPACES
# ============================================================================
order_u = 2  # Quadratic displacement
tmp_order_p = 1  # Linear pressure (LBB-compatible)

reffe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, order_u)
reffe_p = ReferenceFE(lagrangian, Float64, tmp_order_p)

# Test spaces with Dirichlet tags
dδu = TestFESpace(model, reffe_u, conformity=:H1, dirichlet_tags=["top_bottom"])
dδp = TestFESpace(model, reffe_p, conformity=:H1, dirichlet_tags=["wellbore"])
Y    = MultiFieldFESpace([dδu, dδp])

# Transient trial spaces (time‑dependent BCs handled implicitly)
u_t = TransientTrialFESpace(dδu)
p_t = TransientTrialFESpace(dδp)
X_t = MultiFieldFESpace([u_t, p_t])

# ============================================================================
# INITIAL CONDITIONS
# ============================================================================
u0  = VectorValue(0.0,0.0)
uh0 = interpolate_everywhere([u0, p0], X_t(0.0))

# ============================================================================
# CONSTITUTIVE RELATIONS
# ============================================================================
function sigma(u)
  ε = symmetric_gradient(u)
  I = TensorValue(1.0,0.0,0.0,1.0)
  lambda*tr(ε)*I + 2*mu*ε
end

# ============================================================================
# WEAK FORMULATION
# ============================================================================
a(t, (u,p), (δu,δp)) = ∫(
    symmetric_gradient(δu) ⊙ sigma(u)           # elasticity
  - B*divergence(δu)*p                          # Biot coupling (1)
  + δp*(1/M)*∂t(p)                              # fluid storage
  + ∇(δp) ⋅ (k_mu*∇(p))                         # Darcy flow
  + δp*B*divergence(∂t(u))                     # Biot coupling (2)
)*dΩ

l(t, (δu,δp)) = ∫(
  δu ⋅ VectorValue(0.0,0.0)                    # zero traction
)*dΓb

res(t, up, tup) = a(t,up,tup) - l(t,tup)

# ============================================================================
# TRANSIENT PROBLEM SETUP
# ============================================================================
op = TransientFEOperator(res, X_t, Y)
ls = LUSolver()
nls = NLSolver(ls, method=:newton, iterations=10, show_trace=false)
θ = 1.0
ode_solver = ThetaMethod(nls, dt, θ)

# ============================================================================
# SOLVE
# ============================================================================
sol = solve(ode_solver, op, 0.0, T, uh0)

# ============================================================================
# RESULTS VISUALIZATION
# ============================================================================
createpvd(joinpath(output_dir, "results")) do pvd
  # initial snapshot
  disp0,pres0 = uh0
  pvd[0.0] = createvtk(Ω, joinpath(output_dir, "results_0.vtu"),
                       cellfields=["displacement"=>disp0,
                                   "pressure"   =>pres0])
  # time loop
  for (tn, uhn) in sol
    println("Writing t = $tn")
    disp,pres = uhn
    pvd[tn] = createvtk(Ω, joinpath(output_dir, "results_$(tn).vtu"),
                        cellfields=["displacement"=>disp,
                                    "pressure"   =>pres])
  end
end

println("Simulation completed. Results saved in '$output_dir/'.")
