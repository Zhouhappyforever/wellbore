using Gridap
using Gridap.Geometry
using Gridap.FESpaces
using Gridap.MultiField
using Gridap.Io
using Gridap.Fields
using Gridap.TensorValues
using Gridap.ODEs
using Gridap.CellData
using WriteVTK
using GridapGmsh

# -------------------------------------------------------------------
# 1) PARAMETERS & MATERIAL PROPERTIES
# -------------------------------------------------------------------
E      = 20.0e6           # Young's modulus (Pa)
nu     = 0.2              # Poisson's ratio
B      = 0.8              # Biot coefficient
M      = 1.0e9            # Biot modulus (Pa)
k      = 1.0e-3           # Permeability (m^2)
mu_f   = 1.0e-3           # Fluid viscosity (Pa·s)

Pb     = 31.5e6           # Wellbore pressure (Pa)
p0     = 20.0e6           # Initial pore pressure (Pa)

T       = 0.0005          # Final time (s)
num_steps = 100           # Number of time steps
dt      = T / num_steps   # Time step size (s)

# Derived
λ      = E * nu / ((1 + nu)*(1 - 2*nu))  # First Lamé parameter
μ      = E / (2*(1 + nu))               # Shear modulus
kμ     = k / mu_f                       # Hydraulic conductivity

# -------------------------------------------------------------------
# 2) MESH & OUTPUT SETUP
# -------------------------------------------------------------------
output_dir = "results"
isdir(output_dir) || mkdir(output_dir)

model  = GmshDiscreteModel("wellbore.msh")
labels = get_face_labeling(model)
println("Entities tagged as top_bottom: ", findall(labels.tag_to_name .== "top_bottom"))
println("Entities tagged as wellbore:    ", findall(labels.tag_to_name .== "wellbore"))

# -------------------------------------------------------------------
# 3) DOMAIN & INTEGRATION
# -------------------------------------------------------------------
Ω   = Triangulation(model)
dΩ  = Measure(Ω, 2)

Γb  = BoundaryTriangulation(model)
dΓb = Measure(Γb, 2)

# -------------------------------------------------------------------
# 4) FINITE ELEMENT SPACES & BCs
# -------------------------------------------------------------------
order_u, order_p = 2, 1
reffe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, order_u)
reffe_p = ReferenceFE(lagrangian, Float64, order_p)

# Displacement test/trial (u=0 on top_bottom)
δu      = TestFESpace(model, reffe_u, conformity=:H1,
                     dirichlet_tags=["top_bottom"])
u_trial = TrialFESpace(δu, x -> VectorValue(0.0,0.0))

# Pressure test/trial (p=Pb on wellbore)
δp      = TestFESpace(model, reffe_p, conformity=:H1,
                     dirichlet_tags=["wellbore"])
p_trial = TrialFESpace(δp, x -> Pb)

# Combined *test* and *transient‐trial* spaces
Y   = MultiFieldFESpace([δu, δp])
u_t = TransientTrialFESpace(u_trial)
p_t = TransientTrialFESpace(p_trial)
X_t = MultiFieldFESpace([u_t, p_t])

# -------------------------------------------------------------------
# 5) INITIAL CONDITION
# -------------------------------------------------------------------
u0  = VectorValue(0.0,0.0)
uh0 = interpolate_everywhere([u0, p0], X_t(0.0))

# -------------------------------------------------------------------
# 6) CONSTITUTIVE RELATIONS
# -------------------------------------------------------------------
function σ(u)
  ε = symmetric_gradient(u)
  I = TensorValue(1.0,0.0,0.0,1.0)
  λ*tr(ε)*I + 2*μ*ε
end

# -------------------------------------------------------------------
# 7) WEAK FORMS
# -------------------------------------------------------------------
a(t, up, tup) = begin
  u,p   = up
  δu,δp = tup
  ∫(
    symmetric_gradient(δu) ⊙ σ(u) -        # solid mechanics
    B*divergence(δu)*p +                  # Biot coupling
    δp*(1/M)*∂t(p) +                      # fluid storage
    ∇(δp)⋅(kμ*∇(p)) +                     # Darcy flow
    δp*B*divergence(∂t(u))               # coupling (fluid <- solid)
  ) * dΩ
end

# RIGHT: use the constant zero‐vector so that δu⋅zero_vec is a proper vector‐valued integrand
const zero_vec = VectorValue(0.0,0.0)
l(t, (δu,δp)) = ∫( δu ⋅ zero_vec ) * dΓb




res(t, (u,p), (δu,δp)) = a(t,(u,p),(δu,δp)) - l(t,(δu,δp))



# -------------------------------------------------------------------
# 8) SOLVER SETUP & TIME MARCHING
# -------------------------------------------------------------------
op         = TransientFEOperator(res, X_t, Y)
ls         = LUSolver()
nls        = NLSolver(ls, method=:newton, iterations=10, show_trace=false)
θ          = 1.0                        # backward Euler
ode_solver = ThetaMethod(nls, dt, θ)

sol = solve(ode_solver, op, 0.0, T, uh0)

# -------------------------------------------------------------------
# 9) OUTPUT (ParaView .pvd + .vtu)
# -------------------------------------------------------------------
createpvd(joinpath(output_dir,"results")) do pvd
  disp0, pres0 = uh0
  pvd[0.0] = createvtk(Ω, joinpath(output_dir,"results_0.vtu"),
                       cellfields=["displacement"=>disp0,
                                   "pressure"   =>pres0])
  for (tn, uhn) in sol
    println("Writing results at t = $tn")
    disp_n, pres_n = uhn
    pvd[tn] = createvtk(Ω, joinpath(output_dir,"results_$(tn).vtu"),
                        cellfields=["displacement"=>disp_n,
                                    "pressure"   =>pres_n])
  end
end

println("Wellbore poroelastic simulation completed! Results in '$output_dir'.")
