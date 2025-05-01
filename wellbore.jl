using Gridap
using Gridap.Geometry
using Gridap.FESpaces
using Gridap.MultiField
using Gridap.TensorValues
using Gridap.ODEs
using WriteVTK
using GridapGmsh   # for GmshDiscreteModel

# 1) PARAMETERS & MATERIAL PROPERTIES
E, nu, B, M       = 20e6, 0.2, 0.8, 1e9
k, mu_f           = 1e-3, 1e-3
Pb,  p0           = 31.5e6, 20e6
T,   num_steps    = 5e-4, 100
dt                = T/num_steps
λ   = E*nu/((1+nu)*(1-2*nu))    # 1st Lamé
μ   = E/(2*(1+nu))              # shear modulus
kμ  = k/mu_f                    # hydraulic conductivity

# 2) MESH & LABELS
model  = GmshDiscreteModel("wellbore.msh")
labels = get_face_labeling(model)
println("top_bottom tags: ", findall(labels.tag_to_name .== "top_bottom"))
println("wellbore tags:    ", findall(labels.tag_to_name .== "wellbore"))

# 3) DOMAIN & MEASURES
Ω   = Triangulation(model);      dΩ  = Measure(Ω, 2)
Γb  = BoundaryTriangulation(model); dΓb = Measure(Γb, 1)

# 4) SPACES & BC‐CONSTRAINED TRIAL SPACES
order_u, order_p = 2, 1
reffe_u = ReferenceFE(lagrangian, VectorValue{2,Float64}, order_u)
reffe_p = ReferenceFE(lagrangian, Float64,          order_p)

# test (δu, δp)
δu      = TestFESpace(model, reffe_u, conformity=:H1,
                      dirichlet_tags=["top_bottom"])
δp      = TestFESpace(model, reffe_p, conformity=:H1,
                      dirichlet_tags=["wellbore"])
# trial (u, p) — **must** give float‐valued BCs!
u_trial = TrialFESpace(δu, x -> VectorValue(0.0,0.0))
p_trial = TrialFESpace(δp, x -> Pb)

# multi‐field
Y     = MultiFieldFESpace([δu, δp])
u_t   = TransientTrialFESpace(u_trial)
p_t   = TransientTrialFESpace(p_trial)
X_t   = MultiFieldFESpace([u_t, p_t])

# 5) INITIAL CONDITION — also floats
u0    = VectorValue(0.0,0.0)
uh0   = interpolate_everywhere([u0, p0], X_t(0.0))

# 6) CONSTITUTIVE LAW
function σ(u)
  ε = symmetric_gradient(u)
  I = TensorValue(1.0,0.0,0.0,1.0)
  λ*tr(ε)*I + 2*μ*ε
end

# 7) WEAK FORM
a(t, up, tup) = begin
  (u,p)   = up
  (δu,δp) = tup
  ∫( symmetric_gradient(δu) ⊙ σ(u)
    - B*divergence(δu)*p
    + δp*(1/M)*∂t(p)
    + ∇(δp)⋅(kμ*∇(p))
    + δp*B*divergence(∂t(u))
  )*dΩ
end

# zero‐Neumann on u
const zero_vec = VectorValue(0.0,0.0)
l(t, tup) = begin
  (δu,δp) = tup
  ∫( δu ⋅ zero_vec )*dΓb
end

res(t, up, tup) = a(t,up,tup) - l(t,tup)

# 8) SOLVER SETUP
op         = TransientFEOperator(res, X_t, Y)
ls         = LUSolver()
nls        = NLSolver(ls, method=:newton, iterations=10, show_trace=false)
θ          = 1.0
ode_solver = ThetaMethod(nls, dt, θ)

# 9) TIME MARCHING
sol = solve(ode_solver, op, 0.0, T, uh0)

# 10) OUTPUT
createpvd("results/results") do pvd
  disp0,pres0 = uh0
  pvd[0.0] = createvtk(Ω, "results/results_0.vtu",
                       cellfields=["displacement"=>disp0,
                                   "pressure"   =>pres0])
  for (tn, uhn) in sol
    println("Writing t = $tn")
    disp,pres = uhn
    pvd[tn] = createvtk(Ω, "results/results_$(tn).vtu",
                        cellfields=["displacement"=>disp,
                                    "pressure"   =>pres])
  end
end

println("Done.")
