using OptimalControl
using NLPModelsIpopt
using Plots

include("bsat.jl")

# Define the parameters of the problem
Γ = 9.855e-2  
γ = 3.65e-3  
ϵ =0.1
function F0i(q)
    y, z = q
    res = [-Γ*y, γ*(1-z)]
    return res
end

# idem for F1
function F1i(q)
    y, z = q
    res = [-z, y]
    return res
end

F0(q) = [ F0i(q[1:2]); F0i(q[3:4]) ]
F1(q) = [ F1i(q[1:2]); (1 - ϵ) * F1i(q[3:4]) ]
@def ocp_t begin
    tf ∈ R, variable
    t ∈ [0, tf], time
    q = (y₁, z₁, y₂, z₂) ∈ R⁴, state
    u ∈ R, control
    tf ≥ 0
    -1 ≤ u(t) ≤ 1
    q(0) == [0, 1, 0, 1]
    q(tf) == [0, 0, 0, 0]
    q̇(t) == F0(q(t)) + u(t) * F1(q(t))
    tf → min
end

H0 = Lift(F0) 
H1 = Lift(F1)
H01  = @Lie { H0, H1 }
H001 = @Lie { H0, H01 }
H101 = @Lie { H1, H01 }
us(x, p) = -H001(x, p) / H101(x, p)
t = initial_g.times
q = initial_g.state
u = initial_g.control
p = initial_g.costate
φ(t) = H1(q(t), p(t))
uₘₐₓ = 1
H1_plot = plot(t, φ,     label = "H₁(x(t), p(t))")

function shoot!(s, p0, tf , t1, t2, t3)
    q0 = [0, 1, 0, 1]
    fₚ = Flow(ocp_t, (q, p, tf) -> uₘₐₓ)
    fₘ = Flow(ocp_t, (q, p, tf) -> - uₘₐₓ)
    fs = Flow(ocp_t, (q, p, tf) -> us(q, p))
    q1, p1 = fₘ(0, q0, p0, t1)
    q2, p2 = fs(t1, q1, p1, t2)
    p3, q3 = fₚ(t2, q2, p2, t3)
    qf, pf = fs(t3, q3, p3, tf)
    s[1] = H0(q0, p0) - uₘₐₓ * H1(q0, p0) - 1  
    s[2] = H1(q1, p1)
    s[3] = H01(q1, p1)
    s[4] = H1(q3, p3)
    s[5] = H01(q3, p3)
    s[6] = qf[3]
    s[7] = qf[4]
    s[8] = (pf[2] + pf[4]) * γ - 1
end
t0 = 0
tol = 0.01
t13 = [ elt for elt in t if abs(φ(elt)) < tol]
i = 1
t_l = []
while(true)
    if(t13[i+1] - t13[i] > 0.06)
        break
    else 
        push!(t_l, t13[i])
        i += 1
    end
end
t_l 

t1 = min(t_l...)
t2 = max(t_l...)
t3f = [elt for elt in t13 if elt > t2+0.1]
t3 = min(t3f...)
p0 = p(t0)
q0 = [0, 1, 0, 1]
tf = initial_g.variable
using DifferentialEquations
f1 = Flow(ocp_t, (q, p, tf) -> -uₘ)
qi1, pi1 = f1(0, q0, p0, t1)
println("p0 = ", p0)
println("t1 = ", t1)
println("t2 = ", t2)
println("t3 = ", t3)
println("tf = ", tf)
using LinearAlgebra: norm
s = similar(p0, 8)
shoot!(s, p0, tf, t1, t2, t3)
nle = (s, ξ, λ) -> shoot!(s, ξ[1:4], ξ[5], ξ[6], ξ[7], ξ[8])   # auxiliary function
                                                               # with aggregated inputs
ξ = [ p0 ; tf ; t1 ; t2 ; t3 ]                                 # initial guess

prob = NonlinearProblem(nle, ξ)
indirect_sol = NonlinearSolve.solve(prob) 