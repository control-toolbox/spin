using OptimalControl
using NLPModelsIpopt
using Plots

include("bsat.jl")

ϵ = 0.1
q0 = [0.0, 1.0, 0.0, 1.0]

function F0i(q)
    y, z = q
    res = [-Γ*y, γ*(1-z)]
    return res
end

function F1i(q)
    y, z = q
    res = [-z, y]
    return res
end

F0(q) = [ F0i(q[1:2]); F0i(q[3:4]) ]
F1(q) = [ F1i(q[1:2]); (1 - ϵ) * F1i(q[3:4]) ]
function ocp2(q₁₀, q₂₀)
    @def o begin
        tf ∈ R, variable
        t ∈ [0, tf], time
        q = (y₁, z₁, y₂, z₂) ∈ R⁴, state
        u ∈ R, control
        tf ≥ 0
        -1 ≤ u(t) ≤ 1
        qᵢ₁ = [y₁, z₁]
        qᵢ₂ = [y₂, z₂]
        qᵢ₁(0) == q₁₀
        qᵢ₂(0) == q₂₀
        qᵢ₁(tf) == [0, 0]
        qᵢ₂(tf) == [0, 0]
        q̇(t) == F0(q(t)) + u(t) * F1(q(t))
        tf → min
    end
    return o
end
prob = ocp2([0,1], [0,1])

H0 = Lift(F0) 
H1 = Lift(F1)
H01  = @Lie { H0, H1 }
H001 = @Lie { H0, H01 }
H101 = @Lie { H1, H01 }
us(q, p) = -H001(q, p) / H101(q, p)

t = initial_g.times
q = initial_g.state
u = initial_g.control
p = initial_g.costate
φ(t) = H1(q(t), p(t))
umax = 1
H1_plot = plot(t, φ,     label = "H₁(x(t), p(t))")

function shoot!(s, p0, t1, t2, t3, tf)
    fₚ = Flow(prob, (q, p, tf) -> umax)
    fₘ = Flow(prob, (q, p, tf) -> - umax)
    fs = Flow(prob, (q, p, tf) -> us(q, p))
    q1, p1 = fₘ(0, q0, p0, t1)
    q2, p2 = fs(t1, q1, p1, t2)
    q3, p3 = fₚ(t2, q2, p2, t3)
    qf, pf = fs(t3, q3, p3, tf)
    s[1] = H0(q0, p0) - umax * H1(q0, p0) - 1  
    s[2] = H1(q1, p1)
    s[3] = H01(q1, p1)
    s[4] = H1(q3, p3)
    s[5] = H01(q3, p3)
    s[6] = qf[3]
    s[7] = qf[4]
    s[8] = (pf[2] + pf[4]) * γ - 1
end
t0 = 0
tol = 0.02
t13 = [ elt for elt in t if abs(u(elt)) < tol]
i = 1
t_l = []
while(true)
    if(t13[i+1] - t13[i] > 1)
        break
    else 
        push!(t_l, t13[i])
        push!(t_l, t13[i+1])
        i += 1
    end
end
t1 = min(t_l...)
t2 = max(t_l...)
t3f = [elt for elt in t13 if elt > t2+0.1]
t3 = min(t3f...)
p0 =p(t0) + 0.1 *[1, 1, 1, 1] 
tf = initial_g.variable
using DifferentialEquations
println("p0 = ", p0)
println("t1 = ", t1)
println("t2 = ", t2)
println("t3 = ", t3)
println("tf = ", tf)
using LinearAlgebra: norm
using MINPACK
s = similar(p0, 8)
shoot!(s, p0, t1, t2, t3, tf)
println("Norm of the shooting function: ‖s‖ = ", norm(s), "\n")
nle = (s, ξ) -> shoot!(s, ξ[1:4], ξ[5], ξ[6], ξ[7], ξ[8])   # auxiliary function
                                                               # with aggregated inputs
ξ = [ p0 ; t1 ; t2 ; t3 ; tf ]                                 # initial guess
indirect_sol = fsolve(nle, ξ)
