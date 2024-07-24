using DifferentialEquations
using LinearAlgebra: norm
using MINPACK

include("bsat.jl")


ϵ = 0.1
q0 = [0, 1, 0, 1]

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
#solution_2000 = solve(prob, grid_size=2000, init=solf, linear_solver="mumps")
#plot(solution_2000)
H0 = Lift(F0) 
H1 = Lift(F1)
H01  = @Lie { H0, H1 }
H001 = @Lie { H0, H01 }
H101 = @Lie { H1, H01 }
us(q, p) = -H001(q, p) / H101(q, p)

t = solf.times
q = solf.state
u = solf.control
p = solf.costate
#φ(t) = H1(q(t), p(t))
umax = 1
u0 = 0
#tolerances = (abstol=1e-14, reltol=1e-10)
#H1_plot = plot(t, φ,     label = "H₁(x(t), p(t))")
fₚ = Flow(prob, (q, p, tf) -> umax)
fₘ = Flow(prob, (q, p, tf) -> - umax)
fs = Flow(prob, (q, p, tf) -> us(q, p))

function shoot!(s, p0, t1, t2, t3, tf, q1, p1, q2, p2, q3, p3)
    qi1, pi1 = fₘ(0, q0, p0, t1)
    qi2, pi2 = fs(t1, q1, p1, t2)
    qi3, pi3 = fₚ(t2, q2, p2, t3)
    qf, pf = fs(t3, q3, p3, tf)
    s[1] = H0(q0, p0) - umax * H1(q0, p0) - 1  
    s[2] = H1(q1, p1)
    s[3] = H01(q1, p1)
    s[4] = H1(q3, p3)
    s[5] = H01(q3, p3)
    s[6] = qf[3]
    s[7] = qf[4]
    s[8] = (pf[2] + pf[4]) * γ - 1
    s[9:12] = qi1 - q1
    s[13:16] = pi1 - p1
    s[17:20] = qi2 - q2
    s[21:24] = pi2 - p2
    s[25:28] = qi3 - q3
    s[29:32] = pi3 - p3
end
t0 = 0
tol = 2e-2
t13 = [ elt for elt in t if abs(u(elt)) < tol]
i = 1
t_l = []
while(true)
    global i
    if (( i == length(t13)-1) || (t13[i+1] - t13[i] > 1) )
        break
    else 
        push!(t_l, t13[i])
        push!(t_l, t13[i+1])
        i += 1
    end
end
t1 = min(t_l...)
t2 = max(t_l...)
t3f = [elt for elt in t13 if elt > t2]
t3 = min(t3f...)
p0 = p(t0) 
tf = solf.objective
q1, p1 = q(t1), p(t1)
q2, p2 = q(t2), p(t2)
q3, p3 = q(t3), p(t3)
#p0[1], q0[1], p0[3], q0[3]= -p0[1], -q0[1], -p0[3], -q0[3]
#p1[1], q1[1], p1[3], q1[3]= -p1[1], -q1[1], -p1[3], -q1[3]
#p2[1], q2[1], p2[3], q2[3]= -p2[1], -q2[1], -p2[3], -q2[3]
#p3[1], q3[1], p3[3], q3[3]= -p3[1], -q3[1], -p3[3], -q3[3]
println("p0 = ", p0)
println("t1 = ", t1)
println("t2 = ", t2)
println("t3 = ", t3)
println("tf = ", tf)

δ = γ - Γ
zs = γ/(2*δ)

q1[2] = zs
p1[2] = p1[1] * (zs / q1[1])
q1[4] = zs
p1[4] = p1[3] *(zs / q1[3])
p0[1] = -1
p0[3] = -1
s = similar(p0, 32)
shoot!(s, p0, t1, t2, t3, tf, q1, p1, q2, p2, q3, p3)
println("Norm of the shooting function: ‖s‖ = ", norm(s), "\n")
nle = (s, ξ) -> shoot!(s, ξ[1:4], ξ[5], ξ[6], ξ[7], ξ[8], ξ[9:12], ξ[13:16], ξ[17:20], ξ[21:24], ξ[25:28], ξ[29:32])   # auxiliary function
                                                               # with aggregated inputs
ξ = [ p0 ; t1 ; t2 ; t3 ; tf ; q1 ; p1 ; q2 ; p2 ; q3 ; p3 ]                                 

indirect_sol = fsolve(nle, ξ; show_trace=true)


p0i = indirect_sol.x[1:4]
t1i = indirect_sol.x[5]
t2i = indirect_sol.x[6]
t3i = indirect_sol.x[7]
tfi = indirect_sol.x[8]
q1, p1, q2, p2, q3, p3 = indirect_sol.x[9:12], indirect_sol.x[13:16], indirect_sol.x[17:20], indirect_sol.x[21:24], indirect_sol.x[25:28], indirect_sol.x[29:32]

println("p0 = ", p0i)
println("t1 = ", t1i)
println("t2 = ", t2i)
println("t3 = ", t3i)
println("tf = ", tfi)

# Norm of the shooting function at solution
si = similar(p0i, 32)
shoot!(si, p0i, t1i, t2i, t3i, tfi,q1, p1, q2, p2, q3, p3)
println("Norm of the shooting function: ‖si‖ = ", norm(si), "\n")
f_sol = fₘ * (t1i, fs) * (t2i, fₚ) * (t3i, fs)
flow_sol = f_sol((t0, tfi), q0, p0i) 

plt = plot(solf, solution_label="(direct)")
plot(plt, flow_sol, solution_label="(indirect)")