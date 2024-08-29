using OrdinaryDiffEq
using LinearAlgebra: norm
using MINPACK
using OptimalControl
using Plots
using NLPModelsIpopt

# Define the parameters of the problem
Γ = 9.855e-2  
γ = 3.65e-3  
ϵ = 0.1
q0 = [0, 1, 0, 1]
# Define the optimal control problem with parameter ϵ₀

@def ocp1 begin
    tf ∈ R, variable 
    t ∈ [0, tf], time 
    x ∈ R⁴, state    
    u ∈ R, control    
    tf ≥ 0            
    -1 ≤ u(t) ≤ 1     
    x(0) == [0, 1, 0, 1]
    x(tf) == [0, 0, 0, 0] 
    
    ẋ(t) == [ (-Γ*x₁(t) -u(t)*x₂(t)), 
                (γ*(1-x₂(t)) +u(t)*x₁(t)), 
                (-Γ*x₃(t) -(1-ϵ)* u(t)*x₄(t)), 
                (γ*(1-x₄(t)) +(1-ϵ)*u(t)*x₃(t))]
    tf → min 
end

@def ocp2 begin
    tf ∈ R, variable 
    t ∈ [0, tf], time 
    x ∈ R⁴, state    
    u ∈ R, control    
    tf ≥ 0            
    -1 ≤ u(t) ≤ 1     
    x(0) == [0.1, 0.9, 0.1, 0.9]
    x(tf) == [0, 0, 0, 0] 
    
    ẋ(t) == [ (-Γ*x₁(t) -u(t)*x₂(t)), 
                (γ*(1-x₂(t)) +u(t)*x₁(t)), 
                (-Γ*x₃(t) -(1-ϵ)* u(t)*x₄(t)), 
                (γ*(1-x₄(t)) +(1-ϵ)*u(t)*x₃(t))]
    tf → min 
end
# Function to plot the solution of the optimal control problem
function plot_sol(sol)
    q = sol.state
    liste = [q(t) for t in sol.times]
    liste_y1 = [elt[1] for elt in liste]
    liste_z1 = [elt[2] for elt in liste]
    liste_y2 = [elt[3] for elt in liste]
    liste_z2 = [elt[4] for elt in liste]
    plot(
        plot(liste_y1, liste_z1, xlabel="y1", ylabel="z1"),
        plot(liste_y2, liste_z2, xlabel="y2", ylabel="z2"),
        plot(sol.times, sol.control, xlabel="Time", ylabel="Control")
    )
end

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

initial_g = solve(ocp2; grid_size=1000, linear_solver="mumps")
solution = solve(ocp1; grid_size=1000, init=initial_g, linear_solver="mumps")
#plot(solution)
H0 = Lift(F0) 
H1 = Lift(F1)
H01  = @Lie { H0, H1 }
H001 = @Lie { H0, H01 }
H101 = @Lie { H1, H01 }
us(q, p) = -H001(q, p) / H101(q, p)

t = solution.times
q = solution.state
u = solution.control
p = solution.costate

umax = 1
u0 = 0
fₚ = Flow(ocp1, (q, p, tf) -> umax)
fₘ = Flow(ocp1, (q, p, tf) -> - umax)
fs = Flow(ocp1, (q, p, tf) -> us(q, p))

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
tf = solution.objective
q1, p1 = q(t1), p(t1)
q2, p2 = q(t2), p(t2)
q3, p3 = q(t3), p(t3)

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

s = similar(p0, 32)
shoot!(s, p0, t1, t2, t3, tf, q1, p1, q2, p2, q3, p3)
println("Norm of the shooting function: ‖s‖ = ", norm(s), "\n")
#plot(solution)
nle = (s, ξ) -> shoot!(s, ξ[1:4], ξ[5], ξ[6], ξ[7], ξ[8], ξ[9:12], ξ[13:16], ξ[17:20], ξ[21:24], ξ[25:28], ξ[29:32])   # auxiliary function
                                                               # with aggregated inputs
ξ = [ p0 ; t1 ; t2 ; t3 ; tf ; q1 ; p1 ; q2 ; p2 ; q3 ; p3 ]                                 

indirect_sol = fsolve(nle, ξ; show_trace=true, tol=1e-6)


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
f_sol = fₘ * (t1i, fs) * (t2i, fₚ) * (t3i, fs);
flow_sol = f_sol((t0, tfi), q0, p0i) ;

plt = plot(solution, solution_label="(direct)")
plot(plt, flow_sol, solution_label="(indirect)")
