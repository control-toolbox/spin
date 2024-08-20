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


@def ocp1 begin
    s ∈ [0, 1], time
    y = (x1, x2, x3, x4, tf) ∈ R⁵, state
    u ∈ R, control
    tf(s) ≥ 0
    -1 ≤ u(s) ≤ 1
    x1(0) == 0
    x2(0) == 1
    x3(0) == 0
    x4(0) == 1
    x1(1) == 0
    x2(1) == 0
    x3(1) == 0
    x4(1) == 0

    ẏ(s) == tf(s) * [(-Γ * x1(s) - u(s) * x2(s)),
                    (γ * (1 - x2(s)) + u(s) * x1(s)),
                    (-Γ * x3(s) -(1 - ϵ) * u(s) * x4(s)),
                    (γ * (1 - x4(s)) +(1 - ϵ) * u(s) * x3(s)),
                    0 ]
    tf(1) → min
end

@def ocp2 begin
    s ∈ [0, 1], time
    y = (x1, x2, x3, x4, tf) ∈ R⁵, state
    u ∈ R, control
    tf(s) ≥ 0
    -1 ≤ u(s) ≤ 1
    x1(0) == 0.1
    x2(0) == 0.9
    x3(0) == 0.1
    x4(0) == 0.9
    x1(1) == 0
    x2(1) == 0
    x3(1) == 0
    x4(1) == 0

    ẏ(s) == tf(s) * [(-Γ * x1(s) -u(s) * x2(s)),
             (γ * (1 - x2(s)) + u(s) * x1(s)),
             (-Γ * x3(s) -(1 - ϵ) * u(s) * x4(s)),
             (γ * (1 - x4(s)) +(1 - ϵ) * u(s) * x3(s)),
             0 ]
    tf(1) → min
end


initial_g = solve(ocp2; grid_size=1000, linear_solver="mumps")
solution = solve(ocp1; grid_size=1000, init=initial_g, linear_solver="mumps")


# Function to plot the solution of the optimal control problem
function plot_sol(sol)
    q = sol.state
    liste = [q(t) for t in sol.times]
    liste_y1 = [elt[1] for elt in liste]
    liste_z1 = [elt[2] for elt in liste]
    liste_y2 = [elt[3] for elt in liste]
    liste_z2 = [elt[4] for elt in liste]
    liste_tf = [elt[5] for elt in liste]
    plot(
        plot(liste_y1, liste_z1, xlabel="y1", ylabel="z1"),
        plot(liste_y2, liste_z2, xlabel="y2", ylabel="z2"),
        plot(sol.times, sol.control, xlabel="Time", ylabel="Control"),
        plot(sol.times, liste_tf, xlabel="Time", ylabel="tf")
        )
end

plot_sol(solution)