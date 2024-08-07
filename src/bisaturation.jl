using OptimalControl
using NLPModelsIpopt
using Plots

# Define the parameters of the problem
Γ = 9.855e-2  
γ = 3.65e-3   
ϵ₁ = 0.1
# Define the optimal control problem with parameter ϵ₀
function f(ϵ₀)
    @def ocp begin
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
                   (-Γ*x₃(t) -(1-ϵ₀)* u(t)*x₄(t)), 
                   (γ*(1-x₄(t)) +(1-ϵ₀)*u(t)*x₃(t))]
        tf → min 
    end
    return ocp
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

# Define the optimal control problem with initial state x₀
function g(x₀)
    @def ocp begin
        tf ∈ R, variable
        t ∈ [0, tf], time
        x ∈ R⁴, state
        u ∈ R, control
        tf ≥ 0
        -1 ≤ u(t) ≤ 1
        x(0) == x₀
        x(tf) == [0, 0, 0, 0]
        ẋ(t) == [ (-Γ*x₁(t) -u(t)*x₂(t)), 
                   (γ*(1-x₂(t)) +u(t)*x₁(t)), 
                   (-Γ*x₃(t) -(1-ϵ₁)* u(t)*x₄(t)), 
                   (γ*(1-x₄(t)) +(1-ϵ₁)*u(t)*x₃(t))]
        tf → min
    end
    return ocp
end

#################################################
# Bi-Saturation Problem: Homotopy on the Initial Conditions
#################################################
# To solve the problem with the initial condition [0,1,0,1], 
# we use the initial condition [1,0,1,0] as a starting point for homotopy,
# generating a sequence of initial guesses that gradually approach our target problem.


ocp_x = g([1, 0, 1, 0])
sol_x = solve(ocp_x, grid_size=100, linear_solver="mumps")
sol_x.variable
L_x = [sol_x]
for i in 1:10
    x₀ = i / 10 * [0, 1, 0, 1] + (1 - i / 10) * [1, 0, 1, 0]
    ocpi_x = g(x₀)
    sol_i_x = solve(ocpi_x; grid_size=100, init=L_x[end], linear_solver="mumps") 
    push!(L_x, sol_i_x)
end
solution_x = L_x[end]
solution_x.variable
plot_sol(solution_x)
# Conclusion: The solution is a local one.

#################################################
# Monosaturation Problem: One Spin
#################################################
@def ocp1 begin
    tf ∈ R, variable
    t ∈ [0, tf], time
    q = (y, z) ∈ R², state
    u ∈ R, control
    tf ≥ 0
    -1 ≤ u(t) ≤ 1
    q(0) == [0, 1]   
    q(tf) == [0, 0] 
    q̇(t) == [(-Γ * y(t) - u(t) * z(t)), 
              (γ * (1 - z(t)) + u(t) * y(t))]
    tf → min
end

N = 100
sol1 = solve(ocp1, grid_size=N, linear_solver="mumps")
u_init = sol1.control
q_init(t) = [sol1.state(t); sol1.state(t)]
tf_init = sol1.variable

#################################################
# Bi-Saturation Problem: Control Field with Same Intensities on Both Spins
#################################################
@def ocp2 begin
    tf ∈ R, variable
    t ∈ [0, tf], time
    q = (y₁, z₁, y₂, z₂) ∈ R⁴, state
    u ∈ R, control
    tf ≥ 0
    -1 ≤ u(t) ≤ 1
    q(0) == [0, 1, 0, 1] 
    q(tf) == [0, 0, 0, 0] 
    q̇(t) == [-Γ * y₁(t) - u(t) * z₁(t),
              γ * (1 - z₁(t)) + u(t) * y₁(t),
             -Γ * y₂(t) - u(t) * z₂(t),
              γ * (1 - z₂(t)) + u(t) * y₂(t)]
    tf → min
end

init = (state=q_init, control=u_init, variable=tf_init)
sol2 = solve(ocp2; grid_size=N, init=init, linear_solver="mumps")

# Homotopy on ϵ from ϵ = 0 to ϵ = 0.1 to find a potential solution
ϵ = 0
initial_guess = sol2
L = [sol2]
for i in 1:10
    global ϵ = ϵ + 0.01
    global initial_guess
    ocpi = f(ϵ)
    sol_i = solve(ocpi; grid_size=100, init=initial_guess, linear_solver="mumps")
    global L
    push!(L, sol_i)
    initial_guess = sol_i
end
sol_eps = L[end]
sol_eps.variable
plot_sol(sol_eps)
# Conclusion: The solution is a local one. In the results found in the paper, the final time is more optimal.

#################################################
# Bi-Saturation Problem: Initial Guess from a Slightly Different Problem
#################################################
# Use the solution of the problem with x₀ = [0.1, 0.9, 0.1, 0.9] as an initial guess for the bi-saturation problem
@def ocp_h begin
    tf ∈ R, variable
    t ∈ [0, tf], time
    x ∈ R⁴, state
    u ∈ R, control
    tf ≥ 0
    -1 ≤ u(t) ≤ 1
    x(0) == [0.1, 0.9, 0.1, 0.9] # Initial condition
    x(tf) == [0, 0, 0, 0] # Terminal condition
    ẋ(t) == [(-Γ*x₁(t) -u(t)*x₂(t)),
              (γ*(1-x₂(t)) +u(t)*x₁(t)),
              (-Γ*x₃(t) -(1-ϵ₁)* u(t)*x₄(t)),
              (γ*(1-x₄(t)) +(1-ϵ₁)*u(t)*x₃(t))]
    tf → min
end

prob = f(ϵ₁)

initial_g = solve(ocp_h; grid_size=1000, linear_solver="mumps")
direct_sol = solve(prob; grid_size=1000, init=initial_g, linear_solver="mumps")
# Plot the figures
plot_sol(direct_sol)
# Conclusion: This solution seems to be the optimal one.
