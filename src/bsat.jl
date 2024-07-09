using OptimalControl
using Plots

# Define the parameters of the problem
Γ = 9.855e-2  
γ = 3.65e-3   
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
ϵ₀ = 0.1
@def ocpu begin
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
              (-Γ*x₃(t) -(1-ϵ₀)* u(t)*x₄(t)),
              (γ*(1-x₄(t)) +(1-ϵ₀)*u(t)*x₃(t))]
    tf → min
end
ocpf = f(0.1)
initial_g = solve(ocpu, grid_size=100)

for i in 1:10
    global initial_g
    solf = solve(ocpf, grid_size=i*100, init=initial_g)
    initial_g = solf
end

# Plot the figures
plot_sol(initial_g)
# Conclusion: This solution seems to be the optimal one.