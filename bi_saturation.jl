#We'll try to change progressively the value of ϵ to see how the solution changes.
#It will follow the same homotopy methhod used on the initial conditions.
using OptimalControl
# Define the parameters of the problem
Γ = 9.855e-2
γ = 3.65e-3
ϵ =0 # The initial value of ϵ
function f(ϵ₀)
    @def ocp begin
        tf ∈ R, variable
        t ∈ [ 0, tf ], time
        x ∈ R⁴, state
        u ∈ R, control
        tf ≥ 0
        -1 ≤ u(t) ≤ 1
        x(0) == [0, 1, 0, 1 ]
        x(tf) == [0, 0, 0, 0]
        ẋ(t) == [ (-Γ*x₁(t) -u(t)*x₂(t)), (γ*(1-x₂(t)) +u(t)*x₁(t)), (-Γ*x₃(t) -(1-ϵ₀)* u(t)*x₄(t)), (γ*(1-x₄(t)) +(1-ϵ₀)*u(t)*x₃(t))]
        tf → min
    end
    return ocp
end
initial_guess = solve(f(ϵ), grid_size=100)
for i in 1:10
    global ϵ = ϵ + 0.01
    global initial_guess
    ocpi = f(ϵ)
    sol_i = solve(ocpi,  grid_size=100, init=initial_guess)
    initial_guess = sol_i
end