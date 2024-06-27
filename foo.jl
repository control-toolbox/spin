# foo.jl
using OptimalControl
# Define the parameters of the problem
ε = 0.1
Γ = 9.855e-2
γ = 3.65e-3
# Define the optimal control problem
@def ocp1 begin
    tf ∈ R, variable
    t ∈ [ 0, tf ], time
    x ∈ R⁴, state
    u ∈ R, control
    tf ≥ 0
    -1 ≤ u(t) ≤ 1
    x(0) == [0.9, 0.1, 0.9, 0.1]
    x(tf) == [0, 0, 0, 0]
    ẋ(t) == [ (-Γ*x₁(t) -u(t)*x₂(t)), (γ*(1-x₂(t)) +u(t)*x₁(t)) , (-Γ*x₃(t) -(1-ε)* u(t)*x₄(t)), (γ*(1-x₄(t)) +(1-ε)*u(t)*x₃(t))]
    tf → min
end
initial_guess = (state=[3.437560295296817e-44, -9.80908925027372e-45, 0, 0], control=8.881784197001252e-16, variable=43.95189708821126)
direct_sol_iter = solve(ocp1, init=initial_guess)
