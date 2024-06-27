using OptimalControl
# Define the parameters of the problem
ε = 0.1
Γ = 9.855e-2
γ = 3.65e-3
initial_guess = (state = [-3.623620862590374e-18, 2.7737677971034283e-17, -2.033108387305208e-17, -2.78307884576971e-17], control = 0.12016227281172927, variable = 44.23087063246322)
@def ocp begin
    tf ∈ R, variable
    t ∈ [ 0, tf ], time
    x ∈ R⁴, state
    u ∈ R, control
    tf ≥ 0
    -1 ≤ u(t) ≤ 1
    x(0) == [0, 1 , 0, 1 ]
    x(tf) == [0, 0, 0, 0]
    ẋ(t) == [ (-Γ*x₁(t) -u(t)*x₂(t)), (γ*(1-x₂(t)) +u(t)*x₁(t)) , (-Γ*x₃(t) -(1-ε)* u(t)*x₄(t)), (γ*(1-x₄(t)) +(1-ε)*u(t)*x₃(t))]
    tf → min
end
direct_sol_iter = solve(ocp, print_level=0, grid_size= 200, init=initial_guess)


