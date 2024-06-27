# We first start by solving a mono saturation problem, (ie ϵ = 0 ).
# That means that the state q is in 2 dimensions.

using OptimalControl

# Define the parameters of the problem
Γ = 9.855e-2
γ = 3.65e-3

@def ocp1 begin
    tf ∈ R, variable
    t ∈ [0, tf], time
    q = (y, z) ∈ R², state
    u ∈ R, control
    tf ≥ 0
    -1 ≤ u(t) ≤ 1
    q(0) == [0, 1] # North pole
    q(tf) == [0, 0] # Center
    q̇(t) == [-Γ * y(t) - u(t) * z(t), γ * (1 - z(t)) + u(t) * y(t)]
    tf → min
end

N = 200
sol1 = solve(ocp1, grid_size=N)
u_init = sol1.control
q_init(t) = [sol1.state(t); sol1.state(t)]
tf_init = sol1.variable

@def ocp2 begin
    tf ∈ R, variable
    t ∈ [0, tf], time
    q = (y₁, z₁, y₂, z₂) ∈ R⁴, state
    u ∈ R, control
    tf ≥ 0
    -1 ≤ u(t) ≤ 1
    q(0) == [0, 1, 0, 1] # North pole for both spins
    q(tf) == [0, 0, 0, 0] # Center for both
    q̇(t) == [-Γ * y₁(t) - u(t) * z₂(t)
              γ * (1 - z₁(t)) + u(t) * y₁(t)
             -Γ * y₂(t) - u(t) * z₂(t)
              γ * (1 - z₂(t)) + u(t) * y₂(t)]
    tf → min
end

init = (state=q_init, control=u_init, variable=tf_init) 
sol2 = solve(ocp2; grid_size=N, init=init)