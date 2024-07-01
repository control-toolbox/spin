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
    q̇(t) == [(-Γ * y(t) - u(t) * z(t)),( γ * (1 - z(t)) + u(t) * y(t))]
    tf → min
end

N = 100
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
    q̇(t) == [-Γ * y₁(t) - u(t) * z₁(t)
              γ * (1 - z₁(t)) + u(t) * y₁(t)
             -Γ * y₂(t) - u(t) * z₂(t)
              γ * (1 - z₂(t)) + u(t) * y₂(t)]
    tf → min
end

init = (state=q_init, control=u_init, variable=tf_init) 
sol2 = solve(ocp2; grid_size=N, init=init)
ϵ = 0
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
initial_guess = sol2
sol2.variable 
L = [sol2]
for i in 1:10
    global ϵ = ϵ + 0.01
    global initial_guess
    ocpi = f(ϵ)
    sol_i = solve(ocpi,  grid_size=100, init=initial_guess)
    global L 
    push!(L, sol_i)
    initial_guess = sol_i

end
ϵ₀ = 0.1
@def ocpu begin
    tf ∈ R, variable
    t ∈ [ 0, tf ], time
    x ∈ R⁴, state
    u ∈ R, control
    tf ≥ 0
    -1 ≤ u(t) ≤ 1
    x(0) == [0.1, 0.9, 0.1, 0.9 ]
    x(tf) == [0, 0, 0, 0]
    ẋ(t) == [ (-Γ*x₁(t) -u(t)*x₂(t)), (γ*(1-x₂(t)) +u(t)*x₁(t)), (-Γ*x₃(t) -(1-ϵ₀)* u(t)*x₄(t)), (γ*(1-x₄(t)) +(1-ϵ₀)*u(t)*x₃(t))]
    tf → min
end

ocpf = f(0.1)
initial_g = solve(ocpu, grid_size=100)

for i in 1:10
    global initial_g 
    solf = solve(ocpf,  grid_size= i*100, init=initial_g)
    initial_g = solf
end 
#Figures:
using Plots
q = initial_g.state
liste = [q(t) for t in initial_g.times]
liste_y1 = [elt[1] for elt in liste ]
liste_z1 = [elt[2] for elt in liste ]
liste_y2 = [elt[3] for elt in liste ]
liste_z2 = [elt[4] for elt in liste ]
plot(liste_y1, liste_z1, xlabel="y1", ylabel="z1")
plot(liste_y2, liste_z2)
plot(initial_g.control, initial_g.times)