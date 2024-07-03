# Saturation pair of spins and MRI
The problem we are trying to solve is the  \(P_{BS}\) problem, also known as the *time minimal saturation problem of a pair of spin-1/2 particles* (or bi-saturation problem), as described on page 18 of the paper [^1] . This model describes a pair of spins that share the same characteristics, specifically the same relaxation times $T_1$ and $T_2$. However, the control field intensity differs for each spin due to variations as they transition from the North Pole $N := (0,1)$ to the origin $O:=(0,0)$. wahiya

```
\begin{cases}
J(u(\cdot), t_f) := t_f \rightarrow \min \\
\dot{q}(t) = F(q(t)) + u(t) G(q(t)), \quad |u(t)| \leq 1, \quad t \in [0, t_f], \\
q(0) = q_0, \\
q(t_f) = q_f
\end{cases}
```
where $q_0=[0,1,0,1]$, $q_f=[0,0,0,0]$ and $F$ and $G$ are defined by equation 2 on page 5, as well as in sections 2.1 and 3.1. We use the control-toolbox functions to find both local and global solutions.

We first define the problem.

```julia
using OptimalControl
Γ = 9.855e-2
γ = 3.65e-3
ϵ₀ = 0.1
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
```
However, we quickly realize that solving this problem without any prior initial guesses is not feasible. This realization prompts us to explore potential solutions that facilitate problem-solving.

One effective approach involves homotopy on the initial condition. This method begins from an initial point where the problem can be resolved without requiring any initial guesses. Subsequently, we generate a sequence of initial guesses by solving intermediate problems created through homotopy, gradually progressing towards our original initial condition : $[0, 1, 0, 1]$. Using $[1, 0, 1, 0]$ as our initial guess revealed that it can serve as the starting point for this homotopy process.

## Homotopy on the initial condition:
The code below demonstrates how this approach systematically generates initial guesses using homotopy starting from $[1, 0, 1, 0]$, advancing towards the desired initial condition of $[0, 1, 0, 1]$.

Let's first define functions that define the optimal control problem with initial state x₀ and plot the solutions: 
```julia
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
                (-Γ*x₃(t) -(1-ϵ)* u(t)*x₄(t)), 
                (γ*(1-x₄(t)) +(1-ϵ)*u(t)*x₃(t))]
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
```
Then we perform homotopy on the initial condition with a step of 0.1,
```julia
ocp_x = g([1, 0, 1, 0])
sol_x = solve(ocp_x, grid_size=100)
sol_x.variable
L_x = [sol_x]
for i in 1:10
    x₀ = i / 10 * [0, 1, 0, 1] + (1 - i / 10) * [1, 0, 1, 0]
    ocpi_x = g(x₀)
    sol_i_x = solve(ocpi_x, grid_size=100, display=false, init=L_x[end]) 
    push!(L_x, sol_i_x)
end
nothing # hide
```
and plot the solutions.
```julia
solution_x = L_x[end]
solution_x.variable
plot_sol(solution_x)
```
Conclusion: The solution is considered local, as the final time exceeds the one found using the Bocop software as mentioned in [^1].
Now, let's approach solving this problem differently. One potential initial guess could be obtained by solving a monosaturation problem where the control field intensity is the same for both spins, i.e., $ϵ = 0$, and then using homotopy to transition to $ϵ = 0.1$. We start by solving the problem for a single spin and then extend this approach to two identical spins before applying homotopy.
## Homotopy on ϵ : 
### Monosaturation problem :


```julia
@def ocp begin
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
sol = solve(ocp, grid_size=N) 
# Extracting the control, final time and then duplicating the state to be able to have an initial guess for the two spins.
u_init = sol.control  
q_init(t) = [sol.state(t); sol.state(t)]
tf_init = sol.variable
```
### Bi-saturation problem :
```julia
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
sol2 = solve(ocp2; grid_size=N, init=init)
```
Homotopy on ϵ to find a potential solution : 
We define the function below that computes the problem depending on $ϵ$: 
```julia
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
```

```julia
ϵ = 0
initial_guess = sol2
L = [sol2]
for i in 1:10
    global ϵ = ϵ + 0.01
    global initial_guess
    ocpi = f(ϵ)
    sol_i = solve(ocpi, grid_size=100, display=false, init=initial_guess)
    global L
    push!(L, sol_i)
    initial_guess = sol_i
end
sol_eps = L[end]
sol_eps.variable
plot_sol(sol_eps)
```

Conclusion: The solution is a local one.

Another approach involves defining a bi-saturation problem with a slightly adjusted initial condition: $q₀ = [0.1, 0.9, 0.1, 0.9]$.

##  Bi-Saturation Problem: Initial Guess from a Slightly Different Problem
```julia
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
initial_g = solve(ocpu, grid_size=100)
ϵ₀ = 0.1
ocpf = f(ϵ₀)
for i in 1:10
    solf = solve(ocpf, grid_size=i*100, init=initial_g)
    initial_g = solf
end
# Plot the figures
plot_sol(initial_g)
# Conclusion: This solution seems to be the optimal one.
```

## Resources :
[^1]: Bernard Bonnard, Olivier Cots, Jérémy Rouot, Thibaut Verron. Time minimal saturation of a pair of spins and application in magnetic resonance imaging. Mathematical Control and Related Fields, 2020, 10 (1), pp.47-88.


