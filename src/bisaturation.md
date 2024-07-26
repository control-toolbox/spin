# Saturation of pair of spins

The problem we are trying to solve is the time minimal saturation of a pair of spin-$1/2$ particles (or bi-saturation problem), as described in [^1]. This model describes a pair of spins that share the same characteristics, specifically the same relaxation times $T_1$ and $T_2$. However, the control field intensity differs for each spin due to variations as they transition from the North Pole $N := (0,1)$ to the origin $O:=(0,0)$:

```math
\begin{cases}
t_f \rightarrow \min \\
\dot{q}(t) = F(q(t)) + u(t) G(q(t)), \quad |u(t)| \leq 1, \quad t \in [0, t_f], \\
q(0) = q_0, \\
q(t_f) = q_f
\end{cases}
```

where $q_0=[0,1,0,1]$, $q_f=[0,0,0,0]$ and $F$ and $G$ are defined by (2) on page 5, as well as in sections 2.1 and 3.1.

We first define the problem.

```@example main
using OptimalControl
using NLPModelsIpopt
using Plots

Γ = 9.855e-2
γ = 3.65e-3
ϵ = 0.1

function F0(q)
    y, z = q
    res = [-Γ*y, γ*(1-z)]
    return res
end

# idem for F1
function F1(q)
    y, z = q
    res = [-z, y]
    return res
end

F0(q₁, q₂) = [ F0(q₁); F0(q₂) ]
F1(q₁, q₂, ε) = [ F1(q₁); (1 - ε) * F1(q₂) ]

# Define the optimal control problem with initial state q₀ in the case of one spin
function ocp1(q₀)
    @def o begin
        tf ∈ R, variable
        t ∈ [0, tf], time
        q = (y, z) ∈ R², state
        u ∈ R, control
        tf ≥ 0
        -1 ≤ u(t) ≤ 1
        q(0) == q₀
        q(tf) == [0, 0]
        q̇(t) == F0(q(t)) + u(t) * F1(q(t))
        tf → min
    end
    return o
end

# Define the optimal control problem with initial state q₀ in the case of two spins
function ocp2(q₁₀, q₂₀, ε)
    @def o begin
        tf ∈ R, variable
        t ∈ [0, tf], time
        q = (y₁, z₁, y₂, z₂) ∈ R⁴, state
        u ∈ R, control
        tf ≥ 0
        -1 ≤ u(t) ≤ 1
        qᵢ₁ = [y₁, z₁]
        qᵢ₂ = [y₂, z₂]
        qᵢ₁(0) == q₁₀
        qᵢ₂(0) == q₂₀
        qᵢ₁(tf) == [0, 0]
        qᵢ₂(tf) == [0, 0]
        q̇(t) == F0(qᵢ₁(t), qᵢ₂(t)) + u(t) * F1(qᵢ₁(t), qᵢ₂(t), ε)
        tf → min
    end
    return o
end
prob = ocp2([0,1], [0,1], ϵ)
```
However, we quickly realize that solving this problem without any prior initial guesses is not feasible. This realization prompts us to explore potential solutions that facilitate problem-solving.

One effective approach involves homotopy on the initial condition. This method begins from an initial point where the problem can be resolved without requiring any initial guesses. Subsequently, we generate a sequence of initial guesses by solving intermediate problems created through homotopy, gradually progressing towards our original initial condition: $[0, 1, 0, 1]$. Using $[1, 0, 1, 0]$ as our initial guess revealed that it can serve as the starting point for this homotopy process.

## Homotopy on the initial condition
The code below demonstrates how this approach systematically generates initial guesses using homotopy starting from $[1, 0, 1, 0]$, advancing towards the desired initial condition of $[0, 1, 0, 1]$.



```@example main

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

```@example main
q₀₁ = [1, 0]
ocp_x = ocp2(q₀₁, q₀₁, ϵ)
sol_x = solve(ocp_x, grid_size=100)
sol_x.variable
L_x = [sol_x]
for i in 1:10
    x₀ = i / 10 * [0, 1, 0, 1] + (1 - i / 10) * [1, 0, 1, 0]
    ocpi_x = ocp2(x₀[1:2], x₀[3:4], ϵ)
    sol_i_x = solve(ocpi_x, grid_size=100, display=false, init=L_x[end]) 
    push!(L_x, sol_i_x)
end
nothing # hide
```

and plot the solutions.

```@example main
solution_x = L_x[end]
solution_x.variable
plot_sol(solution_x)
```

Conclusion: The solution is considered local, as the final time exceeds the one found using the Bocop software as mentioned in [^1].
Let us now solve this problem differently. One potential initial guess could be obtained by solving a monosaturation problem where the control field intensity is the same for both spins, *i.e.*, $ϵ = 0$, and then using homotopy to transition to $ϵ = 0.1$. We start by solving the problem for a single spin and then extend this approach to two identical spins before applying homotopy.

## Homotopy on ϵ

### Monosaturation problem

```@example main
q₀ = [0, 1]
ocp = ocp1(q₀)
N = 100
sol = solve(ocp, grid_size=N) 
```

Now we extract the control, final time and then duplicate the state to be able to have an initial guess for the two spins.

```@example main
u_init = sol.control  
q_init(t) = [sol.state(t); sol.state(t)]
tf_init = sol.variable
```

### Bi-saturation problem

```@example main
ocp_0 = ocp2(q₀, q₀, 0)
init = (state=q_init, control=u_init, variable=tf_init)
sol2 = solve(ocp_0 ; grid_size=N, init=init)
```

We define the function below that computes the problem depending on $\varepsilon$: 

```@example main

ϵ₁ = 0
initial_guess = sol2
L = [sol2]
for i in 1:10
    global ϵ₁ = ϵ₁ + 0.01
    ocpi = ocp2(q₀, q₀, ϵ₁)
    sol_i = solve(ocpi, grid_size=100, display=false, init=initial_guess)
    global L
    push!(L, sol_i)
    global initial_guess = sol_i
end
sol_eps = L[end]
sol_eps.variable
plot_sol(sol_eps)
```

Conclusion: The solution is a local one.

Another approach involves defining a bi-saturation problem with a slightly adjusted initial condition: $q₀ = [0.1, 0.9, 0.1, 0.9]$.

##  Bi-Saturation Problem: initial Guess from a Slightly Different Problem
```@example main
ϵ = 0.1
q₁₉ = [0.1, 0.9]
ocpu = ocp2(q₁₉, q₁₉, ϵ)
initial_g = solve(ocpu, grid_size=100)
ocpf = prob
for i in 1:10
    global initial_g
    solf = solve(ocpf, grid_size=i*100, init=initial_g)
    initial_g = solf
end
# Plot the figures
plot_sol(initial_g)
```

Conclusion: This solution is better that the two previous, which proves that these were strict local minimisers.

## References
[^1]: Bernard Bonnard, Olivier Cots, Jérémy Rouot, Thibaut Verron. Time minimal saturation of a pair of spins and application in magnetic resonance imaging. Mathematical Control and Related Fields, 2020, 10 (1), pp.47-88.