# Saturation pair of spins and MRI
The problem we are trying to solve is the  $P_{BS}$ problem, also known as the *time minimal saturation problem of a pair of spin-1/2 particles* (or bi-saturation problem), as described on page 18 of the paper [^1] . This model describes a pair of spins that share the same characteristics, specifically the same relaxation times $T_1$ and $T_2$. However, the control field intensity differs for each spin due to variations as they transition from the North Pole $N := (0,1)$ to the origin $O:=(0,0)$.

```math
\begin{cases}
J(u(\cdot), t_f) := t_f \rightarrow \min \\
\dot{q}(t) = F(q(t)) + u(t) G(q(t)), \quad |u(t)| \leq 1, \quad t \in [0, t_f], \\
q(0) = q_0, \\
q(t_f) = q_f
\end{cases}
```

where $q_0=[0,1,0,1]$, $q_f=[0,0,0,0]$ and $F$ and $G$ are defined by equation 2 on page 5, as well as in sections 2.1 and 3.1. We use the control-toolbox functions to find both local and global solutions.

We first define the problem.

```@example main
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
    ẋ(t) == [ (-Γ*x₁(t) -u(t)*x₂(t)), (γ*(1-x₂(t)) +u(t)*x₁(t)), (-Γ*x₃(t) -(1-ϵ₀)* u(t)*x₄(t)), (γ*(1-x₄(t)) +(1-ϵ₀)*u(t)*x₃(t))]
    tf → min
end
```
However, we quickly realize that solving this problem without any prior initial guesses is not feasible. This realization prompts us to explore potential solutions that facilitate problem-solving.

One effective approach involves homotopy on the initial condition. This method begins from an initial point where the problem can be resolved without requiring any initial guesses. Subsequently, we generate a sequence of initial guesses by solving intermediate problems created through homotopy, gradually progressing towards our original initial condition : $[0, 1, 0, 1]$. Using $[1, 0, 1, 0]$ as our initial guess revealed that it can serve as the starting point for this homotopy process.

## References
[^1]: Bernard Bonnard, Olivier Cots, Jérémy Rouot, Thibaut Verron. Time minimal saturation of a pair of spins and application in magnetic resonance imaging. Mathematical Control and Related Fields, 2020, 10 (1), pp.47-88.