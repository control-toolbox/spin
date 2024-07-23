# Saturation of pair of spins : direct and indirect solutions

Previously, we attempted to solve the bi-saturation problem as mentioned in [^1] using a direct method. We will now proceed to solve the same problem using an indirect method, which will require us to use the solution from the direct method as an initial guess. 

## Direct Method : 
Let's first import the necessary packages, *OptimalControl*, *Plots* ... : 

```@example main
using OptimalControl
using NLPModelsIpopt
using Plots
using OrdinaryDiffEq
using LinearAlgebra
using MINPACK
```

We will now define the parameters and the functions that we will use later on : 

```@example main 
# Define the parameters of the problem
Γ = 9.855e-2  
γ = 3.65e-3   
ϵ = 0.1
q0 = [0, 1, 0, 1]

function F0i(q)
    y, z = q
    res = [-Γ*y, γ*(1-z)]
    return res
end

function F1i(q)
    y, z = q
    res = [-z, y]
    return res
end

F0(q) = [ F0i(q[1:2]); F0i(q[3:4]) ]
F1(q) = [ F1i(q[1:2]); (1 - ϵ) * F1i(q[3:4]) ]
function ocp(q₁₀, q₂₀)
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
        q̇(t) == F0(q(t)) + u(t) * F1(q(t))
        tf → min
    end
    return o
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

We will use the same technique used before to solve the problem which involves using the solution of the same problem but with a slight change in the initial conditions, as an initial guess. 

```@example main 
prob = ocp([0, 1], [0, 1])
ocp_h = ocp([0.1, 0.9], [0.1, 0.9])
initial_g = solve(ocp_h, grid_size=100)
```

The provided code performs an iterative process to refine the solution. 

```@example main 
for i ∈ 1:10
    global initial_g
    solf = solve(prob, grid_size=i*100, init=initial_g)
    initial_g = solf
end
direct_sol = initial_g
```

We will now plot the solution : 

```@example main 
plt = plot(direct_sol, solution_label="(direct)", size=(800, 800))
```

## Indirect Method

A quick look on the plot of the control u, reveals that the optimal solution consists of a bang arc with minimal control(-1), followed by a singular arc, then another bang arc with maximal control (+1), and the final arc is a singular arc, which means that **we have a solution with a structure of the form BSBS, i.e. Bang-Singular-Bang-Singular** [^1]. 
First, let's define the Hamiltonian operator.
Since : 
```math
q\dot = F_0(q) + u * F_1(q)
```
then
```math
H(q,p) = p' * F_0(q) + u * p' * F_1(q)
``` 
We'll note $H_0(q,p) = p' * F_0(q) $ and $H_1(q,p) = p' * F_1(q)$
Let $u_{+} = 1$, the positive bang control (resp. $u_{-} = -1$ the negative bang control), 
and ```math 
u_s(q,p) = \frac{H_{001}}{H_{101}} ``` 
the singular control, where : $H_{001} ​= {H_0 ​, {H_0​, H_1​}}, H_{101} ​= {H_1​, {H_0​, H_1​}}$ and for two Hamiltonien operators $H_0​, H_1$ :  
```math
{H_0, H_1}:=({\nabla}_p H_0  ∣ {\nabla}_x H_1 ) − ({\nabla}_x H_0 ∣ {\nabla}_p H_1)
```
First, we refine the solution with a higher grid size for better accuracy. We also lift the vector fields to their Hamiltonian counterparts and compute the Lie brackets of these Hamiltonian vector fields. Additionally, we define the singular control function and extract the solution components.

```@example main 
# Refine the solution with a higher grid size for better accuracy
solution_2000 = solve(prob, grid_size=2000, display=false, init=initial_g)

# Lift the vector fields to their Hamiltonian counterparts
H0 = Lift(F0) 
H1 = Lift(F1)

# Compute the Lie brackets of the Hamiltonian vector fields
H01  = @Lie { H0, H1 }
H001 = @Lie { H0, H01 }
H101 = @Lie { H1, H01 }

# Define the singular control function
us(q, p) = -H001(q, p) / H101(q, p)

# Extract the solution components
t = solution_2000.times
q = solution_2000.state
u = solution_2000.control
p = solution_2000.costate

# Define the flows for maximum, minimum, and singular controls
fₚ = Flow(prob, (q, p, tf) -> umax)
fₘ = Flow(prob, (q, p, tf) -> -umax)
fs = Flow(prob, (q, p, tf) -> us(q, p))
```
Next, we define a function to compute the shooting function for the indirect method. This function calculates the state and costate at the switching times and populates the shooting function residuals.
```@example main 
# Function to compute the shooting function for the indirect method
function shoot!(s, p0, t1, t2, t3, tf, q1, p1, q2, p2, q3, p3)
    qi1, pi1 = fₘ(0, q0, p0, t1)
    qi2, pi2 = fs(t1, q1, p1, t2)
    qi3, pi3 = fₚ(t2, q2, p2, t3)
    qf, pf = fs(t3, q3, p3, tf)
    s[1] = H0(q0, p0) - umax * H1(q0, p0) - 1  
    s[2] = H1(q1, p1)
    s[3] = H01(q1, p1)
    s[4] = H1(q3, p3)
    s[5] = H01(q3, p3)
    s[6] = qf[3]
    s[7] = qf[4]
    s[8] = (pf[2] + pf[4]) * γ - 1
    s[9:12] = qi1 - q1
    s[13:16] = pi1 - p1
    s[17:20] = qi2 - q2
    s[21:24] = pi2 - p2
    s[25:28] = qi3 - q3
    s[29:32] = pi3 - p3
end
```
We then initialize parameters to find the switching times. We identify the intervals where the control is near zero, indicating singular control, and determine the switching times.

```@example main
# Initialize parameters for finding switching times
t0 = 0
tol = 2e-2

# Find times where control is near zero (singular control)
t13 = [elt for elt in t if abs(u(elt)) < tol]
i = 1
t_l = []

# Identify intervals for switching times
while true
    global i, t13, t_l
    if (( i == length(t13)-1) || (t13[i+1] - t13[i] > 1) )
        break
    else 
        push!(t_l, t13[i])
        push!(t_l, t13[i+1])
        i += 1
    end
end

# Determine the switching times
t1 = min(t_l...)
t2 = max(t_l...)
t3f = [elt for elt in t13 if elt > t2]
t3 = min(t3f...)

# Extract initial and intermediate costates and states and final time
p0 = p(t0) 
q0 = [0, 1, 0, 1]
tf = solution_2000.objective
q1, p1 = q(t1), p(t1)
q2, p2 = q(t2), p(t2)
q3, p3 = q(t3), p(t3)
p0[1], q0[1], p0[3], q0[3]= -p0[1], -q0[1], -p0[3], -q0[3]
p1[1], q1[1], p1[3], q1[3]= -p1[1], -q1[1], -p1[3], -q1[3]
p2[1], q2[1], p2[3], q2[3]= -p2[1], -q2[1], -p2[3], -q2[3]
p3[1], q3[1], p3[3], q3[3]= -p3[1], -q3[1], -p3[3], -q3[3]
```

Next, we initialize the shooting function residuals and compute the initial residuals for the shooting function to verify the solution's accuracy. 

```@example main
# Initialize the shooting function residuals
s = similar(p0, 32)

# Compute the initial residuals for the shooting function
## shoot!(s, p0, t1, t2, t3, tf, q1, p1, q2, p2, q3, p3)
## println("Norm of the shooting function: ‖s‖ = ", norm(s), "\n")

```

We define a nonlinear equation solver for the shooting method. This solver refines the initial costate and switching times to find the optimal solution using the shooting function.

## TO BE COMPLETED

## References
[^1]: Bernard Bonnard, Olivier Cots, Jérémy Rouot, Thibaut Verron. Time minimal saturation of a pair of spins and application in magnetic resonance imaging. Mathematical Control and Related Fields, 2020, 10 (1), pp.47-88.