var documenterSearchIndex = {"docs":
[{"location":"bisaturation.html#Saturation-of-pair-of-spins","page":"Bisaturation","title":"Saturation of pair of spins","text":"","category":"section"},{"location":"bisaturation.html","page":"Bisaturation","title":"Bisaturation","text":"The problem we are trying to solve is the time minimal saturation of a pair of spin-12 particles (or bi-saturation problem), as described in [1]. This model describes a pair of spins that share the same characteristics, specifically the same relaxation times T_1 and T_2. However, the control field intensity differs for each spin due to variations as they transition from the North Pole N = (01) to the origin O=(00):","category":"page"},{"location":"bisaturation.html","page":"Bisaturation","title":"Bisaturation","text":"begincases\nt_f rightarrow min \ndotq(t) = F(q(t)) + u(t) G(q(t)) quad u(t) leq 1 quad t in 0 t_f \nq(0) = q_0 \nq(t_f) = q_f\nendcases","category":"page"},{"location":"bisaturation.html","page":"Bisaturation","title":"Bisaturation","text":"where q_0=0101, q_f=0000 and F and G are defined by (2) on page 5, as well as in sections 2.1 and 3.1.","category":"page"},{"location":"bisaturation.html","page":"Bisaturation","title":"Bisaturation","text":"We first define the problem.","category":"page"},{"location":"bisaturation.html","page":"Bisaturation","title":"Bisaturation","text":"using OptimalControl\nΓ = 9.855e-2\nγ = 3.65e-3\nϵ = 0.1\n@def ocp begin\n    tf ∈ R, variable\n    t ∈ [ 0, tf ], time\n    x ∈ R⁴, state\n    u ∈ R, control\n    tf ≥ 0\n    -1 ≤ u(t) ≤ 1\n    x(0) == [0, 1, 0, 1 ]\n    x(tf) == [0, 0, 0, 0]\n    ẋ(t) == [ (-Γ*x₁(t) -u(t)*x₂(t)), (γ*(1-x₂(t)) +u(t)*x₁(t)), (-Γ*x₃(t) -(1-ϵ)* u(t)*x₄(t)), (γ*(1-x₄(t)) +(1-ϵ)*u(t)*x₃(t))]\n    tf → min\nend","category":"page"},{"location":"bisaturation.html","page":"Bisaturation","title":"Bisaturation","text":"However, we quickly realize that solving this problem without any prior initial guesses is not feasible. This realization prompts us to explore potential solutions that facilitate problem-solving.","category":"page"},{"location":"bisaturation.html","page":"Bisaturation","title":"Bisaturation","text":"One effective approach involves homotopy on the initial condition. This method begins from an initial point where the problem can be resolved without requiring any initial guesses. Subsequently, we generate a sequence of initial guesses by solving intermediate problems created through homotopy, gradually progressing towards our original initial condition: 0 1 0 1. Using 1 0 1 0 as our initial guess revealed that it can serve as the starting point for this homotopy process.","category":"page"},{"location":"bisaturation.html#Homotopy-on-the-initial-condition","page":"Bisaturation","title":"Homotopy on the initial condition","text":"","category":"section"},{"location":"bisaturation.html","page":"Bisaturation","title":"Bisaturation","text":"The code below demonstrates how this approach systematically generates initial guesses using homotopy starting from 1 0 1 0, advancing towards the desired initial condition of 0 1 0 1.","category":"page"},{"location":"bisaturation.html","page":"Bisaturation","title":"Bisaturation","text":"Letus first define functions that define the optimal control problem with initial state x₀ and plot the solutions: ","category":"page"},{"location":"bisaturation.html","page":"Bisaturation","title":"Bisaturation","text":"# Define the optimal control problem with initial state x₀\nfunction g(x₀)\n    @def ocp begin\n        tf ∈ R, variable\n        t ∈ [0, tf], time\n        x ∈ R⁴, state\n        u ∈ R, control\n        tf ≥ 0\n        -1 ≤ u(t) ≤ 1\n        x(0) == x₀\n        x(tf) == [0, 0, 0, 0]\n        ẋ(t) == [ (-Γ*x₁(t) -u(t)*x₂(t)), \n                (γ*(1-x₂(t)) +u(t)*x₁(t)), \n                (-Γ*x₃(t) -(1-ϵ)* u(t)*x₄(t)), \n                (γ*(1-x₄(t)) +(1-ϵ)*u(t)*x₃(t))]\n        tf → min\n    end\n    return ocp\nend\n\nfunction plot_sol(sol)\n    q = sol.state\n    liste = [q(t) for t in sol.times]\n    liste_y1 = [elt[1] for elt in liste]\n    liste_z1 = [elt[2] for elt in liste]\n    liste_y2 = [elt[3] for elt in liste]\n    liste_z2 = [elt[4] for elt in liste]\n    plot(\n        plot(liste_y1, liste_z1, xlabel=\"y1\", ylabel=\"z1\"),\n        plot(liste_y2, liste_z2, xlabel=\"y2\", ylabel=\"z2\"),\n        plot(sol.times, sol.control, xlabel=\"Time\", ylabel=\"Control\")\n    )\nend","category":"page"},{"location":"bisaturation.html","page":"Bisaturation","title":"Bisaturation","text":"Then we perform homotopy on the initial condition with a step of 0.1,","category":"page"},{"location":"bisaturation.html","page":"Bisaturation","title":"Bisaturation","text":"ocp_x = g([1, 0, 1, 0])\nsol_x = solve(ocp_x, grid_size=100)\nsol_x.variable\nL_x = [sol_x]\nfor i in 1:10\n    x₀ = i / 10 * [0, 1, 0, 1] + (1 - i / 10) * [1, 0, 1, 0]\n    ocpi_x = g(x₀)\n    sol_i_x = solve(ocpi_x, grid_size=100, display=false, init=L_x[end]) \n    push!(L_x, sol_i_x)\nend\nnothing # hide","category":"page"},{"location":"bisaturation.html","page":"Bisaturation","title":"Bisaturation","text":"and plot the solutions.","category":"page"},{"location":"bisaturation.html","page":"Bisaturation","title":"Bisaturation","text":"solution_x = L_x[end]\nsolution_x.variable\nplot_sol(solution_x)","category":"page"},{"location":"bisaturation.html","page":"Bisaturation","title":"Bisaturation","text":"Conclusion: The solution is considered local, as the final time exceeds the one found using the Bocop software as mentioned in [1]. Let us now solve this problem differently. One potential initial guess could be obtained by solving a monosaturation problem where the control field intensity is the same for both spins, i.e., ϵ = 0, and then using homotopy to transition to ϵ = 01. We start by solving the problem for a single spin and then extend this approach to two identical spins before applying homotopy.","category":"page"},{"location":"bisaturation.html#Homotopy-on-ϵ","page":"Bisaturation","title":"Homotopy on ϵ","text":"","category":"section"},{"location":"bisaturation.html#Monosaturation-problem","page":"Bisaturation","title":"Monosaturation problem","text":"","category":"section"},{"location":"bisaturation.html","page":"Bisaturation","title":"Bisaturation","text":"@def ocp begin\n    tf ∈ R, variable\n    t ∈ [0, tf], time\n    q = (y, z) ∈ R², state\n    u ∈ R, control\n    tf ≥ 0\n    -1 ≤ u(t) ≤ 1\n    q(0) == [0, 1] # North pole\n    q(tf) == [0, 0] # Center\n    q̇(t) == [(-Γ * y(t) - u(t) * z(t)),( γ * (1 - z(t)) + u(t) * y(t))]\n    tf → min\nend\nN = 100\nsol = solve(ocp, grid_size=N) ","category":"page"},{"location":"bisaturation.html","page":"Bisaturation","title":"Bisaturation","text":"Now we extract the control, final time and then duplicate the state to be able to have an initial guess for the two spins.","category":"page"},{"location":"bisaturation.html","page":"Bisaturation","title":"Bisaturation","text":"u_init = sol.control  \nq_init(t) = [sol.state(t); sol.state(t)]\ntf_init = sol.variable","category":"page"},{"location":"bisaturation.html#Bi-saturation-problem","page":"Bisaturation","title":"Bi-saturation problem","text":"","category":"section"},{"location":"bisaturation.html","page":"Bisaturation","title":"Bisaturation","text":"@def ocp2 begin\n    tf ∈ R, variable\n    t ∈ [0, tf], time\n    q = (y₁, z₁, y₂, z₂) ∈ R⁴, state\n    u ∈ R, control\n    tf ≥ 0\n    -1 ≤ u(t) ≤ 1\n    q(0) == [0, 1, 0, 1] \n    q(tf) == [0, 0, 0, 0] \n    q̇(t) == [-Γ * y₁(t) - u(t) * z₁(t),\n            γ * (1 - z₁(t)) + u(t) * y₁(t),\n            -Γ * y₂(t) - u(t) * z₂(t),\n            γ * (1 - z₂(t)) + u(t) * y₂(t)]\n    tf → min\nend\n\ninit = (state=q_init, control=u_init, variable=tf_init)\nsol2 = solve(ocp2; grid_size=N, init=init)","category":"page"},{"location":"bisaturation.html","page":"Bisaturation","title":"Bisaturation","text":"We define the function below that computes the problem depending on varepsilon: ","category":"page"},{"location":"bisaturation.html","page":"Bisaturation","title":"Bisaturation","text":"# Define the optimal control problem with parameter ϵ\nfunction f(ϵ)\n    @def ocp begin\n        tf ∈ R, variable \n        t ∈ [0, tf], time \n        x ∈ R⁴, state    \n        u ∈ R, control    \n        tf ≥ 0            \n        -1 ≤ u(t) ≤ 1     \n        x(0) == [0, 1, 0, 1] \n        x(tf) == [0, 0, 0, 0] \n        \n        ẋ(t) == [ (-Γ*x₁(t) -u(t)*x₂(t)), \n                (γ*(1-x₂(t)) +u(t)*x₁(t)), \n                (-Γ*x₃(t) -(1-ϵ)* u(t)*x₄(t)), \n                (γ*(1-x₄(t)) +(1-ϵ)*u(t)*x₃(t))]\n        tf → min \n    end\n    return ocp\nend\n\nϵ = 0\ninitial_guess = sol2\nL = [sol2]\nfor i in 1:10\n    global ϵ = ϵ + 0.01\n    global initial_guess\n    ocpi = f(ϵ)\n    sol_i = solve(ocpi, grid_size=100, display=false, init=initial_guess)\n    global L\n    push!(L, sol_i)\n    initial_guess = sol_i\nend\nsol_eps = L[end]\nsol_eps.variable\nplot_sol(sol_eps)","category":"page"},{"location":"bisaturation.html","page":"Bisaturation","title":"Bisaturation","text":"Conclusion: The solution is a local one.","category":"page"},{"location":"bisaturation.html","page":"Bisaturation","title":"Bisaturation","text":"Another approach involves defining a bi-saturation problem with a slightly adjusted initial condition: q₀ = 01 09 01 09.","category":"page"},{"location":"bisaturation.html#Bi-Saturation-Problem:-initial-Guess-from-a-Slightly-Different-Problem","page":"Bisaturation","title":"Bi-Saturation Problem: initial Guess from a Slightly Different Problem","text":"","category":"section"},{"location":"bisaturation.html","page":"Bisaturation","title":"Bisaturation","text":"ϵ = 0.1\n\n@def ocpu begin\ntf ∈ R, variable\nt ∈ [0, tf], time\nx ∈ R⁴, state\nu ∈ R, control\ntf ≥ 0\n-1 ≤ u(t) ≤ 1\nx(0) == [0.1, 0.9, 0.1, 0.9] # Initial condition\nx(tf) == [0, 0, 0, 0] # Terminal condition\nẋ(t) == [(-Γ*x₁(t) -u(t)*x₂(t)),\n          (γ*(1-x₂(t)) +u(t)*x₁(t)),\n          (-Γ*x₃(t) -(1-ϵ)* u(t)*x₄(t)),\n          (γ*(1-x₄(t)) +(1-ϵ)*u(t)*x₃(t))]\n          tf → min\nend\ninitial_g = solve(ocpu, grid_size=100)\nocpf = f(ϵ)\nfor i in 1:10\n    global initial_g\n    solf = solve(ocpf, grid_size=i*100, init=initial_g)\n    initial_g = solf\nend\n# Plot the figures\nplot_sol(initial_g)","category":"page"},{"location":"bisaturation.html","page":"Bisaturation","title":"Bisaturation","text":"Conclusion: This solution is better that the two previous, which proves that these were strict local minimisers.","category":"page"},{"location":"bisaturation.html#References","page":"Bisaturation","title":"References","text":"","category":"section"},{"location":"bisaturation.html","page":"Bisaturation","title":"Bisaturation","text":"[1]: Bernard Bonnard, Olivier Cots, Jérémy Rouot, Thibaut Verron. Time minimal saturation of a pair of spins and application in magnetic resonance imaging. Mathematical Control and Related Fields, 2020, 10 (1), pp.47-88.","category":"page"},{"location":"index.html#spin","page":"Introduction","title":"spin","text":"","category":"section"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"spin is part of the control-toolbox ecosystem.","category":"page"},{"location":"index.html","page":"Introduction","title":"Introduction","text":"note: Install\nTo install a package from the control-toolbox ecosystem,  please visit the installation page.","category":"page"}]
}