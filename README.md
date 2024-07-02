# spin
Spin

## Lieu

LJAD / McTAO, UniCA (Nice / Sophia)
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

## Sujet

Dans le cadre du projet ct: control-toolbox [^1] on s'intéresse à la modélisation et à la résolution en langage Julia de problèmes de contrôle optimal quantique. On cherchera notamment à reproduire certains des résultats obtenus dans [^2] sur le contrôle de deux spins couplés par un champ magnétique (MRI).

[^1]: [control-toolbox.org](https://control-toolbox.org)

[^2]: Bonnard, B.; Cots, O.; Rouot, J.; Verron, T. Time minimal saturation of a pair of spins and application in magnetic resonance imaging. Mathematical Control and Related Fields, 2020, 10 (1), pp.47-88. https://inria.hal.science/hal-01779377
