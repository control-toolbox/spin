# spin
Spin

## Lieu

LJAD / McTAO, UniCA (Nice / Sophia)
$$
\begin{cases}
J(u(\cdot), t_f) := t_f \rightarrow \min \\
\dot{q}(t) = F(q(t)) + u(t) G(q(t)), \quad |u(t)| \leq 1, \quad t \in [0, t_f], \\
q(0) = q_0, \\
q(t_f) = q_f
\end{cases}
$$


## Sujet

Dans le cadre du projet ct: control-toolbox [^1] on s'intéresse à la modélisation et à la résolution en langage Julia de problèmes de contrôle optimal quantique. On cherchera notamment à reproduire certains des résultats obtenus dans [^2] sur le contrôle de deux spins couplés par un champ magnétique (MRI).

[^1]: [control-toolbox.org](https://control-toolbox.org)

[^2]: Bonnard, B.; Cots, O.; Rouot, J.; Verron, T. Time minimal saturation of a pair of spins and application in magnetic resonance imaging. Mathematical Control and Related Fields, 2020, 10 (1), pp.47-88. https://inria.hal.science/hal-01779377
