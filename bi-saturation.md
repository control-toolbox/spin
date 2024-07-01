# [Saturation pair of spins and MRI](@id sat)
The problem that we are trying to solve is the PBS problem in page 18 of the paper [1], which models a couple of spins with the same characteristics, i.e. the same relaxation times $T\_1$ and $ T\_2 $, but for which for each, the control field has different intensities, because of inhomogeneities. 

'''math
        \begin{cases}
        J(u(\cdot), t_f) := t_f \rightarrow \min \\
        \dot{q}(t) = F(q(t)) + u(t) G(q(t)), \quad |u(t)| \leq 1, \quad t \in [0, t_f], \\
        q(0) = q_0, \\
        q(t_f) = q_f
        \end{cases}
'''