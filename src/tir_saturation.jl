using OptimalControl
using Plots
include("bi_saturation.jl")
# Define the parameters of the problem
Γ = 9.855e-2  
γ = 3.65e-3  
ϵ =0.1
function F_2(q::Vector{Float64})
    return [-Γ*q[1], γ*(1-q[2])]
end 
function F(q::Vector{Float64})
    return [F_2(q[1:2]); F_2(q[3:4])]
end 
function G_2(q::Vector{Float64})
    return [-q[2],q[1]]
end
function G(q::Vector{Float64})
    return [G_2(q[1:2]); (1- ϵ)*G_2(q[3:4])]
end
@def ocp_t begin
    tf ∈ R, variable
    t ∈ [0, tf], time
    q ∈ R⁴, state
    u ∈ R, control
    tf ≥ 0
    -1 ≤ u(t) ≤ 1
    q(0) == [0, 1, 0, 1]
    q(tf) == [0, 0, 0, 0]
    q̇(t) == F(q(t))+ G(q(t))*u(t)
    tf → min
end

HF = Lift(F) 
HG = Lift(G)
H01  = @Lie { HF, HG }
H001 = @Lie { HF, H01 }
H101 = @Lie { HG, H01 }
us(x, p) = -H001(x, p) / H101(x, p)
t = initial_g.times
q = initial_g.state
u = initial_g.control
p = initial_g.costate
φ(t) = HG(q(t), p(t))
uₘ = 1

function shoot(p0, tf , t1, t2, t3, z1, z2, z3)
    s = zeros(eltype(p0), 32)
    q0 = [0, 1, 0, 1]
    q1 = z1[1]
    q2 = z2[1]
    q3 = z3[1]
    p1 = z1[2]
    p2 = z2[2]
    p3 = z3[2]
    u0 = sign(HG(q0, p0))
    f0 = Flow(ocp, (q, p, tf) -> uₘ)
    f1 = Flow(ocp, (q, p, tf) -> -uₘ)
    fs = Flow(ocp, (q, p, tf) -> us(q, p))
    qi1, pi1 = f1(0, q0, p0, t1)
    qi2, pi2 = fs(t1, q1, p1, t2)
    qi3, qi3 = f0(t2, q2, p2, t3)
    qif, pif = fs(t3, q3, p3, tf)
    s[1] = u0*HG(q0, p0) -1 
    s[2] = HG(q1, p1)
    s[3] = H01(q1, p1)
    s[4] = HG(q3, p3)
    s[5] = H01(q3, p3)
    s[6] = q(t3)[3]
    s[7] = q(t3)[4]
    s[8] = (pif(tf)[2] + pif(tf)[4])*γ -1
    s[9:12] = qi1- q1
    s[13:16] = pi1- p1
    s[17:20] = qi2- q2
    s[21:24] = pi2- p2
    s[25:28] = qi3- q3
    s[29:32] = pi3- p3

    
end
