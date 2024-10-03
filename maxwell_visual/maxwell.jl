# A Bernstein basis DG solver for the 3D Maxwell's equation.

using OrdinaryDiffEq
using StartUpDG
using LinearAlgebra
using SparseArrays
using StaticArrays
using BernsteinBasis

using Statistics

###############################################################################
# Reference: Hesthaven and Warburton pg. 432-437
# Let u represent the state of the system where u = (Hx, Hy, Hz, Ex, Ey, Ez), where
# Hx, Hy, Hz, Ex, Ey, Ez are functions of x, y, z, t.
#
# We frame the problem as
#
# du/dt + dfx(u)/dx + dfy(u)/dy + dfz(u)/dz = 0.
# where fx(u) = (0, Ez, -Ey, 0, -Hz, Hy), fy(u) = (-Ez, 0, Ex, Hz, 0, -Hx), fz = (Ey, -Ex, 0, -Hy, Hx, 0).

function fx(u)
    Hx, Hy, Hz, Ex, Ey, Ez = u
    return SVector(0, Ez, -Ey, 0, -Hz, Hy)
end

function fy(u)
    Hx, Hy, Hz, Ex, Ey, Ez = u
    return SVector(-Ez, 0, Ex, Hz, 0, -Hx)
end

function fz(u)
    Hx, Hy, Hz, Ex, Ey, Ez = u
    return SVector(Ey, -Ex, 0, -Hy, Hx, 0)
end

# Computes d(Hx, Hy, Hz, Ex, Ey, Ez)/dt as a function of u = (Hx, Hy, Hz, Ex, Ey, Ez).
function rhs_matvec!(du, u, params, t)
    (; rd, md, Dr, Ds, Dt, LIFT) = params
    
    (; uM, interface_flux, duM, dfxdr, dfxds, dfxdt, dfydr, dfyds, dfydt, dfzdr, dfzds, dfzdt, fxu, fyu, fzu) = params.cache
    
    uM .= view(u, rd.Fmask, :)

    @inbounds for e in axes(uM, 2)
        for i in axes(uM, 1)
            duM = uM[md.mapP[i,e]] - uM[i,e]
            ndotdH =    md.nxJ[i,e] * duM[1] + 
                        md.nyJ[i,e] * duM[2] +
                        md.nzJ[i,e] * duM[3]
            ndotdE =    md.nxJ[i,e] * duM[4] + 
                        md.nyJ[i,e] * duM[5] +
                        md.nzJ[i,e] * duM[6]
            interface_flux[i, e] = 0.5 * fx(duM) * md.nxJ[i,e] +
                                0.5 * fy(duM) * md.nyJ[i,e] +
                                0.5 * fz(duM) * md.nzJ[i,e] + 
                                0.5 * (duM) -
                                0.5 * SVector(
                                    ndotdH * md.nxJ[i,e],
                                    ndotdH * md.nyJ[i,e],
                                    ndotdH * md.nzJ[i,e],
                                    ndotdE * md.nxJ[i,e],
                                    ndotdE * md.nyJ[i,e],
                                    ndotdE * md.nzJ[i,e],
                                )
        end
    end

    @inbounds for e in axes(du, 2)
        fxu .= fx.(view(u, :, e))
        fyu .= fy.(view(u, :, e))
        fzu .= fz.(view(u, :, e))

        mul!(view(dfxdr, :, e), Dr, fxu)
        mul!(view(dfxds, :, e), Ds, fxu)
        mul!(view(dfxdt, :, e), Dt, fxu)
        mul!(view(dfydr, :, e), Dr, fyu)
        mul!(view(dfyds, :, e), Ds, fyu)
        mul!(view(dfydt, :, e), Dt, fyu)
        mul!(view(dfzdr, :, e), Dr, fzu)
        mul!(view(dfzds, :, e), Ds, fzu)
        mul!(view(dfzdt, :, e), Dt, fzu)

        mul!(view(du, :, e), LIFT, view(interface_flux, :, e))

        for i in axes(du, 1)
            du[i, e] += md.rxJ[1, e] * dfxdr[i, e] + md.sxJ[1, e] * dfxds[i, e] + md.txJ[1, e] * dfxdt[i, e] + 
            md.ryJ[1, e] * dfydr[i, e] + md.syJ[1, e] * dfyds[i, e] + md.tyJ[1, e] * dfydt[i, e] + 
            md.rzJ[1, e] * dfzdr[i, e] + md.szJ[1, e] * dfzds[i, e] + md.tzJ[1, e] * dfzdt[i, e]
        end

        # if mean(md.y[:, e]) > 0
        #     du[:,e] *= 0.5
        # end
    end

    # Note md.J is constant matrix.
    @. du = du / md.J[1,1]
end

# @time rhs_matvec!(similar(u), u, params, 0);
# @code_warntype rhs_matvec!(similar(u), u, params, 0);

# Set polynomial order
N = 8

rd = RefElemData(Tet(), N)

(; r, s, Fmask) = rd
Fmask = reshape(Fmask, :, 4)
rf, sf = rd.r[Fmask[:,1]], rd.t[Fmask[:,1]]

rd = RefElemData(Tet(), N; quad_rule_face = (rf, sf, ones(length(rf))))
md = MeshData(uniform_mesh(rd.element_type, 9), rd;               
              is_periodic=true)
              
# Problem setup
tspan = (0.0, 0.75)
(; x, y, z) = md

# Parameters for the Gaussian pulse
σ = 0.1  # Width of the pulse
kx = 2 * pi  # Wave number in x-direction
ky = 2 * pi  # Wave number in y-direction
kz = 2 * pi  # Wave number in z-direction

# Initial Conditions
# Initial Conditions
# Initial Conditions
σ = 0.2  # Width of the Gaussian pulse (controls smoothness)
center_x, center_y, center_z = 0.0, -0.25, 0.0  # Center of the Gaussian below y=0.5
amplitude_Hx = 10  # Amplitude for Hx
amplitude_Ex = 10  # Amplitude for Ex

# Initial Conditions: Smooth Expanding Wave Centered Below y=0.5
u0 = @. SVector{6, Float64}(
    amplitude_Hx * exp(-((x - center_x)^2 + (y - center_y)^2 + (z - center_z)^2) / σ^2),  # Hx: Gaussian pulse
    0.0,  # Hy: Initially zero
    0.0,  # Hz: Initially zero
    amplitude_Ex * exp(-((x - center_x)^2 + (y - center_y)^2 + (z - center_z)^2) / σ^2),  # Ex: Gaussian pulse
    0.0,  # Ey: Initially zero
    0.0   # Ez: Initially zero
)


# u0 = @. SVector{6, Float64}(
#     0, 
#     0, 
#     0, 
#     3 * exp(-50((x-0.5)^2 + y^2 + z^2)),
#     0, 
#     0)

# E0 = 3.0                # Amplitude of the electric field
# alpha = 50.0            # Width parameter for the Gaussian
# x0 = 0.5                # Center of the Gaussian in the x-direction
# eta = 1.0               # Intrinsic impedance (set to 1 for simplicity)

# H0 = E0 / eta

# u0 = @. SVector{6, Float64}(
#     0,                                       # Hx
#     0,                                       # Hy
#     H0 * exp(-alpha * ((x - x0)^2 + y^2 + z^2)),  # Hz
#     0,                                       # Ex
#     E0 * exp(-alpha * ((x - x0)^2 + y^2 + z^2)),  # Ey
#     0                                        # Ez
# )

# Convert initial conditions to Bernstein coefficients
(; r, s, t) = rd
vande, _ = bernstein_basis(Tet(), N, r, s, t)
modal_u0 = inv(vande) * u0

# Initialize operators
Dr = BernsteinDerivativeMatrix_3D_r(N)
Ds = BernsteinDerivativeMatrix_3D_s(N)
Dt = BernsteinDerivativeMatrix_3D_t(N)
LIFT = BernsteinLift{SVector{6, Float64}}(N)

# Cache temporary arrays (values are initialized to get the right dimensions)
cache = (; uM = modal_u0[rd.Fmask, :], interface_flux = modal_u0[rd.Fmask, :], 
           duM = modal_u0[rd.Fmask, :],
           dfxdr = similar(modal_u0), dfxds = similar(modal_u0), dfxdt = similar(modal_u0),
           dfydr = similar(modal_u0), dfyds = similar(modal_u0), dfydt = similar(modal_u0),
           dfzdr = similar(modal_u0), dfzds = similar(modal_u0), dfzdt = similar(modal_u0),
           fxu = similar(modal_u0[:, 1]), fyu = similar(modal_u0[:, 1]), fzu = similar(modal_u0[:, 1]))

# Combine parameters
params = (; rd, md, Dr, Ds, Dt, LIFT, cache)

# Solve ODE system
ode = ODEProblem(rhs_matvec!, modal_u0, tspan, params)

sol = solve(ode, Tsit5(), saveat=LinRange(tspan..., 25))

# Convert Bernstein coefficients back to point evaluations
u = vande * sol.u[end]

# Test against analytical solution
u_exact = @. SVector{6, Float64}(
    0, 
    0, 
    0, 
    exp(-50((x-0.5)^2 + y^2 + z^2)),
    0, 
    0)
@show norm(u - u_exact, Inf)

# Visualization

using NodesAndModes

equi_node = NodesAndModes.equi_nodes(Tet(), N)
equi_vande = bernstein_basis(Tet(), N, equi_node[1], equi_node[2], equi_node[3])[1]

for i in 1:length(sol.u)
    u_equi = equi_vande * sol.u[i]
    v = map((x) -> x[5], u_equi)
    filename = "3d_maxwell_t" * string(Int(round(tspan[2] / length(sol.u) * i * 100)))
    StartUpDG.MeshData_to_vtk(md, rd, [v], ["x"], filename, true, true)
end