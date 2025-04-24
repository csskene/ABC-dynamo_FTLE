using DifferentialEquations
using LinearAlgebra
using ForwardDiff
using FileIO

# Parse the command line arguments
if length(ARGS)>=3
    Nx = parse(Int64, ARGS[1])
    Ny = parse(Int64, ARGS[2])
    Nt = parse(Int64, ARGS[3])
else
    println("Using default resolution")
    Nx = 32
    Ny = 32
    Nt = 1
end
if length(ARGS)==4
    om = parse(Float64, ARGS[4])
else
    println("Using default frequency")
    om = 1
end
println("Number of threads = ", Threads.nthreads())
println("Nx = ", Nx)
println("Ny = ", Ny)
println("Nt = ", Nt)
println("om = ", om)

# Define the initial condition for the deformation
J0 = zeros((3, 3))
J0[1,1] = 1/sqrt(2)
J0[2,2] = 1/sqrt(2)
J0[3,3] = 0

# Define the velocity
epsi = 1
out = zeros(3)
function Velocity!(out, t, y)
    eps_sin_om_t = epsi*sin(om*t)
    out[1] = sin(y[3] + eps_sin_om_t) + cos(y[2] + eps_sin_om_t)
    out[2] = sin(y[1] + eps_sin_om_t) + cos(y[3] + eps_sin_om_t)
    out[3] = sin(y[2] + eps_sin_om_t) + cos(y[1] + eps_sin_om_t)
end

function rhs!(dy, y, p, t)
    # Get position and J from y
    position = y[1:3]
    J = reshape(y[4:end], (3, 3))
    # Get the velocity and the gradient of velocity (with automatic differentiation)
    # at the current position
    result = DiffResults.JacobianResult(position)
    result = ForwardDiff.jacobian!(result, (out, position)->Velocity!(out, t, position), out, position)
    U = DiffResults.value(result)
    dudx = DiffResults.jacobian(result)
    # Multiply J by the velocity gradient
    R = dudx*J
    # Set the output
    dy[1:3] = U
    dy[4:end] = reshape(R, (9, ))
end

# Timestepping parameters
T = 30
x_range = range(0, stop=2π, length=Nx)
y_range = range(0, stop=2π, length=Ny)
t_range = range(0, stop=2π/om, length=Nt+1)
t_range = t_range[1:end-1]
grid = [[x, y, t] for x in x_range, y in y_range, t in t_range]

function prob_func(prob, i, repeat)
    init = zeros(12)
    xx, yy, tt = grid[i]
    init[1:3] = [xx, yy, 0]
    init[4:end] = reshape(J0, (9,))
    remake(prob, u0=init, tspan=(tt, tt + T))
end

function output_func(sol, i)
    # FTLE calculation
    C = zeros(length(sol[1,:]))
    jacs = reshape(sol[4:end,:], (3, 3, length(sol[1,:]))) 
    for b in 1:length(sol[1,:])
        C[b] = tr(transpose(jacs[:,:,b])*jacs[:,:,b])
        C[b] = log(C[b]/2)/2
    end
    # Least-squares
    X = hcat(ones(length(sol.t)), sol.t)
    # intercept, slope = inv(X'*X)\(X'*C)
    intercept, slope = (X'*X)\(X'*C)
    slope, false
end

prob = ODEProblem(rhs!, init, (0.0, T))
ensemble_prob = EnsembleProblem(prob, prob_func=prob_func, output_func=output_func)
@time begin
    sol = solve(ensemble_prob, Tsit5(), EnsembleThreads(), trajectories=Nx*Ny*Nt)
end

L = reshape(sol.u, (Nx, Ny, Nt))
save("FTLEwobbly_Om_$om.jld2", "FTLE", L)
