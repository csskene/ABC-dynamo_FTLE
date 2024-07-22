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
    om = parse(Float64,ARGS[4])
else
    println("Using default frequency")
    om = 1
end
println("Number of threads = ",Threads.nthreads())
println("Nx = ",Nx)
println("Ny = ",Ny)
println("Nt = ",Nt)
println("om = ",om)

# Define the initial condition for the deformation
J0 = zeros((3, 3))
J0[1,1] = 1/sqrt(2)
J0[2,2] = 1/sqrt(2)
J0[3,3] = 0

# Define the velocity
epsi = 1
out = zeros(3)
function Velocity!(out, t, y)
    out[1] = sin(y[3] + epsi*sin(om*t)) + cos(y[2] + epsi*sin(om*t))
    out[2] = sin(y[1] + epsi*sin(om*t)) + cos(y[3] + epsi*sin(om*t))
    out[3] = sin(y[2] + epsi*sin(om*t)) + cos(y[1] + epsi*sin(om*t))
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
    dy[4:end] = reshape(R, (9,))
end

# Timestepping parameters
T = 30
x = range(0, stop=2π, length=Nx)
y = range(0, stop=2π, length=Ny)
t = range(0, stop=2π/om, length=Nt+1)
t = t[1:end-1]
L = zeros((Nx, Ny, Nt))
init = zeros(12)
@time begin
    Threads.@threads for ijk in CartesianIndices(L)
        i = ijk[1]
        j = ijk[2]
        k = ijk[3]

        x0 = x[i]
        y0 = y[j]

        init[1:3] = [x0, y0, 0]
        init[4:end] = reshape(J0, (9,))

        tspan = (t[k], t[k]+T)
        prob = ODEProblem(rhs!, init, tspan, save_everystep=true)
        sol = solve(prob, Tsit5())
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
        L[i,j,k] = slope
    end
end

save("FTLEwobbly_Om_$om.jld2", "FTLE", L)
