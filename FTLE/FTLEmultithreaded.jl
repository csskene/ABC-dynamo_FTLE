using DifferentialEquations
using LinearAlgebra
using FileIO

if length(ARGS)==3
    Nx = parse(Int64,ARGS[1])
    Ny = parse(Int64,ARGS[2])
    Nt = parse(Int64,ARGS[3])
else
    println("Using default arguments")
    Nx = 32
    Ny = 32
    Nt = 2
end

println("Number of threads = ",Threads.nthreads())
println("Nx = ",Nx)
println("Ny = ",Ny)
println("Nt = ",Nt)
J = zeros((3,3))
J[1,1] = 1/sqrt(2)
J[2,2] = 1/sqrt(2)
J[3,3] = 0

S = zeros((3,3))
om = 1

function strain(t,y)
    S = zeros((3,3))
    S[1,:] = [0,cos(y[2]+cos(om*t)),0]
    S[2,:] = [-sin(y[1]+sin(om*t)),0,0]
    S[3,:] = [cos(y[1]+sin(om*t)),-sin(y[2]+cos(om*t)),0]
    S *= sqrt(3/2)
    return S
end

function rhsBig(y,p,t)
    dy0 = sqrt(3/2)*sin(y[2]+cos(om*t))
    dy1 = sqrt(3/2)*cos(y[1]+sin(om*t))
    dy2 = sqrt(3/2)*(cos(y[2]+cos(om*t))+sin(y[1]+sin(om*t)))
    S = strain(t,y[1:3])

    yp = reshape(y[4:end],(3,3))
    R = S*yp
    R = reshape(R,(9,))
    return vcat([dy0,dy1,dy2],R)
end

@time begin
T = 30
x = range(0,stop=2π,length=Nx)
y = range(0,stop=2π,length=Ny)
t = range(0,stop=2π/om,length=Nt+1)
t = t[1:end-1]
L = zeros((Nx,Ny,Nt))
Threads.@threads for k=1:Nt
    for (i,x0) in enumerate(x)
        for (j,y0) in enumerate(y)
            init = vcat([x0,y0,0], reshape(J,(9,)))
            tspan = (t[k],t[k]+T)
            prob = ODEProblem(rhsBig,init,tspan,save_everystep=false)
            sol = solve(prob, Tsit5())
            last = reshape(sol[end][4:end],(3,3))
            C = transpose(last)*last
            eigs = eigvals(C)
            L[i,j,k] = log(max(eigs...))/T
        end
    end
end
end

save("FTLE.jld2", "FTLE",L)
