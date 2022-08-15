using DifferentialEquations
using LinearAlgebra
using FileIO

if length(ARGS)==3
    Nx = parse(Int64,ARGS[1])
    Ny = parse(Int64,ARGS[2])
    Nt = parse(Int64,ARGS[3])
else
    println("Using default resolution")
    Nx = 32
    Ny = 32
    Nt = 2
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
J = zeros((3,3))
J[1,1] = 1/sqrt(2)
J[2,2] = 1/sqrt(2)
J[3,3] = 0

S = zeros((3,3))

epsi = 1
function strain(t,y)
    S = zeros((3,3))
    S[1,:] = [0,-sin(y[2]+epsi*sin(om*t)),cos(y[3]+epsi*sin(om*t))]
    S[2,:] = [cos(y[1]+epsi*sin(om*t)),0,-sin(y[3]+epsi*sin(om*t))]
    S[3,:] = [-sin(y[1]+epsi*sin(om*t)),cos(y[2]+epsi*sin(om*t)),0]
    return S
end

function rhsBig(y,p,t)
    dy0 = sin(y[3]+epsi*sin(om*t)) + cos(y[2]+epsi*sin(om*t))
    dy1 = sin(y[1]+epsi*sin(om*t)) + cos(y[3]+epsi*sin(om*t))
    dy2 = sin(y[2]+epsi*sin(om*t)) + cos(y[1]+epsi*sin(om*t))

    S = strain(t,y[1:3])

    yp = reshape(y[4:end],(3,3))
    R = S*yp
    R = reshape(R,(9,))
    return vcat([dy0,dy1,dy2],R)
end

# @time begin
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
            prob = ODEProblem(rhsBig,init,tspan,save_everystep=true)
            sol = solve(prob, Tsit5())
            jacs = reshape(sol[4:end,:],(3,3,length(sol[1,:])))
            C = zeros(length(sol[1,:]))
            for i in 1:length(sol[1,:])
                C[i] = tr(transpose(jacs[:,:,i])*jacs[:,:,i])
                C[i] = log(C[i]/2)/2
            end
            X = hcat(ones(length(sol.t)),sol.t)
            intercept,slope = inv(X'*X)*(X'*C)
            L[i,j,k] = slope
        end
    end
end
# end

save("FTLEwobbly_Om_$om.jld2", "FTLE",L)
