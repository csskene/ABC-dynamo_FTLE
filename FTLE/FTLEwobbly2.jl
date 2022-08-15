using FileIO
using Distributed
using SharedArrays
@everywhere include("ABC.jl");

if length(ARGS)>=3
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
epsi = 1
println("Number of threads = ",Threads.nthreads())
println("Nx = ",Nx)
println("Ny = ",Ny)
println("Nt = ",Nt)
println("om = ",om)

x = range(0,stop=2π,length=Nx)
y = range(0,stop=2π,length=Ny)
t = range(0,stop=2π/om,length=Nt+1)
t = t[1:end-1]
Ls = zeros((Nx,Ny,Nt))
L = SharedArray{Float64}((Nx,Ny,Nt))
@time begin
    @sync @distributed for ijk in CartesianIndices(Ls)
        i = ijk[1]
        j = ijk[2]
        k = ijk[3]

        x0 = x[i]
        y0 = y[j]

        L[i,j,k] = calcFTLE(x0,y0,t[k],om,epsi)
    end
end

Ls[:,:,:] = L[:,:,:]
save("FTLEwobbly_Om_$om.jld2", "FTLE",Ls)
