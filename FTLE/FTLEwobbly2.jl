using FileIO
using Distributed

@everywhere include("ABC.jl");
@everywhere using SparseArrays
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
for k in 1:Nt
    println("k = ",k)
    Lxy = @distributed (+) for ij in CartesianIndices(Ls[:,:,k])
        Lxy = spzeros((Nx,Ny))
        i = ij[1]
        j = ij[2]

        x0 = x[i]
        y0 = y[j]

        Lxy[i,j] = calcFTLE(x0,y0,t[k],om,epsi)
        Lxy
    end
    Ls[:,:,k] = Lxy
end

# Ls[:,:,:] = Ltmp[:,:,:]
save("FTLEwobbly_Om_$om.jld2", "FTLE",Ls)
