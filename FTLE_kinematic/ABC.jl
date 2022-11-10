using DifferentialEquations
using LinearAlgebra

T = 30

J = zeros((3,3))
J[1,1] = 1/sqrt(2)
J[2,2] = 1/sqrt(2)
J[3,3] = 0

function strain(t,y,epsi,om)
    S = zeros((3,3))
    S[1,:] = [0,-sin(y[2]+epsi*sin(om*t)),cos(y[3]+epsi*sin(om*t))]
    S[2,:] = [cos(y[1]+epsi*sin(om*t)),0,-sin(y[3]+epsi*sin(om*t))]
    S[3,:] = [-sin(y[1]+epsi*sin(om*t)),cos(y[2]+epsi*sin(om*t)),0]
    return S
end

function rhsBig(y,p,t)
    om   = p[1]
    epsi = p[2]
    dy0 = sin(y[3]+epsi*sin(om*t)) + cos(y[2]+epsi*sin(om*t))
    dy1 = sin(y[1]+epsi*sin(om*t)) + cos(y[3]+epsi*sin(om*t))
    dy2 = sin(y[2]+epsi*sin(om*t)) + cos(y[1]+epsi*sin(om*t))

    S = strain(t,y[1:3],epsi,om)

    yp = reshape(y[4:end],(3,3))
    R = S*yp
    R = reshape(R,(9,))
    return vcat([dy0,dy1,dy2],R)
end

function calcFTLE(x0,y0,t0,om,epsi)
    init = vcat([x0,y0,0], reshape(J,(9,)))
    tspan = (t0,t0+T)
    prob = ODEProblem(rhsBig,init,tspan,[om,epsi],save_everystep=true)
    sol = solve(prob, Tsit5())
    jacs = reshape(sol[4:end,:],(3,3,length(sol[1,:])))
    C = zeros(length(sol[1,:]))
    for b in 1:length(sol[1,:])
        C[b] = tr(transpose(jacs[:,:,b])*jacs[:,:,b])
        C[b] = log(C[b]/2)/2
    end
    X = hcat(ones(length(sol.t)),sol.t)
    intercept,slope = inv(X'*X)*(X'*C)
    return slope
end
