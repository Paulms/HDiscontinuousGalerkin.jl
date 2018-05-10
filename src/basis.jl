"""
Return the dimension of an `Interpolation`
"""
@inline getdim(ip::Interpolation{dim}) where {dim} = dim

"""
Return the reference shape of an `Interpolation`
"""
@inline getrefshape(ip::Interpolation{dim,shape}) where {dim,shape} = shape

"""
Return the polynomial order of the `Interpolation`
"""
@inline getorder(ip::Interpolation{dim,shape,order}) where {dim,shape,order} = order

"""
Compute the value of the shape functions at a point ξ for a given interpolation
"""
function value(ip::Interpolation{dim}, ξ::Vec{dim,T}) where {dim,T}
    [value(ip, i, ξ) for i in 1:getnbasefunctions(ip)]
end

"""
Compute the gradients of the shape functions at a point ξ for a given interpolation
"""
function derivative(ip::Interpolation{dim}, ξ::Vec{dim,T}) where {dim,T}
    [gradient_value(ip, i, ξ) for i in 1:getnbasefunctions(ip)]
end

function value(ip::Interpolation{dim}, j::Int, ξ::Tuple) where {dim}
    value(ip,j,Vec{dim}(ξ))
end

function gradient_value(ip::Interpolation{dim}, j::Int, ξ::Tuple) where {dim}
    gradient_value(ip,j,Vec{dim}(ξ))
end
############
# Dubiner #
###########
struct Dubiner{dim,shape,order} <: Interpolation{dim,shape,order} end

getnbasefunctions(::Dubiner{2,RefTetrahedron,order}) where {order} = (order+1)*(order+2)/2
#nvertexdofs(::Dubiner{2,RefTetrahedron,order}) = 1

#vertices(::Dubiner{2,RefTetrahedron,order}) where {order} = (1,2,3)
#faces(::Dubiner{2,RefTetrahedron,order}) where {order} = ((1,2), (2,3), (3,1))

function reference_coordinates(::Dubiner{2,RefTetrahedron,order}) where {order}
    return [Vec{2, Float64}((1.0, 0.0)),
            Vec{2, Float64}((0.0, 1.0)),
            Vec{2, Float64}((0.0, 0.0))]
end

"""
value(ip::Dubiner{2,RefTetrahedron,order}, j::Int, ξ::AbstactVector) where {order}
Compute value of dubiner basis `j` at point ξ
on the reference triangle ((0,0),(1,0),(0,1))
"""
function value(ip::Dubiner{2,RefTetrahedron,order}, j::Int, ξ::Vec{2,T}) where {order, T}
    r = ξ[1]
    s = ξ[2]
    if j == 0; return zero(T)
    elseif j == 1; return sqrt(2)*one(T)
    elseif j == 2; return 2*sqrt(3)*(2*r + s - 1)
    elseif j == 3; return 2*(3*s - 1)
    elseif j == 4; return sqrt(30)*(6*r^2 + 6*r*(s - 1) + s^2 - 2*s + 1)
    elseif j == 5; return 3*sqrt(2)*(5*s - 1)*(2*r + s - 1)
    elseif j == 6; return sqrt(6)*(10*s^2 - 8*s + 1)
    elseif j == 7; return 2*sqrt(14)*(2*r + s - 1)*(10*r^2 + 10*r*(s - 1) + s^2 - 2*s + 1)
    elseif j == 8; return 2*sqrt(10)*(7*s - 1)*(6*r^2 + 6*r*(s - 1) + s^2 - 2*s + 1)
    elseif j == 9; return 2*sqrt(6)*(21*s^2 - 12*s + 1)*(2*r + s - 1)
    elseif j == 10; return 2*sqrt(2)*(35*s^3 - 45*s^2 + 15*s - 1)
    elseif j == 11; return 3*sqrt(10)*(70*r^4 + 140*r^3*(s - 1) + 90*r^2*(s^2 - 2*s + 1) + 20*r*(s^3 - 3*s^2 + 3*s - 1) + s^4 - 4*s^3 + 6*s^2 - 4*s + 1)
    elseif j == 12; return sqrt(70)*(9*s - 1)*(2*r + s - 1)*(10*r^2 + 10*r*(s - 1) + s^2 - 2*s + 1)
    elseif j == 13; return 5*sqrt(2)*(36*s^2 - 16*s + 1)*(6*r^2 + 6*r*(s - 1) + s^2 - 2*s + 1)
    elseif j == 14; return sqrt(30)*(84*s^3 - 84*s^2 + 21*s - 1)*(2*r + s - 1)
    elseif j == 15; return sqrt(10)*(126*s^4 - 224*s^3 + 126*s^2 - 24*s + 1)
    else; return dubiner_basis(r,s,j)
    end
end

"""
gradient_value(ip::Dubiner{2,RefTetrahedron,order}, j::Int, ξ::AbstactVector) where {order}
Compute value of dubiner basis `j` derivative at point ξ
on the reference triangle ((0,0),(1,0),(0,1))
"""
function gradient_value(ip::Dubiner{2,RefTetrahedron,order}, j::Int, ξ::Vec{2,T}) where {order,T}
    r = ξ[1]
    s = ξ[2]
    if j == 0; return [zero(T),zero(T)]
    elseif j == 1; return [zero(T),zero(T)]
    elseif j == 2; return [4*sqrt(3), 2*sqrt(3)]*one(T)
    elseif j == 3; return [zero(T), one(T)*6]
    elseif j == 4; return [12*sqrt(30)*r + 6*sqrt(30)*(s - 1),
                            6*sqrt(30)*r + 2*sqrt(30)*s - 2*sqrt(30)]
    elseif j == 5; return [6*sqrt(2)*(5*s - 1),
                            30*sqrt(2)*r + 30*sqrt(2)*s - 18*sqrt(2)]
    elseif j == 6; return [zero(T),20*sqrt(6)*s - 8*sqrt(6)]
    elseif j == 7; return [24*sqrt(14)*(5*r^2 + 5*r*(s - 1) + s^2 - 2*s + 1),
                            60*sqrt(14)*r^2 + r*(48*sqrt(14)*s - 48*sqrt(14)) + 6*sqrt(14)*s^2 - 12*sqrt(14)*s + 6*sqrt(14)]
    elseif j == 8; return [12*sqrt(10)*(7*s - 1)*(2*r + s - 1),
                            84*sqrt(10)*r^2 + r*(168*sqrt(10)*s - 96*sqrt(10)) + 42*sqrt(10)*s^2 - 60*sqrt(10)*s + 18*sqrt(10)]
    elseif j == 9; return [4*sqrt(6)*(21*s^2 - 12*s + 1),
                            r*(168*sqrt(6)*s - 48*sqrt(6)) + 126*sqrt(6)*s^2 - 132*sqrt(6)*s + 26*sqrt(6)]
    elseif j == 10; return [zero(T),210*sqrt(2)*s^2 - 180*sqrt(2)*s + 30*sqrt(2)]
    elseif j == 11; return [840*sqrt(10)*r^3 + 1260*sqrt(10)*r^2*(s - 1) + 540*sqrt(10)*r*(s^2 - 2*s + 1) + 60*sqrt(10)*(s^3 - 3*s^2 + 3*s - 1),
                            420*sqrt(10)*r^3 + r^2*(540*sqrt(10)*s - 540*sqrt(10)) + r*(180*sqrt(10)*s^2 - 360*sqrt(10)*s + 180*sqrt(10)) + 12*sqrt(10)*s^3 - 36*sqrt(10)*s^2 + 36*sqrt(10)*s - 12*sqrt(10)]
    elseif j == 12; return [12*sqrt(70)*(9*s - 1)*(5*r^2 + 5*r*(s - 1) + s^2 - 2*s + 1),
                            180*sqrt(70)*r^3 + r^2*(540*sqrt(70)*s - 300*sqrt(70)) + r*(324*sqrt(70)*s^2 - 456*sqrt(70)*s + 132*sqrt(70)) + 36*sqrt(70)*s^3 - 84*sqrt(70)*s^2 + 60*sqrt(70)*s - 12*sqrt(70)]
    elseif j == 13; return [30*sqrt(2)*(36*s^2 - 16*s + 1)*(2*r + s - 1),
                            r^2*(2160*sqrt(2)*s - 480*sqrt(2)) + r*(3240*sqrt(2)*s^2 - 3120*sqrt(2)*s + 510*sqrt(2)) + 720*sqrt(2)*s^3 - 1320*sqrt(2)*s^2 + 690*sqrt(2)*s - 90*sqrt(2)]
    elseif j == 14; return [2*sqrt(30)*(84*s^3 - 84*s^2 + 21*s - 1),
                            r*(504*sqrt(30)*s^2 - 336*sqrt(30)*s + 42*sqrt(30)) + 336*sqrt(30)*s^3 - 504*sqrt(30)*s^2 + 210*sqrt(30)*s - 22*sqrt(30)]
    elseif j == 15; return [zero(T),504*sqrt(10)*s^3 - 672*sqrt(10)*s^2 + 252*sqrt(10)*s - 24*sqrt(10)]
    else; return ∇dubiner_basis(r,s,j)
    end
end

"""
djacobi(x,n::Integer,α, β)
Evaluate the `n` order jacobi polynomial derivative
at point x∈[-1,1]
"""
function djacobi(x,n::Integer,α, β)
    T = typeof((2+α+β)*x/2)
    a = zero(T)
    if n <= 0;return a;end
    return (n+α+β+1)/2*jacobi(x,n-1,α+1,β+1)
end

"""
dubiner(x,y,n::Integer,m::Integer)
Compute the dubiner polynomial of degree `n`,`m` at point (x,y)
on the reference triangle ((0,0),(1,0),(0,1))
"""
function dubiner(x,y,n::Integer,m::Integer)
    #check domain
    @assert ((y>=0)&&(x>=0))&&(1>=x+y) "point not in domain"
    # Map to reference square
    ξ=2*x/(1-y)-1
    η=2*y-1
    k=2*n+1
    #Compute Dubiner_nm(ξ, η)
    P=jacobi(ξ,n,0,0)*jacobi(η, m,k,0)*((1-η)/2)^n
    #normalize
    N=sqrt(2/((2*n+1)*(m+n+1)))
    return (2*P)/N
end

"""
∇dubiner(x,y,n::Integer,m::Integer)
Compute the gradient of dubiner polynomial of degree `(n,m)` at point (x,y)
on the reference triangle ((0,0),(1,0),(0,1))
"""
function ∇dubiner(x,y,n::Integer,m::Integer)
    #check domain
    @assert ((y>=0)&&(x>=0))&&(1>=x+y) "point not in domain"
    # Map to reference square
    ξ=2*x/(1-y)-1
    k=2*n+1
    η=2*y-1
    #Compute ∇Dubiner_nm(ξ, η)
    Dφn = djacobi(ξ,n,0,0)
    φn = jacobi(ξ,n,0,0)
    φm = jacobi(η,m,k,0)
    Dφm = djacobi(η,m,k,0)
    Px=2/(1-η)*Dφn*φm*((1-η)/2)^n
    N=sqrt(2/((2*n+1)*(m+n+1)))
    Py=(2*x/(1-y)^2*Dφn*((1-η)/2)^n-n*((1-η)/2)^(n-1)*φn)*φm + 2*φn*((1-η)/2)^n*Dφm
    return [(4*Px)/N,(2*Py)/N]
end

"""
dubiner_basis(x,y,j::Integer)
Evaluate the dubiner basis `j` at point (x,y)
on the reference triangle ((0,0),(1,0),(0,1))
"""
function dubiner_basis(x,y,j::Integer)
    #Compute degrees
    t=-3/2+(1/2)*sqrt(1+8*j)
    n=((ceil(t)+1)*(ceil(t)+2))/2-j
    m=ceil(t)-n
    #Compute Dubiner_nm(ξ, η)
    return dubiner(x,y,Int(n),Int(m))
end

"""
∇dubiner_basis(x,y,j::Integer)
Compute the gradient of dubiner basis `j` at point (x,y)
on the reference triangle ((0,0),(1,0),(0,1))
"""
function ∇dubiner_basis(x,y,j::Integer)
    #Compute degrees
    t=-3/2+(1/2)*sqrt(1+8*j)
    n=Int(((ceil(t)+1)*(ceil(t)+2))/2-j)
    m=Int(ceil(t)-n)
    #Compute ∇Dubiner_nm(ξ, η)
    return ∇dubiner(x,y,n,m)
end
