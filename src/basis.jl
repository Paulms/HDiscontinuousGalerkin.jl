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

# Default to nodal interpolator for geom
function get_default_geom_interpolator(dim, shape)
    @assert 1 <= dim <= 2 "No default interpolator available for $shape of dimension $dim"
    if dim == 2
        return Lagrange{dim,shape,1}()
    end
    return Legendre{dim,shape,1}()
end
############
# Dubiner #
###########
struct Dubiner{dim,shape,order} <: Interpolation{dim,shape,order} end

getnbasefunctions(::Dubiner{2,RefTetrahedron,order}) where {order} = Int((order+1)*(order+2)/2)
#nvertexdofs(::Dubiner{2,RefTetrahedron,order}) = 1

#vertices(::Dubiner{2,RefTetrahedron,order}) where {order} = (1,2,3)
#faces(::Dubiner{2,RefTetrahedron,order}) where {order} = ((1,2), (2,3), (3,1))

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
    if j == 0; return Vec{2}([zero(T),zero(T)])
    elseif j == 1; return Vec{2}([zero(T),zero(T)])
    elseif j == 2; return Vec{2}([4*sqrt(3), 2*sqrt(3)]*one(T))
    elseif j == 3; return Vec{2}([zero(T), one(T)*6])
    elseif j == 4; return Vec{2}([12*sqrt(30)*r + 6*sqrt(30)*(s - 1),
                            6*sqrt(30)*r + 2*sqrt(30)*s - 2*sqrt(30)])
    elseif j == 5; return Vec{2}([6*sqrt(2)*(5*s - 1),
                            30*sqrt(2)*r + 30*sqrt(2)*s - 18*sqrt(2)])
    elseif j == 6; return Vec{2}([zero(T),20*sqrt(6)*s - 8*sqrt(6)])
    elseif j == 7; return Vec{2}([24*sqrt(14)*(5*r^2 + 5*r*(s - 1) + s^2 - 2*s + 1),
                            60*sqrt(14)*r^2 + r*(48*sqrt(14)*s - 48*sqrt(14)) + 6*sqrt(14)*s^2 - 12*sqrt(14)*s + 6*sqrt(14)])
    elseif j == 8; return Vec{2}([12*sqrt(10)*(7*s - 1)*(2*r + s - 1),
                            84*sqrt(10)*r^2 + r*(168*sqrt(10)*s - 96*sqrt(10)) + 42*sqrt(10)*s^2 - 60*sqrt(10)*s + 18*sqrt(10)])
    elseif j == 9; return Vec{2}([4*sqrt(6)*(21*s^2 - 12*s + 1),
                            r*(168*sqrt(6)*s - 48*sqrt(6)) + 126*sqrt(6)*s^2 - 132*sqrt(6)*s + 26*sqrt(6)])
    elseif j == 10; return Vec{2}([zero(T),210*sqrt(2)*s^2 - 180*sqrt(2)*s + 30*sqrt(2)])
    elseif j == 11; return Vec{2}([840*sqrt(10)*r^3 + 1260*sqrt(10)*r^2*(s - 1) + 540*sqrt(10)*r*(s^2 - 2*s + 1) + 60*sqrt(10)*(s^3 - 3*s^2 + 3*s - 1),
                            420*sqrt(10)*r^3 + r^2*(540*sqrt(10)*s - 540*sqrt(10)) + r*(180*sqrt(10)*s^2 - 360*sqrt(10)*s + 180*sqrt(10)) + 12*sqrt(10)*s^3 - 36*sqrt(10)*s^2 + 36*sqrt(10)*s - 12*sqrt(10)])
    elseif j == 12; return Vec{2}([12*sqrt(70)*(9*s - 1)*(5*r^2 + 5*r*(s - 1) + s^2 - 2*s + 1),
                            180*sqrt(70)*r^3 + r^2*(540*sqrt(70)*s - 300*sqrt(70)) + r*(324*sqrt(70)*s^2 - 456*sqrt(70)*s + 132*sqrt(70)) + 36*sqrt(70)*s^3 - 84*sqrt(70)*s^2 + 60*sqrt(70)*s - 12*sqrt(70)])
    elseif j == 13; return Vec{2}([30*sqrt(2)*(36*s^2 - 16*s + 1)*(2*r + s - 1),
                            r^2*(2160*sqrt(2)*s - 480*sqrt(2)) + r*(3240*sqrt(2)*s^2 - 3120*sqrt(2)*s + 510*sqrt(2)) + 720*sqrt(2)*s^3 - 1320*sqrt(2)*s^2 + 690*sqrt(2)*s - 90*sqrt(2)])
    elseif j == 14; return Vec{2}([2*sqrt(30)*(84*s^3 - 84*s^2 + 21*s - 1),
                            r*(504*sqrt(30)*s^2 - 336*sqrt(30)*s + 42*sqrt(30)) + 336*sqrt(30)*s^3 - 504*sqrt(30)*s^2 + 210*sqrt(30)*s - 22*sqrt(30)])
    elseif j == 15; return Vec{2}([zero(T),504*sqrt(10)*s^3 - 672*sqrt(10)*s^2 + 252*sqrt(10)*s - 24*sqrt(10)])
    else; return ∇dubiner_basis(r,s,j)
    end
end

"""
    jacobi(x, p::Integer, α, β)
Evaluate the Legendre polynomial with parameters `α`, `β` of degree `p` at `x`
using the three term recursion [Karniadakis and Sherwin, Spectral/hp Element
Methods for CFD, Appendix A].
Author: H. Ranocha (see PolynomialBases.jl)
"""
function jacobi(x, p::Integer, α, β)
    T = typeof( (2+α+β)*x / 2 )
    a = one(T)
    b = ((2+α+β)*x + α - β) / 2
    if p <= 0
        return a
    elseif p == 1
        return b
    end

    for n in 2:p
        a1 = 2n*(n+α+β)*(2n-2+α+β)
        a2 = (2n-1+α+β)*(α+β)*(α-β)
        a3 = (2n-2+α+β)*(2n-1+α+β)*(2n+α+β)
        a4 = 2*(n-1+α)*(n-1+β)*(2n+α+β)
        a, b = b, ( (a2+a3*x)*b - a4*a ) / a1
    end
    b
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
#TODO: It could be better to return a tensor
function ∇dubiner_basis(x,y,j::Integer)
    #Compute degrees
    t=-3/2+(1/2)*sqrt(1+8*j)
    n=Int(((ceil(t)+1)*(ceil(t)+2))/2-j)
    m=Int(ceil(t)-n)
    #Compute ∇Dubiner_nm(ξ, η)
    return Vec{2}(∇dubiner(x,y,n,m))
end

####################
# Lagrange
####################
struct Lagrange{dim,shape,order,T} <: Interpolation{dim,shape,order}
    nodal_base_coefs::T
end

function Lagrange{1,shape,order}() where {shape, order}
    nodals=[x->x(point) for point in get_nodal_points(shape(), Val{1}, order)]
    ip_prime = Legendre{1,shape,order}()
    nbasefuncs = getnbasefunctions(ip_prime)
    prime_base = [x->value(ip_prime, j, x) for j in 1:nbasefuncs]
    V = reshape([nodals[i](prime_base[j]) for j = 1:nbasefuncs for i=1:nbasefuncs],(nbasefuncs,nbasefuncs))
    nodal_base_coefs = inv(V)
    Lagrange{1, shape, order, typeof(nodal_base_coefs)}(nodal_base_coefs)
end

function Lagrange{2,shape,order}() where {shape, order}
    nodals=[x->x(point) for point in get_nodal_points(shape(), Val{2}, order)]
    ip_prime = Dubiner{2,shape,order}()
    nbasefuncs = getnbasefunctions(ip_prime)
    prime_base = [x->value(ip_prime, j, x) for j in 1:nbasefuncs]
    V = reshape([nodals[i](prime_base[j]) for j = 1:nbasefuncs for i=1:nbasefuncs],(nbasefuncs,nbasefuncs))
    nodal_base_coefs = inv(V)
    Lagrange{2, shape, order, typeof(nodal_base_coefs)}(nodal_base_coefs)
end

getnbasefunctions(::Lagrange{1,RefTetrahedron,order}) where {order} = order + 1
getnbasefunctions(::Lagrange{2,RefTetrahedron,order}) where {order} = Int((order+1)*(order+2)/2)

"""
value(ip::Lagrange{2,RefTetrahedron,order}, j::Int, ξ::AbstactVector) where {order}
Compute value of Lagrange basis `j` at point ξ
on the reference triangle ((0,0),(1,0),(0,1))
"""
function value(ip::Lagrange{2,RefTetrahedron,order}, k::Int, ξ::Vec{2,T}) where {order, T}
    if k > getnbasefunctions(ip);throw(ArgumentError("no shape function $k for interpolation $ip"));end
    n = getnbasefunctions(ip)
    dot(ip.nodal_base_coefs[:,k], value(Dubiner{2,RefTetrahedron,order}(), ξ))
end

function value(ip::Lagrange{1,RefTetrahedron,order}, k::Int, ξ::Vec{1,T}) where {order, T}
    if k > getnbasefunctions(ip);throw(ArgumentError("no shape function $k for interpolation $ip"));end
    n = getnbasefunctions(ip)
    dot(ip.nodal_base_coefs[:,k], value(Legendre{1,RefTetrahedron,order}(), ξ))
end

"""
gradient_value(ip::Lagrange{2,RefTetrahedron,order}, j::Int, ξ::AbstactVector) where {order}
Compute value of Lagrange basis `j` derivative at point ξ
on the reference triangle ((0,0),(1,0),(0,1))
"""
function gradient_value(ip::Lagrange{dim,RefTetrahedron,order}, k::Int, ξ::Vec{2,T}) where {dim,order,T}
    if k >getnbasefunctions(ip);throw(ArgumentError("no shape function $k for interpolation $ip"));end
    # a = nodal_basis_coefs[1,k]*gradient_value(interpolation, 1, ξ)
    # n = getnbasefunctions(ip)
    # for j in 2:n
    #     a += nodal_basis_coefs[j,k]*gradient_value(interpolation, j, ξ)
    # end
    # return a
    gradient(ξ -> value(ip, k, ξ), ξ)
end

####################
# Legendre
####################
struct Legendre{dim,shape,order} <: Interpolation{dim,shape,order} end

getnbasefunctions(::Legendre{1,RefTetrahedron,order}) where {order} = order + 1

"""
value(ip::Legendre{1,RefTetrahedron,order}, j::Int, ξ::AbstactVector) where {order}
Compute value of Legendre basis `j` at point ξ
on the reference line (-1,1)
"""
function value(ip::Legendre{1,RefTetrahedron,order}, k::Int, ξ::Vec{1,T}) where {order, T}
    if k > getnbasefunctions(ip);throw(ArgumentError("no shape function $k for interpolation $ip"));end
    return sqrt((2*k+1)/2)*jacobi(ξ[1],k,0.0,0.0)
end

"""
gradient_value(ip::Legendre{1,RefTetrahedron,order}, j::Int, ξ::AbstactVector) where {order}
Compute value of Legendre basis `j` derivative at point ξ
on the reference line (-1,1)
"""
function gradient_value(ip::Legendre{1,RefTetrahedron,order}, k::Int, ξ::Vec{1,T}) where {order, T}
    if k >getnbasefunctions(ip);throw(ArgumentError("no shape function $k for interpolation $ip"));end
    # a = 0; b = 1;
    # if k==0
    #     return a
    # elseif k==1
    #     return b
    # else
    #     for n=2:k
    #         J =((n+1)/2)* jacobi(ξ,k-1,1,1) ;
    #     end
    # end
    # return 2*sqrt(2*k+1)*J
    gradient(ξ -> value(ip, k, ξ), ξ)[1]
end
