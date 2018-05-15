struct QuadratureRule{dim,shape,T}
    weights::Vector{T}
    points::Vector{Vec{dim,T}}
end

getweights(qr::QuadratureRule) = qr.weights

getpoints(qr::QuadratureRule) = qr.points

#quadratures
struct GrundmannMoeller <: AbstractQuadratureRule end
struct Strang <: AbstractQuadratureRule end
struct DefaultQuad <: AbstractQuadratureRule end
struct GaussLegendre <: AbstractQuadratureRule end

# utils
function (::Type{QuadratureRule{2,RefTetrahedron}})(quad_type::DefaultQuad, order::Int)
    if order <= 5
        return QuadratureRule{2,RefTetrahedron}(Strang(),order)
    elseif order > 5 && isodd(order)
        s = Int((order-1)/2)
        return QuadratureRule{2,RefTetrahedron}(GrundmannMoeller(),s)
    else
        throw(ArgumentError("Quadrature rule of order $order not available"))
    end
end

# Get GaussLegendre weigths and points from FastGaussQuadrature
function (::Type{QuadratureRule{1,RefTetrahedron}})(quad_type::GaussLegendre, order::Int)
    points, weights = gausslegendre(order);
    return QuadratureRule{1,RefTetrahedron,Float64}(weigths, points)
end
