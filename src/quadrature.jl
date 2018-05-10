struct QuadratureRule{dim,shape,T}
    weights::Vector{T}
    points::Vector{Vec{dim,T}}
end

getweights(qr::QuadratureRule) = qr.weights

getpoints(qr::QuadratureRule) = qr.points

#quadratures
struct GrundmannMoeller <: AbstractQuadratureRule end
