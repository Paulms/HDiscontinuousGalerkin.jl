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
    if order <= 6
        return QuadratureRule{2,RefTetrahedron}(Strang(),order)
    elseif order > 6 && isodd(order)
        s = Int((order-1)/2)
        return QuadratureRule{2,RefTetrahedron}(GrundmannMoeller(),s)
    else
        throw(ArgumentError("Quadrature rule of order $order not available"))
    end
end

# Get GaussLegendre weigths and points from FastGaussQuadrature
function (::Type{QuadratureRule{1,RefTetrahedron}})(quad_type::GaussLegendre, order::Int)
    points, weights = gausslegendre(order)
    # Shift interval from (-1,1) to (0,1)
    weights *= 0.5
    points = points .+ 1.0; points /= 2.0
    return QuadratureRule{1,RefTetrahedron,Float64}(weights, [Tensor{1,1}([x]) for x in points])
end

function (::Type{QuadratureRule{1,RefTetrahedron}})(quad_type::DefaultQuad, order::Int)
    return QuadratureRule{1,RefTetrahedron}(GaussLegendre(), order)
end

"""
Face Quadrature Rules
"""
function create_face_quad_rule(quad_rule::QuadratureRule{1,shape,T}, itp::Interpolation{2,shape}) where {T,shape}
    w = getweights(quad_rule)
    p = getpoints(quad_rule)
    geom_face_interpol = get_default_geom_interpolator(shape, Val{1})
    face_quad_rule = QuadratureRule{2,shape,T}[]
    _populate_face_quad_rule!(face_quad_rule, geom_face_interpol, p, w)
end

function create_face_quad_rule(quad_rule::QuadratureRule{1,shape,T}) where {T,shape}
    w = getweights(quad_rule)
    p = getpoints(quad_rule)
    geom_face_interpol = get_default_geom_interpolator(shape, Val{1})
    face_quad_rule = QuadratureRule{2,shape,T}[]
    _populate_face_quad_rule!(face_quad_rule, geom_face_interpol, p, w)
end

function _populate_face_quad_rule!(face_quad_rule::Vector{QuadratureRule{2,shape,T}}, geom_face_interpol::Interpolation{1,shape,order}, p::Vector{Vec{1,T}}, w::Vector{T}) where {shape, order, T}
    face_coords = reference_edges(shape, Val{2})
    for j = 1:get_num_faces(shape, Val{2})
        new_points = Vec{2,T}[]
        for (qp, ξ) in enumerate(p)
            #Map from reference dim-1 shape to reference dim shape face/edge
            η = zero(Vec{2,T})
            for (k,x) in enumerate(face_coords[j])
                η += value(geom_face_interpol, k, ξ)*x
            end
            push!(new_points, η)
        end
        push!(face_quad_rule, QuadratureRule{2,shape,T}(w, new_points))
    end
    return face_quad_rule
end

#OTher Functions
"""
function integrate(f::function,qr::QuadratureRule)
integrate function f on reference shape of quadrature qr
"""
function integrate(f::Function,qr::QuadratureRule)
    p = getpoints(qr); w = getweights(qr)
    int_val = zero(typeof(f(p[1])))
    for (i,x) in enumerate(p)
        int_val += f(x)*w[i]
    end
    int_val
end
