abstract type FiniteElement{dim,shape,FuncOrder,GeomOrder} end

####################
# Template for Finite Elements
####################

struct GenericFiniteElement{dim,shape,qorder,gorder} <: FiniteElement{dim,shape,qorder,gorder}
    func_basis::Interpolation{dim,shape,qorder}
    geom_basis::Interpolation{dim,shape,gorder}
    topology::Dict{Int,Int}
    geom_topology::Dict{Int,Int}
    M::Matrix{Float64}
end

function GenericFiniteElement(func_basis::Interpolation{dim,shape,qorder},
    geom_basis::Interpolation{dim,shape,gorder} = get_default_geom_interpolator(shape, Val{dim})) where {dim,shape<:AbstractRefShape, qorder,gorder}
    topology = gettopology(func_basis)
    geom_topology = gettopology(geom_basis)
    n_geom_basefuncs = getnbasefunctions(geom_basis)
    if gorder == 1
        M = one(Matrix{Float64}(n_geom_basefuncs,n_geom_basefuncs))
    else
        M = _nodal_geom_data(geom_basis)
    end
    GenericFiniteElement{dim, shape, qorder, gorder}(func_basis, geom_basis, topology, geom_topology, M)
end

function _nodal_geom_data(geom_interpol::Lagrange{dim,shape,order}) where {dim,shape,order}
    # Matrix to get spacial coordinates
    nodal_points, topology = get_nodal_points(shape, Val{dim}, order)
    T = eltype(nodal_points[1])
    qrs = QuadratureRule{dim,shape,T}(fill(T(NaN), length(nodal_points)), nodal_points) # weights will not be used
    n_qpoints = length(getweights(qrs))
    n_geom_basefuncs = getnbasefunctions(geom_interpol)
    M =    fill(zero(T)           * T(NaN), n_geom_basefuncs, n_qpoints)
    for (qp, ξ) in enumerate(qrs.points)
        for i in 1:n_geom_basefuncs
            M[i, qp] = value(geom_interpol, i, ξ)
        end
    end
    M
end

@inline getnbasefunctions(fe::GenericFiniteElement) = getnbasefunctions(fe.func_basis)
@inline getngeombasefunctions(fe::GenericFiniteElement) = getnbasefunctions(fe.geom_basis)
@inline gettopology(fe::GenericFiniteElement) = fe.topology
@inline getgeomtopology(fe::GenericFiniteElement) = fe.geom_topology

"""
value(ip::ContinuousLagrange{dim,shape<:AbstractRefShape,order}, k::Int, ξ::Vec{dim,T}) where {dim,shape,order, T}
Compute value of Continuous Lagrange Finite Element basis `j` at point ξ on shape `shape`
"""
function value(ip::GenericFiniteElement{dim}, k::Int, ξ::Vec{dim,T}) where {dim, T}
    value(ip.func_basis, k, ξ)
end

"""
geom_value(ip::ContinuousLagrange{dim,shape<:AbstractRefShape,order}, k::Int, ξ::Vec{dim,T}) where {dim,shape,order, T}
Compute value of Continuous Lagrange Finite Element basis `j` at point ξ on shape `shape`
"""
function geom_value(ip::GenericFiniteElement{dim}, k::Int, ξ::Vec{dim,T}) where {dim, T}
    value(ip.geom_basis, k, ξ)
end

"""
gradient_value(ip::FiniteElement{dim,shape<:AbstractRefShape,order}, k::Int, ξ::Vec{dim,T}) where {dim,shape,order, T}
Compute value of Finite Element basis `j` derivative at point ξ
on the reference shape `shape`
"""
function gradient_value(ip::FiniteElement{dim}, k::Int, ξ::Vec{dim,T}) where {dim, T}
    gradient(ξ -> value(ip, k, ξ), ξ)
end

function spatial_nodal_coordinate(fe::FiniteElement{dim}, n_point::Int, x::AbstractVector{Vec{dim,T}}) where {dim,T}
    n_base_funcs = getngeombasefunctions(fe)
    @assert length(x) == n_base_funcs
    vec = zero(Vec{dim,T})
    @inbounds for i in 1:n_base_funcs
        vec += fe.M[i, n_point] * x[i]
    end
    return vec
end

function spatial_nodal_coordinate(fe::FiniteElement{dim,shape,1,1}, n_point::Int, x::AbstractVector{Vec{dim,T}}) where {dim,shape,T}
    @assert length(x) ==  getngeombasefunctions(fe)
    return x[n_point]
end
