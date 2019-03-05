####################
# Continuous Lagrange
####################

struct ContinuousLagrange{dim,shape,qorder,gorder} <: FiniteElement{dim,shape,qorder,gorder}
    nodal_base_coefs::Matrix{Float64}
    geom_base_coefs::Matrix{Float64}
    topology::Dict{Int,Int}
    geom_topology::Dict{Int,Int}
    M::Matrix{Float64}
end

function ContinuousLagrange{dim,shape,qorder}() where {dim,shape<:AbstractRefShape, qorder}
    return ContinuousLagrange{dim, shape, qorder,1}()
end

function ContinuousLagrange{dim,shape,qorder,gorder}() where {dim,shape<:AbstractRefShape, qorder, gorder}
    nodal_base_coefs, topology = _nodal_data(dim, shape, qorder)
    if qorder == gorder
        geom_base_coefs = nodal_base_coefs
        geom_topology = topology
    else
        geom_base_coefs, geom_topology = _nodal_data(dim, shape, gorder)
    end
    if gorder == 1
        M = one(geom_base_coefs)
    else
        M = _nodal_geom_data(dim, shape, gorder, size(geom_base_coefs,1), geom_base_coefs)
    end
    ContinuousLagrange{dim, shape, qorder, gorder}(nodal_base_coefs, geom_base_coefs, topology, geom_topology, M)
end

function _nodal_data(dim::Int,shape::Type{Shape}, order::Int) where {Shape <: AbstractRefShape}
    nodal_points, topology = get_nodal_points(shape, Val{dim}, order)
    ip_prime = getdefaultdualbasis(dim,shape,order)
    nbasefuncs = getnbasefunctions(ip_prime)
    V = [value(ip_prime, j, nodal_points[i]) for i=1:nbasefuncs,j = 1:nbasefuncs]
    return inv(V), topology
end

function _nodal_geom_data(dim::Int,shape::Type{Shape}, order::Int, n_geom_basefuncs::Int, geom_base_coefs::Matrix) where {Shape <: AbstractRefShape}
    # Matrix to get spacial coordinates
    nodal_points, topology = get_nodal_points(shape, Val{dim}, order)
    T = eltype(nodal_points[1])
    qrs = QuadratureRule{dim,shape,T}(fill(T(NaN), length(nodal_points)), nodal_points) # weights will not be used
    n_qpoints = length(getweights(qrs))
    M =    fill(zero(T)           * T(NaN), n_geom_basefuncs, n_qpoints)
    for (qp, ξ) in enumerate(qrs.points)
        for i in 1:n_geom_basefuncs
            M[i, qp] = dot(geom_base_coefs[:,i], value(defaultDualBasis(dim,shape,order), ξ))
        end
    end
    M
end

@inline getnbasefunctions(::ContinuousLagrange{1,RefTetrahedron,order}) where {order} = order + 1
@inline getnbasefunctions(::ContinuousLagrange{2,RefTetrahedron,order}) where {order} = Int((order+1)*(order+2)/2)
@inline getngeombasefunctions(::ContinuousLagrange{1,RefTetrahedron,order,gorder}) where {order,gorder} = gorder + 1
@inline getngeombasefunctions(::ContinuousLagrange{2,RefTetrahedron,order,gorder}) where {order,gorder} = Int((gorder+1)*(gorder+2)/2)
@inline gettopology(ip::ContinuousLagrange) = ip.topology
@inline getgeomtopology(ip::ContinuousLagrange) = ip.geom_topology

"""
value(ip::ContinuousLagrange{dim,shape<:AbstractRefShape,order}, k::Int, ξ::Vec{dim,T}) where {dim,shape,order, T}
Compute value of Continuous Lagrange Finite Element basis `j` at point ξ on shape `shape`
"""
function value(ip::ContinuousLagrange{dim,shape,order}, k::Int, ξ::Vec{dim,T}) where {dim,shape<:AbstractRefShape,order, T}
    if k > getnbasefunctions(ip);throw(ArgumentError("no shape function $k for interpolation $ip"));end
    n = getnbasefunctions(ip)
    dot(ip.nodal_base_coefs[:,k], value(defaultDualBasis(dim,shape,order), ξ))
end

"""
geom_value(ip::ContinuousLagrange{dim,shape<:AbstractRefShape,order}, k::Int, ξ::Vec{dim,T}) where {dim,shape,order, T}
Compute value of Continuous Lagrange Finite Element basis `j` at point ξ on shape `shape`
"""
function geom_value(ip::ContinuousLagrange{dim,shape,order,gorder}, k::Int, ξ::Vec{dim,T}) where {dim,shape<:AbstractRefShape,order,gorder, T}
    if k > getnbasefunctions(ip);throw(ArgumentError("no shape function $k for interpolation $ip"));end
    n = getnbasefunctions(ip)
    dot(ip.geom_base_coefs[:,k], value(defaultDualBasis(dim,shape,gorder), ξ))
end
