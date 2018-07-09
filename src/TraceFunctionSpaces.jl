abstract type AbstractScalarTraceFunctionSpace{dim,T,fdim,shape,order,N1,N2,N3} <: DiscreteFunctionSpace{dim,T,shape} end
# Scalar Trace Function Space (Scalar functions defined only on cell boundaries)
struct ScalarTraceFunctionSpace{dim,T<:Real,fdim,shape<:AbstractRefShape, order,N1,N2,N3} <: AbstractScalarTraceFunctionSpace{dim,T,fdim,shape,order,N1,N2,N3}
    N::Matrix{T}
    L::Array{T,3}
    dNdξ::Matrix{Vec{fdim,T}}
    detJ::Matrix{T}
    qr_weights::Vector{T}
    mesh::PolygonalMesh{dim,N1,N2,N3,T}
end

#Constructor
function ScalarTraceFunctionSpace(mesh::PolygonalMesh, psp::FacesFunctionSpace{dim,fdim,T,1},
    func_interpol::Interpolation{fdim,shape,order}) where {dim,fdim,shape,order,T}
    detJ = psp.detJf
    qr_weights = psp.qr_face_weigths
    qr_points = psp.qr_face_points
    ScalarTraceFunctionSpace(func_interpol, detJ, qr_weights, qr_points, psp.L, mesh)
end

function ScalarTraceFunctionSpace(func_interpol::Interpolation{dim,refshape,order},
    detJ::Matrix{T}, qr_weights::Vector, qr_points::Vector, L::Array{T,3}, mesh::PolygonalMesh{dim2,N1,N2,N3,T}) where {dim, refshape, order, T,N1,N2,N3,dim2}
    n_qpoints = length(qr_weights)

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_interpol)
    N    = fill(zero(T)          * T(NaN), n_func_basefuncs, n_qpoints)
    dNdξ = fill(zero(Vec{dim,T}) * T(NaN), n_func_basefuncs, n_qpoints)

    for (qp, ξ) in enumerate(qr_points)
        for i in 1:n_func_basefuncs
            dNdξ[i, qp], N[i, qp]  = gradient(ξ -> value(func_interpol, i, ξ), ξ, :all)
        end
    end
    ScalarTraceFunctionSpace{dim2,T,dim,refshape,order,N1,N2,N3}(N, L, dNdξ,detJ, qr_weights, mesh)
end

#Data
@inline getnbasefunctions(fs::ScalarTraceFunctionSpace) = size(fs.N,1)
@inline getnquadpoints(fs::ScalarTraceFunctionSpace) = length(fs.qr_weights)
#@inline getdetJdS(fs::ScalarTraceFunctionSpace{dim,T,2}, cell::Int, face::Int, q_point::Int) where {dim,T} = fs.detJ[cell,face,q_point]*fs.qr_weights[q_point]
@inline getdetJdS(fs::ScalarTraceFunctionSpace, cell::Int, face::Int, q_point::Int) = fs.detJ[cell,face]*fs.qr_weights[q_point]
@inline shape_value(fs::ScalarTraceFunctionSpace, q_point::Int, base_func::Int) = fs.N[base_func, q_point]
@inline getngeobasefunctions(fs::ScalarTraceFunctionSpace) = size(fs.L,1)
@inline geometric_value(fs::ScalarTraceFunctionSpace, face::Int, q_point::Int, base_func::Int) = fs.L[base_func, q_point, face]
@inline getnlocaldofs(fs::ScalarTraceFunctionSpace) = getnbasefunctions(fs)*size(fs.L,3)
@inline getmesh(fs::ScalarTraceFunctionSpace) = fs.mesh

"""
function spatial_coordinate(fs::ScalarTraceFunctionSpace{dim}, q_point::Int, x::AbstractVector{Vec{dim,T}}, orientation=true)
Map coordinates of quadrature point `q_point` of Scalar Trace Function Space `fs`
into domain with vertices `x`
"""
function spatial_coordinate(fs::ScalarTraceFunctionSpace{dim}, face::Int, q_point::Int, x::AbstractVector{Vec{dim,T}}, orientation=true) where {dim,T}
    n_base_funcs = getngeobasefunctions(fs)
    @assert length(x) == n_base_funcs
    vec = zero(Vec{dim,T})
    n = getnquadpoints(fs)
    @inbounds for i in 1:n_base_funcs
        or_q_point = orientation ? q_point : n - q_point + 1
        vec += geometric_value(fs, face, or_q_point, i) * x[i]
    end
    return vec
end
