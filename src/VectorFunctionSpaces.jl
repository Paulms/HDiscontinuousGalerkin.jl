abstract type AbstractVectorFunctionSpace{dim,T,FE<:FiniteElement,MM,N1,N2,N3,fdim} <: DiscreteFunctionSpace{dim,T,FE} end

# VectorFunctionSpace
struct VectorFunctionSpace{dim,T<:Real,FE<:FiniteElement,MM,N1,N2,N3,fdim} <: AbstractVectorFunctionSpace{dim,T,FE,MM,N1,N2,N3,fdim}
    n_dof::Int
    ssp::ScalarFunctionSpace{dim,T,FE,MM,N1,N2,N3,fdim}
end

#Constructor
function VectorFunctionSpace(mesh::PolygonalMesh, felem::FiniteElement{dim,shape,order,1};
    update_face_data = true, quad_degree = order+1) where {dim, shape, order}
    quad_rule = QuadratureRule{dim,shape}(DefaultQuad(), quad_degree)
    f_quad_rule = QuadratureRule{dim-1,shape}(DefaultQuad(), quad_degree)
    fs = _scalar_fs(Float64, mesh, quad_rule, f_quad_rule, felem, update_face_data)
    n_func_basefuncs = getnbasefunctions(felem)
    dof = n_func_basefuncs*dim
    return VectorFunctionSpace(dof, fs)
end

#Data
@inline getnquadpoints(fs::AbstractVectorFunctionSpace) = length(fs.ssp.qr_weights)
@inline getdetJdV(fs::AbstractVectorFunctionSpace, q_point::Int) = getdetJdV(fs.ssp, q_point)
@inline getnbasefunctions(fs::AbstractVectorFunctionSpace) = fs.n_dof
@inline getnlocaldofs(fs::AbstractVectorFunctionSpace) = fs.n_dof
@inline getfiniteelement(fs::AbstractVectorFunctionSpace) = fs.ssp.fe
@inline getmesh(fs::AbstractVectorFunctionSpace) = fs.ssp.mesh

function shape_value(fs::AbstractVectorFunctionSpace{dim,T}, q_point::Int, base_func::Int) where {dim,T}
    @assert 1 <= base_func <= fs.n_dof "invalid base function index: $base_func"
    N_comp = zeros(T, dim)
    n = size(fs.ssp.N,1)
    N_comp[div(base_func,n+1)+1] = fs.ssp.N[mod1(base_func,n),q_point]
    return N_comp
end

function shape_gradient(fs::VectorFunctionSpace{dim,T}, q_point::Int, base_func::Int)  where {dim,T}
    @assert 1 <= base_func <= fs.n_dof "invalid base function index: $base_func"
    dN_comp = zeros(T, dim, dim)
    n = size(fs.ssp.N,1)
    dN_comp[div(base_func,n+1)+1, :] = fs.ssp.dNdξ[mod1(base_func,n), q_point]
    return Tensor{2,dim,T}((dN_comp...,)) ⋅ fs.ssp.Jinv[]
end

function shape_divergence(fs::AbstractVectorFunctionSpace{dim,T}, q_point::Int, base_func::Int)  where {dim,T}
    @assert 1 <= base_func <= fs.n_dof "invalid base function index: $base_func"
    return tr(shape_gradient(fs, q_point, base_func))
end

#Face Data
function shape_value(fs::AbstractVectorFunctionSpace{dim,T}, face::Int, q_point::Int, base_func::Int, orientation::Bool=true) where {dim,T}
    @assert 1 <= base_func <= size(fs.ssp.E,1)*dim "invalid base function index: $base_func"
    N_comp = zeros(T, dim)
    n = size(fs.ssp.E,1)
    q_p = orientation ? q_point : getnfacequadpoints(fs.ssp)-q_point + 1
    N_comp[div(base_func,n+1)+1] = fs.ssp.E[mod1(base_func,n), q_p][face]
    return N_comp
end

@inline get_normal(fs::AbstractVectorFunctionSpace, face::Int) = get_normal(fs.ssp, face)

function reinit!(fs::AbstractVectorFunctionSpace{dim}, x::AbstractVector{Vec{dim,T}}) where {dim,T}
    reinit!(fs.ssp, x)
end
