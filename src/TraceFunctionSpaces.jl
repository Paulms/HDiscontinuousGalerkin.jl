abstract type AbstractScalarTraceFunctionSpace{dim,T,FE,fdim} <: DiscreteFunctionSpace{dim,T,FE} end
# Scalar Trace Function Space (Scalar functions defined only on cell boundaries)
struct ScalarTraceFunctionSpace{dim,T<:Real,FE<:FiniteElement,fdim, FE2 <: FiniteElement,MM,N1,N2,N3} <: AbstractScalarTraceFunctionSpace{dim,T,FE,fdim}
    N::Matrix{T}
    dNdξ::Matrix{Vec{fdim,T}}
    fe::FE
    fs::ScalarFunctionSpace{dim,T, FE2, MM,N1,N2,N3,fdim}
end

#Constructor
function ScalarTraceFunctionSpace(psp::ScalarFunctionSpace{dim,T, FE, MM,N1,N2,N3,fdim},
    felem::FiniteElement{fdim,shape,order,gorder}) where {dim, T, FE, MM, N1,N2,N3,fdim,shape,order,gorder}
    qr_points = psp.qr_face_points
    qr_weights = psp.qr_face_weigths
    n_qpoints = length(qr_weights)

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(felem)
    N    = fill(zero(T)          * T(NaN), n_func_basefuncs, n_qpoints)
    dNdξ = fill(zero(Vec{fdim,T}) * T(NaN), n_func_basefuncs, n_qpoints)

    for (qp, ξ) in enumerate(qr_points)
        for i in 1:n_func_basefuncs
            dNdξ[i, qp], N[i, qp]  = gradient(ξ -> value(felem, i, ξ), ξ, :all)
        end
    end
    ScalarTraceFunctionSpace{dim,T,typeof(felem),fdim, FE, MM,N1,N2,N3}(N, dNdξ,felem, psp)
end

#Data
@inline getnbasefunctions(fs::ScalarTraceFunctionSpace) = size(fs.N,1)
@inline getnquadpoints(fs::ScalarTraceFunctionSpace) = length(fs.fs.qr_face_weigths)
#@inline getdetJdS(fs::ScalarTraceFunctionSpace{dim,T,2}, cell::Int, face::Int, q_point::Int) where {dim,T} = fs.detJ[cell,face,q_point]*fs.qr_weights[q_point]
@inline getfacedetJdS(fs::ScalarTraceFunctionSpace, face::Int, q_point::Int) = getfacedetJdS(fs.fs, face, q_point)
@inline shape_value(fs::ScalarTraceFunctionSpace, q_point::Int, base_func::Int) = fs.N[base_func, q_point]
@inline getngeobasefunctions(fs::ScalarTraceFunctionSpace) = size(fs.fs.L,1)
@inline geometric_value(fs::ScalarTraceFunctionSpace, face::Int, q_point::Int, base_func::Int) = fs.fs.L[base_func, q_point, face]
@inline getnlocaldofs(fs::ScalarTraceFunctionSpace) = getnbasefunctions(fs)*size(fs.fs.L,3)
@inline getmesh(fs::ScalarTraceFunctionSpace) = getmesh(fs.fs)
@inline getfiniteelement(fs::ScalarTraceFunctionSpace) = fs.fe

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
