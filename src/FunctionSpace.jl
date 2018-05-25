#ScalarFunctionSpace
struct ScalarFunctionSpace{dim,T<:Real,refshape<:AbstractRefShape,M} <: DiscreteFunctionSpace{dim,T,refshape}
    N::Matrix{T}
    dNdξ::Matrix{Vec{dim,T}}
    detJdV::Matrix{T}
    Jinv::Matrix{Tensor{2,dim,T,M}}
    M::Matrix{T}
    qr_weights::Vector{T}
end

getnquadpoints(fs::ScalarFunctionSpace) = length(fs.qr_weights)
getdetJdV(fs::ScalarFunctionSpace, cell::Int, q_point::Int) = fs.detJdV[cell,q_point]
@inline shape_value(fs::ScalarFunctionSpace, q_point::Int, base_func::Int) = fs.N[base_func, q_point]

function ScalarFunctionSpace(mesh::PolygonalMesh, func_interpol::Interpolation{dim,shape,order},
    quad_degree = order+1,geom_interpol::Interpolation=get_default_geom_interpolator(dim, shape)) where {dim, shape, order}
    quad_rule = QuadratureRule{dim,shape}(DefaultQuad(), quad_degree)
    ScalarFunctionSpace(Float64, mesh, quad_rule, func_interpol, geom_interpol)
end

function ScalarFunctionSpace(::Type{T}, mesh::PolygonalMesh, quad_rule::QuadratureRule{dim,shape}, func_interpol::Interpolation,
        geom_interpol::Interpolation=get_default_geom_interpolator(dim, shape)) where {dim,T,shape<:AbstractRefShape}

    @assert getdim(func_interpol) == getdim(geom_interpol)
    @assert getrefshape(func_interpol) == getrefshape(geom_interpol) == shape
    n_qpoints = length(getweights(quad_rule))
    n_cells = numcells(mesh)

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_interpol)
    N    = fill(zero(T)          * T(NaN), n_func_basefuncs, n_qpoints)
    dNdξ = fill(zero(Vec{dim,T}) * T(NaN), n_func_basefuncs, n_qpoints)

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(geom_interpol)
    M    = fill(zero(T)          * T(NaN), n_geom_basefuncs, n_qpoints)
    dMdξ = fill(zero(Vec{dim,T}) * T(NaN), n_geom_basefuncs, n_qpoints)

    for (qp, ξ) in enumerate(quad_rule.points)
        for i in 1:n_func_basefuncs
            dNdξ[i, qp] = gradient_value(func_interpol, i, ξ)
            N[i, qp] = value(func_interpol, i, ξ)
        end
        for i in 1:n_geom_basefuncs
            dMdξ[i, qp], M[i, qp] = gradient(ξ -> value(geom_interpol, i, ξ), ξ, :all)
        end
    end

    detJdV = fill(T(NaN), n_cells, n_qpoints)
    Jinv = fill(zero(Tensor{2,dim,T}) * T(NaN), n_cells, n_qpoints)
    #Precompute detJdV and invJ
    weights = getweights(quad_rule)
    for (k,cell) in enumerate(get_cells(mesh))
        x = get_coordinates(cell, mesh)
        @inbounds for i in 1:n_qpoints
            w = weights[i]
            fecv_J = zero(Tensor{2,dim})
            for j in 1:n_geom_basefuncs
                fecv_J += x[j] ⊗ dMdξ[j, i]
            end
            detJ = det(fecv_J)
            detJ > 0.0 || throw(ArgumentError("det(J) is not positive: det(J) = $(detJ)"))
            detJdV[k, i] = detJ * w
            Jinv[k,i] = inv(fecv_J)
        end
    end
    MM = Tensors.n_components(Tensors.get_base(eltype(Jinv)))
    ScalarFunctionSpace{dim,T,shape,MM}(N, dNdξ, detJdV, Jinv, M, quad_rule.weights)
end

# VectorFunctionSpace
struct VectorFunctionSpace{dim,T<:Real,refshape<:AbstractRefShape,M,NN <:Int} <: DiscreteFunctionSpace{dim,T,refshape}
    n_dof::NN
    ssp::ScalarFunctionSpace{dim,T,refshape,M}
end

getnquadpoints(fs::VectorFunctionSpace) = length(fs.ssp.qr_weights)
getdetJdV(fs::VectorFunctionSpace, cell::Int, q_point::Int) = fs.ssp.detJdV[cell,q_point]
function shape_value(fs::VectorFunctionSpace{dim,T}, q_point::Int, base_func::Int) where {dim,T}
    @assert 1 <= base_func <= fs.n_dof "invalid base function index: $base_func"
    N_comp = zeros(T, dim)
    n = size(fs.ssp.N,1)
    N_comp[div(base_func,n+1)+1] = fs.ssp.N[mod1(base_func,n),q_point]
    return N_comp
end

function VectorFunctionSpace(mesh::PolygonalMesh, func_interpol::Interpolation{dim,shape,order},
    quad_degree = order+1,geom_interpol::Interpolation=get_default_geom_interpolator(dim, shape)) where {dim, shape, order}
    quad_rule = QuadratureRule{dim,shape}(DefaultQuad(), quad_degree)
    ssp = ScalarFunctionSpace(Float64, mesh, quad_rule, func_interpol, geom_interpol)
    n_func_basefuncs = getnbasefunctions(func_interpol)
    dof = n_func_basefuncs*dim
    VectorFunctionSpace(dof,ssp)
end
