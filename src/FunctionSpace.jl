# VectorFunctionSpace
struct VectorFunctionSpace{dim,T<:Real,refshape<:AbstractRefShape,M} <: DiscreteFunctionSpace{dim,T,refshape}
    N::Matrix{Vec{dim,T}}
    dNdξ::Matrix{Tensor{2,dim,T,M}}
    detJdV::Matrix{T}
    Jinv::Matrix{Tensor{2,dim,T,M}}
    M::Matrix{T}
    qr_weights::Vector{T}
end

getnquadpoints(fs::VectorFunctionSpace) = length(fs.qr_weights)
getdetJdV(fs::VectorFunctionSpace, cell::Int, q_point::Int) = fs.detJdV[cell,q_point]
@inline shape_value(fs::VectorFunctionSpace, q_point::Int, base_func::Int) = fs.N[base_func, q_point]

function VectorFunctionSpace(mesh::PolygonalMesh, func_interpol::Interpolation{dim,shape,order},
    quad_degree = order+1,geom_interpol::Interpolation=get_default_geom_interpolator(dim, shape)) where {dim, shape, order}
    quad_rule = QuadratureRule{dim,shape}(DefaultQuad(), quad_degree)
    VectorFunctionSpace(Float64, mesh, quad_rule, func_interpol, geom_interpol)
end

function VectorFunctionSpace(::Type{T}, mesh::PolygonalMesh, quad_rule::QuadratureRule{dim,shape}, func_interpol::Interpolation,
        geom_interpol::Interpolation=get_default_geom_interpolator(dim, shape)) where {dim,T,shape<:AbstractRefShape}

    @assert getdim(func_interpol) == getdim(geom_interpol)
    @assert getrefshape(func_interpol) == getrefshape(geom_interpol) == shape
    n_qpoints = length(getweights(quad_rule))
    n_cells = numcells(mesh)

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_interpol) * dim
    N    = fill(zero(Vec{dim,T})      * T(NaN), n_func_basefuncs, n_qpoints)
    dNdξ = fill(zero(Tensor{2,dim,T}) * T(NaN), n_func_basefuncs, n_qpoints)

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(geom_interpol)
    M    = fill(zero(T)          * T(NaN), n_geom_basefuncs, n_qpoints)
    dMdξ = fill(zero(Vec{dim,T}) * T(NaN), n_geom_basefuncs, n_qpoints)

    for (qp, ξ) in enumerate(quad_rule.points)
        basefunc_count = 1
        for basefunc in 1:getnbasefunctions(func_interpol)
            dNdξ_temp = gradient_value(func_interpol, basefunc, ξ)
            N_temp = value(func_interpol, basefunc, ξ)
            for comp in 1:dim
                N_comp = zeros(T, dim)
                N_comp[comp] = N_temp
                N[basefunc_count, qp] = Vec{dim,T}((N_comp...))

                dN_comp = zeros(T, dim, dim)
                dN_comp[comp, :] = dNdξ_temp
                dNdξ[basefunc_count, qp] = Tensor{2,dim,T}((dN_comp...))
                basefunc_count += 1
            end
        end
        for basefunc in 1:n_geom_basefuncs
            dMdξ[basefunc, qp], M[basefunc, qp] = gradient(ξ -> value(geom_interpol, basefunc, ξ), ξ, :all)
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
    MM = Tensors.n_components(Tensors.get_base(eltype(dNdξ)))
    VectorFunctionSpace{dim,T,shape,MM}(N, dNdξ, detJdV, Jinv, M, quad_rule.weights)
end
