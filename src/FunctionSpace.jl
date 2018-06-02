#ScalarFunctionSpace
struct ScalarFunctionSpace{dim,T<:Real,NN,refshape<:AbstractRefShape,fdim,order,M} <: DiscreteFunctionSpace{dim,T,refshape}
    N::Matrix{T}
    dNdξ::Matrix{Vec{dim,T}}
    E :: Matrix{Vector{T}}
    detJ::Matrix{T}
    detJf::Vector{Matrix{T}}
    Jinv::Matrix{Tensor{2,dim,T,M}}
    M::Matrix{T}
    qr_weights::Vector{T}
    qr_face::QuadratureRule{fdim,refshape,T}
end

@inline getnbasefunctions(fs::ScalarFunctionSpace) = length(fs.N)
@inline getnquadpoints(fs::ScalarFunctionSpace) = length(fs.qr_weights)
@inline getnfacequadpoints(fs::ScalarFunctionSpace) = length(getpoints(fs.qr_face))
@inline getdetJdV(fs::ScalarFunctionSpace{dim,T,2}, cell::Int, q_point::Int) where {dim,T} = fs.detJ[cell,q_point]*fs.qr_weights[q_point]
@inline getdetJdV(fs::ScalarFunctionSpace{dim,T,1}, cell::Int, q_point::Int) where {dim,T} = fs.detJ[cell]*fs.qr_weights[q_point]
@inline getdetJdS(fs::ScalarFunctionSpace{dim,T,2}, cell::Int, face::Int, q_point::Int) where {dim,T} = fs.detJf[cell][face, q_point]*fs.qr_face_weights[q_point]
@inline getdetJdS(fs::ScalarFunctionSpace{dim,T,1}, cell::Int, face::Int, q_point::Int) where {dim,T} = fs.detJf[cell][face]*getweights(fs.qr_face)[q_point]
@inline shape_value(fs::ScalarFunctionSpace, q_point::Int, base_func::Int) = fs.N[base_func, q_point]
@inline face_shape_value(fs::ScalarFunctionSpace, face::Int, q_point::Int, base_func::Int) = fs.E[base_func, q_point][face]
@inline shape_gradient(fs::ScalarFunctionSpace{dim,T,2}, q_point::Int, base_func::Int, cell::Int) where {dim,T} = fs.dNdξ[base_func, q_point] ⋅ fs.Jinv[cell,q_point]
@inline shape_gradient(fs::ScalarFunctionSpace{dim,T,1}, q_point::Int, base_func::Int, cell::Int) where {dim,T} = fs.dNdξ[base_func, q_point] ⋅ fs.Jinv[cell]
@inline shape_divergence(fs::ScalarFunctionSpace{dim,T,2}, q_point::Int, base_func::Int, cell::Int) where {dim,T} = sum(fs.dNdξ[base_func, q_point] ⋅ fs.Jinv[cell,q_point])
@inline shape_divergence(fs::ScalarFunctionSpace{dim,T,1}, q_point::Int, base_func::Int, cell::Int) where {dim,T} = sum(fs.dNdξ[base_func, q_point] ⋅ fs.Jinv[cell])
@inline getdim(::ScalarFunctionSpace{dim}) where {dim} = dim

function ScalarFunctionSpace(mesh::PolygonalMesh, func_interpol::Interpolation{dim,shape,order},
    quad_degree = order+1,geom_interpol::Interpolation=get_default_geom_interpolator(dim, shape)) where {dim, shape, order}
    quad_rule = QuadratureRule{dim,shape}(DefaultQuad(), quad_degree)
    face_quad_rule = QuadratureRule{dim-1,shape}(DefaultQuad(), quad_degree)
    ScalarFunctionSpace(Float64, mesh, order, quad_rule, face_quad_rule, func_interpol, geom_interpol)
end

function ScalarFunctionSpace(::Type{T}, mesh::PolygonalMesh, order, quad_rule::QuadratureRule{dim,shape}, face_quad_rule::QuadratureRule{dim1,shape},
    func_interpol::Interpolation,geom_interpol::Interpolation=get_default_geom_interpolator(dim, shape)) where {dim,dim1, T,shape<:AbstractRefShape}

    @assert getdim(func_interpol) == getdim(geom_interpol)
    @assert getrefshape(func_interpol) == getrefshape(geom_interpol) == shape
    @assert dim1 == dim - 1
    geom_face_interpol = getlowerdiminterpol(geom_interpol)
    n_qpoints = length(getpoints(quad_rule))
    n_faceqpoints = length(getpoints(face_quad_rule))
    n_cells = numcells(mesh)
    n_faces = get_maxnfaces(mesh)
    isJconstant = (getorder(geom_interpol) == 1)
    NN = isJconstant ? 1 : 2
    Jdim = isJconstant ? 1 : n_qpoints
    Jfdim = isJconstant ? 1 : n_faceqpoints

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_interpol)
    N    = fill(zero(T)          * T(NaN), n_func_basefuncs, n_qpoints)
    dNdξ = fill(zero(Vec{dim,T}) * T(NaN), n_func_basefuncs, n_qpoints)

    # Face interpolation
    E    = fill(zeros(T,n_faces) * T(NaN), n_func_basefuncs, n_faceqpoints)

    # Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(geom_interpol)
    M    = fill(zero(T)          * T(NaN), n_geom_basefuncs, n_qpoints)
    dMdξ = fill(zero(Vec{dim,T}) * T(NaN), n_geom_basefuncs, n_qpoints)

    # Geometry face Interpolation
    n_geom_face_basefuncs = getnbasefunctions(geom_face_interpol)
    dLdξ = fill(zero(Vec{dim-1,T}) * T(NaN), n_geom_face_basefuncs, n_faceqpoints)

    for (qp, ξ) in enumerate(quad_rule.points)
        for i in 1:n_func_basefuncs
            dNdξ[i, qp] = gradient_value(func_interpol, i, ξ)
            N[i, qp] = value(func_interpol, i, ξ)
        end
        for i in 1:n_geom_basefuncs
            dMdξ[i, qp], M[i, qp] = gradient(ξ -> value(geom_interpol, i, ξ), ξ, :all)
        end
    end
    face_coords = reference_edges(shape(), Val{dim})
    for (qp, ξ) in enumerate(face_quad_rule.points)
        for i in 1:n_func_basefuncs
            E_f = zeros(T, n_faces)
            for j in 1:n_faces  # TODO: Here I assume all cells have the same number of faces
                #Map from reference dim-1 shape to reference dim shape face/edge
                η = zero(Vec{2,T})
                for (k,x) in enumerate(face_coords[j])
                    η += value(geom_face_interpol, k, ξ)*x
                end
                #Evaluate shape function on q_point map to edge j
                E_f[j] = value(func_interpol, i, η)
            end
            E[i, qp] = E_f
        end
        for i in 1:n_geom_face_basefuncs
            dLdξ[i, qp] = gradient(ξ -> value(geom_face_interpol, i, ξ), ξ)
        end
    end

    detJ = fill(T(NaN), n_cells, Jdim)
    detJf = fill(zeros(T,n_faces, Jfdim)*T(NaN), n_cells)
    Jinv = fill(zero(Tensor{2,dim,T}) * T(NaN), n_cells, Jdim)
    #Precompute detJ and invJ
    for (k,cell) in enumerate(get_cells(mesh))
        x = get_coordinates(cell, mesh)
        @inbounds for i in 1:Jdim
            fe_J = zero(Tensor{2,dim})
            for j in 1:n_geom_basefuncs
                fe_J += x[j] ⊗ dMdξ[j, i]
            end
            detJ_c = det(fe_J)
            detJ_c > 0.0 || throw(ArgumentError("det(J) is not positive: det(J) = $(detJ_c)"))
            detJ[k, i] = detJ_c
            Jinv[k,i] = inv(fe_J)
        end
        detJf_c = zeros(T,n_faces, Jfdim)
        @inbounds for i in 1:Jfdim
            for (l, face) in enumerate(get_faces(cell, mesh))
                x = get_coordinates(face, mesh)
                fef_J = zero(Tensor{1,dim})
                for j in 1:n_geom_face_basefuncs
                    fef_J += x[j] * dLdξ[j, i][1]  #TODO:search something better than this hack
                end
                #for line integral
                detJ_f = norm(fef_J)
                detJf_c[l, i] = detJ_f
            end
        end
        detJf[k] = detJf_c
    end
    MM = Tensors.n_components(Tensors.get_base(eltype(Jinv)))
    ScalarFunctionSpace{dim,T,NN,shape,dim1,order,MM}(N, dNdξ, E, detJ, detJf, Jinv, M, quad_rule.weights, face_quad_rule)
end

# VectorFunctionSpace
struct VectorFunctionSpace{dim,T<:Real,N,refshape<:AbstractRefShape,M,order,NN <:Int} <: DiscreteFunctionSpace{dim,T,refshape}
    n_dof::NN
    ssp::ScalarFunctionSpace{dim,T,N,refshape,M, order}
end

@inline getnquadpoints(fs::VectorFunctionSpace) = length(fs.ssp.qr_weights)
@inline getdetJdV(fs::VectorFunctionSpace, cell::Int, q_point::Int) = getdetJdV(fs.ssp, cell, q_point)
@inline getnbasefunctions(fs::VectorFunctionSpace) = fs.n_dof

function shape_value(fs::VectorFunctionSpace{dim,T}, q_point::Int, base_func::Int) where {dim,T}
    @assert 1 <= base_func <= fs.n_dof "invalid base function index: $base_func"
    N_comp = zeros(T, dim)
    n = size(fs.ssp.N,1)
    N_comp[div(base_func,n+1)+1] = fs.ssp.N[mod1(base_func,n),q_point]
    return N_comp
end

function face_shape_value(fs::VectorFunctionSpace{dim,T}, face::Int, q_point::Int, base_func::Int) where {dim, T}
    @assert 1 <= base_func <= fs.n_dof "invalid base function index: $base_func"
    N_comp = zeros(T, dim)
    n = size(fs.ssp.N,1)
    N_comp[div(base_func,n+1)+1] = fs.ssp.E[mod1(base_func,n), q_point][face]
    return N_comp
end

function shape_gradient(fs::VectorFunctionSpace{dim,T,2}, q_point::Int, base_func::Int, cell::Int)  where {dim,T}
    @assert 1 <= base_func <= fs.n_dof "invalid base function index: $base_func"
    dN_comp = zeros(T, dim, dim)
    n = size(fs.ssp.N,1)
    dN_comp[div(base_func,n+1)+1, :] = fs.ssp.dNdξ[mod1(base_func,n), q_point]
    return Tensor{2,dim,T}((dN_comp...)) ⋅ fs.ssp.Jinv[cell,q_point]
end

function shape_gradient(fs::VectorFunctionSpace{dim,T,1}, q_point::Int, base_func::Int, cell::Int)  where {dim,T}
    @assert 1 <= base_func <= fs.n_dof "invalid base function index: $base_func"
    dN_comp = zeros(T, dim, dim)
    n = size(fs.ssp.N,1)
    dN_comp[div(base_func,n+1)+1, :] = fs.ssp.dNdξ[mod1(base_func,n), q_point]
    return Tensor{2,dim,T}((dN_comp...)) ⋅ fs.ssp.Jinv[cell]
end

function shape_divergence(fs::VectorFunctionSpace{dim,T}, q_point::Int, base_func::Int, cell::Int)  where {dim,T}
    @assert 1 <= base_func <= fs.n_dof "invalid base function index: $base_func"
    return trace(shape_gradient(fs, q_point, base_func, cell))
end

function VectorFunctionSpace(mesh::PolygonalMesh, func_interpol::Interpolation{dim,shape,order},
    quad_degree = order+1,geom_interpol::Interpolation=get_default_geom_interpolator(dim, shape)) where {dim, shape, order}
    quad_rule = QuadratureRule{dim,shape}(DefaultQuad(), quad_degree)
    face_quad_rule = QuadratureRule{dim-1,shape}(DefaultQuad(), quad_degree)
    ssp = ScalarFunctionSpace(Float64, mesh, order, quad_rule, face_quad_rule, func_interpol, geom_interpol)
    n_func_basefuncs = getnbasefunctions(func_interpol)
    dof = n_func_basefuncs*dim
    VectorFunctionSpace(dof,ssp)
end

# Scalar Trace Function Space (Scalar functions defined only on cell boundaries)
struct ScalarTraceFunctionSpace{dim,T<:Real,NN,refshape<:AbstractRefShape, order} <: DiscreteFunctionSpace{dim,T,refshape}
    N::Matrix{T}
    dNdξ::Matrix{Vec{dim,T}}
    detJ::Vector{Matrix{T}}
    qr_weights::Vector{T}
end

@inline getnbasefunctions(fs::ScalarTraceFunctionSpace) = length(fs.N)
@inline getnquadpoints(fs::ScalarTraceFunctionSpace) = length(fs.qr_weights)
@inline getdetJdS(fs::ScalarTraceFunctionSpace{dim,T,2}, cell::Int, q_point::Int) where {dim,T} = fs.detJ[cell,q_point]*fs.qr_weights[q_point]
@inline getdetJdS(fs::ScalarTraceFunctionSpace{dim,T,1}, cell::Int, q_point::Int) where {dim,T} = fs.detJ[cell]*fs.qr_weights[q_point]
@inline shape_value(fs::ScalarTraceFunctionSpace, q_point::Int, base_func::Int) = fs.N[base_func, q_point]

function ScalarTraceFunctionSpace(psp::ScalarFunctionSpace{dim,T,N,refshape,order,M},
    func_interpol::Interpolation{dim1,refshape,order}) where {dim, T,N,refshape, order,M, dim1}
    @assert getdim(psp) == getdim(func_interpol)+1
    detJ = psp.detJf
    qr_weights = getweights(psp.qr_face)
    qr_points = getpoints(psp.qr_face)
    ScalarTraceFunctionSpace(Float64, N, func_interpol, detJ, qr_weights, qr_points)
end

function ScalarTraceFunctionSpace(::Type{T}, NN, func_interpol::Interpolation{dim,refshape,order},
    detJ::Vector, qr_weights::Vector, qr_points::Vector) where {dim, refshape, order, T}
    n_qpoints = length(qr_points)

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(func_interpol)
    N    = fill(zero(T)          * T(NaN), n_func_basefuncs, n_qpoints)
    dNdξ = fill(zero(Vec{dim,T}) * T(NaN), n_func_basefuncs, n_qpoints)

    for (qp, ξ) in enumerate(qr_points)
        for i in 1:n_func_basefuncs
            dNdξ[i, qp] = gradient_value(func_interpol, i, ξ)
            N[i, qp] = value(func_interpol, i, ξ)
        end
    end
    ScalarTraceFunctionSpace{dim,T,NN,refshape, order}(N, dNdξ,detJ, qr_weights)
end
