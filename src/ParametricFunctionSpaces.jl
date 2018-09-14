# Scalar Spaces
struct ParametricScalarFunctionSpace{dim,fdim,T,comps} <: AbstractScalarFunctionSpace{dim,fdim,T,comps}
    L::Array{T,3}
    E :: Matrix{Vector{T}}
    detJf::Array{T,3}
    normals::Array{Vec{dim,T},3}
    qr_face_weigths::Vector{T}
    qr_face_points::Vector{Vec{fdim,T}}
end
struct ParametricScalarFunctionSpace{dim,T<:Real,FE<:FiniteElement,M,N1,N2,N3} <: AbstractScalarFunctionSpace{dim,T,FE,M,N1,N2,N3}
    N::Matrix{T}
    dNdξ::Matrix{Vec{dim,T}}
    detJ::Matrix{T}
    Jinv::Matrix{Tensor{2,dim,T,M}}
    M::Matrix{T}
    qr_weights::Vector{T}
    fe::FE
    mesh::PolygonalMesh{dim,N1,N2,N3,T}
end
function _scalar_fs(::Type{T}, mesh::PolygonalMesh{dim,N1,N2,N3,T}, quad_rule::QuadratureRule{dim,shape},
    felem::FiniteElement{dim,shape,order,gorder}) where {dim, T,shape<:AbstractRefShape,N1,N2,N3,order,gorder}
    n_qpoints = length(getpoints(quad_rule))
    n_cells = getncells(mesh)

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(felem)
    N    = fill(zero(T)          * T(NaN), n_func_basefuncs, n_qpoints)
    dNdξ = fill(zero(Vec{dim,T}) * T(NaN), n_func_basefuncs, n_qpoints)

    # Geometry interpolation
    n_geom_basefuncs = getngeombasefunctions(felem)
    M    = fill(zero(T)          * T(NaN), n_geom_basefuncs, n_qpoints)
    dMdξ = fill(zero(Vec{dim,T}) * T(NaN), n_geom_basefuncs, n_qpoints)

    for (qp, ξ) in enumerate(quad_rule.points)
        for i in 1:n_func_basefuncs
            dNdξ[i, qp], N[i, qp]  = gradient(ξ -> value(felem, i, ξ), ξ, :all)
        end
        for i in 1:n_geom_basefuncs
            dMdξ[i, qp], M[i, qp] = gradient(ξ -> geom_value(felem, i, ξ), ξ, :all)
        end
    end

    detJ = fill(T(NaN), n_cells, n_qpoints)
    Jinv = fill(zero(Tensor{2,dim,T}) * T(NaN), n_cells, n_qpoints)
    coords = fill(zero(Vec{dim,T}) * T(NaN), n_nodes_per_cell(mesh))
    #Precompute detJ and invJ
    for k in 1:n_cells
        x = get_cell_coordinates!(coords,k, mesh)
        @inbounds for i in 1:n_qpoints
            fe_J = zero(Tensor{2,dim})
            for j in 1:n_geom_basefuncs
                fe_J += x[j] ⊗ dMdξ[j, i]
            end
            detJ_c = det(fe_J)
            detJ_c > 0.0 || throw(ArgumentError("det(J) is not positive: det(J) = $(detJ_c)"))
            detJ[k, i] = detJ_c
            Jinv[k,i] = inv(fe_J)
        end
    end
    MM = Tensors.n_components(Tensors.get_base(eltype(Jinv)))
    ParametricScalarFunctionSpace{dim,T,eltype(felem),MM,N1,N2,N3}(N, dNdξ, detJ, Jinv,
    M, getweights(quad_rule), felem, mesh)
end

function _sface_data(::Type{T}, mesh::PolygonalMesh{dim,N1,N2,N3,T}, quad_degree::Int,
    felem::FiniteElement{dim,shape,order,gorder}, comps = 1) where {dim, T,shape<:AbstractRefShape,N1,N2,N3,order,gorder}
    fdim = dim - 1
    f_quad_rule = QuadratureRule{fdim,shape}(DefaultQuad(), quad_degree)
    n_faceqpoints = length(getpoints(f_quad_rule))
    q_ref_facepoints = getpoints(f_quad_rule)
    q_ref_faceweights = getweights(f_quad_rule)
    face_quad_rule = create_face_quad_rule(f_quad_rule)


    n_cells = getncells(mesh)
    n_faces = n_faces_per_cell(mesh)

    # Face interpolation
    n_func_basefuncs = getnbasefunctions(felem)
    E    = fill(zeros(T,n_faces) * T(NaN), n_func_basefuncs, n_faceqpoints)

    # Geometry face Interpolation
    n_geom_basefuncs = getngeombasefunctions(felem)
    L =    fill(zero(T)          * T(NaN), n_geom_basefuncs, n_faceqpoints, n_faces)
    dLdξ = fill(zero(Vec{dim,T}) * T(NaN), n_geom_basefuncs, n_faceqpoints, n_faces)

    for face in 1:n_faces, (qp, ξ) in enumerate(face_quad_rule[face].points)
        for i in 1:n_geom_basefuncs
            dLdξ[i, qp, face], L[i, qp, face] = gradient(ξ -> geom_value(felem, i, ξ), ξ, :all)
        end
    end

    for qp in 1:n_faceqpoints
        for i in 1:n_func_basefuncs
            E_f = zeros(T, n_faces)
            for j in 1:n_faces
                #Evaluate shape function on q_point map to edge j
                E_f[j] = value(felem, i, face_quad_rule[j].points[qp])
            end
            E[i, qp] = E_f
        end
    end

    detJf = fill(T(NaN), n_cells, n_faces, n_faceqpoints)
    coords = fill(zero(Vec{dim,T}) * T(NaN), n_nodes_per_cell(mesh))
    normals = fill(zero(Vec{dim,T}) * T(NaN), n_cells, n_faces, n_faceqpoints)
    #Precompute detJ and invJ
    for k in 1:n_cells
        x = get_cell_coordinates!(coords,k, mesh)
        @inbounds for i in 1:n_faceqpoints
            for l in 1:n_faces
                fef_J = zero(Tensor{2,dim})
                for j in 1:n_geom_basefuncs
                    fef_J += x[j] ⊗ dLdξ[j, i, l]
                end
                weight_norm = weighted_normal(fef_J, l, shape, Val{dim})
                detJ_f = norm(weight_norm)
                normals[k,l,i] = weight_norm / detJ_f
                detJ_f > 0.0 || throw(ArgumentError("det(Jf) is not positive: det(Jf) = $(detJ_f)"))
                detJf[k,l,i] = detJ_f
            end
        end
    end
    ParametricScalarFunctionSpace{dim,fdim,T,comps}(L, E, detJf, normals, q_ref_faceweights, q_ref_facepoints)
end

#Data
@inline getdetJdV(fs::ParametricScalarFunctionSpace, cell::Int, q_point::Int) = fs.detJ[cell,q_point]*fs.qr_weights[q_point]
@inline shape_gradient(fs::ParametricScalarFunctionSpace, q_point::Int, base_func::Int, cell::Int) = fs.dNdξ[base_func, q_point] ⋅ fs.Jinv[cell,q_point]
@inline shape_divergence(fs::ParametricScalarFunctionSpace, q_point::Int, base_func::Int, cell::Int) = sum(fs.dNdξ[base_func, q_point] ⋅ fs.Jinv[cell,q_point])
@inline getdetJdS(fs::ParametricScalarFunctionSpace, cell::Int, face::Int, q_point::Int) = fs.detJf[cell,face, q_point]*fs.qr_face_weigths[q_point]
"""
    getnormal(fs::ScalarFunctionSpace, cell::Int, face::Int, qp::Int)
Return the normal at the quadrature point `qp` for the face `face` at
cell `cell` of the `ScalarFunctionSpace` object.
"""
@inline get_normal(fs::ParametricScalarFunctionSpace, cell::Int, face::Int, qp::Int) = fs.normals[cell, face, qp]

#Vector Space
# VectorFunctionSpace
struct ParametricVectorFunctionSpace{dim,T<:Real,FE<:FiniteElement,M,N1,N2,N3} <: AbstractVectorFunctionSpace{dim,T,FE,M,N1,N2,N3}
    n_dof::Int
    ssp::ParametricScalarFunctionSpace{dim,T,FE,M,N1,N2,N3}
end

#Constructor
function VectorFunctionSpace(mesh::PolygonalMesh, felem::FiniteElement{dim,shape,order,gorder};
    face_data = true, quad_degree = order+1) where {dim, shape, order, gorder}
    quad_rule = QuadratureRule{dim,shape}(DefaultQuad(), quad_degree)
    fs = _scalar_fs(Float64, mesh, quad_rule, felem)
    fd = face_data ? _sface_data(Float64, mesh, quad_degree, felem, dim) : nothing
    n_func_basefuncs = getnbasefunctions(felem)
    dof = n_func_basefuncs*dim
    if face_data
        return ParametricVectorFunctionSpace(dof,fs), fd
    else
        return ParametricVectorFunctionSpace(dof,fs)
    end
end

#Data
function shape_gradient(fs::ParametricVectorFunctionSpace{dim,T}, q_point::Int, base_func::Int, cell::Int)  where {dim,T}
    @assert 1 <= base_func <= fs.n_dof "invalid base function index: $base_func"
    dN_comp = zeros(T, dim, dim)
    n = size(fs.ssp.N,1)
    dN_comp[div(base_func,n+1)+1, :] = fs.ssp.dNdξ[mod1(base_func,n), q_point]
    return Tensor{2,dim,T}((dN_comp...,)) ⋅ fs.ssp.Jinv[cell,q_point]
end
