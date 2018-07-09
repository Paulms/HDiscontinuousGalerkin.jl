abstract type AbstractScalarFunctionSpace{dim,T,shape,order,M,N1,N2,N3} <: DiscreteFunctionSpace{dim,T,shape} end
abstract type AbstractFacesFunctionSpace{dim,fdim,T,comps} end

# ScalarFunctionSpace
struct FacesFunctionSpace{dim,fdim,T,comps} <: AbstractFacesFunctionSpace{dim,fdim,T,comps}
    L::Array{T,3}
    E :: Matrix{Vector{T}}
    detJf::Matrix{T}
    normals::Matrix{Vec{dim,T}}
    qr_face_weigths::Vector{T}
    qr_face_points::Vector{Vec{fdim,T}}
end

struct ScalarFunctionSpace{dim,T<:Real,shape<:AbstractRefShape,order,M,N1,N2,N3} <: AbstractScalarFunctionSpace{dim,T,shape,order,M,N1,N2,N3}
    N::Matrix{T}
    dNdξ::Matrix{Vec{dim,T}}
    detJ::Vector{T}
    Jinv::Vector{Tensor{2,dim,T,M}}
    M::Matrix{T}
    qr_weights::Vector{T}
    fe::FiniteElement{dim,shape,order,1}
    mesh::PolygonalMesh{dim,N1,N2,N3,T}
end

function ScalarFunctionSpace(mesh::PolygonalMesh, felem::FiniteElement{dim,shape,order,gorder};face_data = true,
    quad_degree = order+1) where {dim, shape, order,gorder}
    quad_rule = QuadratureRule{dim,shape}(DefaultQuad(), quad_degree)
    fs = _scalar_fs(Float64, mesh, quad_rule, felem)
    fd = face_data ? _sface_data(Float64, mesh, quad_degree, felem) : nothing
    if face_data
        return fs, fd
    else
        return fs
    end
end

function _scalar_fs(::Type{T}, mesh::PolygonalMesh{dim,N1,N2,N3,T}, quad_rule::QuadratureRule{dim,shape},
    felem::FiniteElement{dim,shape,order,1}) where {dim, T,shape<:AbstractRefShape,N1,N2,N3,order}
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

    detJ = fill(T(NaN), n_cells)
    Jinv = fill(zero(Tensor{2,dim,T}) * T(NaN), n_cells)
    coords = fill(zero(Vec{dim,T}) * T(NaN), n_nodes_per_cell(mesh))
    #Precompute detJ and invJ
    for k in 1:n_cells
        x = get_cell_coordinates!(coords,k, mesh)
        fe_J = zero(Tensor{2,dim})
        for j in 1:n_geom_basefuncs
            fe_J += x[j] ⊗ dMdξ[j, 1]
        end
        detJ_c = det(fe_J)
        detJ_c > 0.0 || throw(ArgumentError("det(J) is not positive: det(J) = $(detJ_c)"))
        detJ[k] = detJ_c
        Jinv[k] = inv(fe_J)
    end
    MM = Tensors.n_components(Tensors.get_base(eltype(Jinv)))
    ScalarFunctionSpace{dim,T,shape,order,MM,N1,N2,N3}(N, dNdξ, detJ, Jinv,
    M, getweights(quad_rule), felem, mesh)
end

function _sface_data(::Type{T}, mesh::PolygonalMesh{dim,N1,N2,N3,T}, quad_degree::Int,
    felem::FiniteElement{dim,shape,order,1}, comps = 1) where {dim, T,shape<:AbstractRefShape,N1,N2,N3,order}
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

    detJf = fill(T(NaN), n_cells, n_faces)
    coords = fill(zero(Vec{dim,T}) * T(NaN), n_nodes_per_cell(mesh))
    normals = fill(zero(Vec{dim,T}) * T(NaN), n_cells, n_faces)
    #Precompute detJ and invJ
    for k in 1:n_cells
        x = get_cell_coordinates!(coords,k, mesh)
        for l in 1:n_faces
            fef_J = zero(Tensor{2,dim})
            for j in 1:n_geom_basefuncs
                fef_J += x[j] ⊗ dLdξ[j, 1, l]
            end
            weight_norm = weighted_normal(fef_J, l, shape, Val{dim})
            detJ_f = norm(weight_norm)
            normals[k,l] = weight_norm / detJ_f
            detJ_f > 0.0 || throw(ArgumentError("det(Jf) is not positive: det(Jf) = $(detJ_f)"))
            detJf[k,l] = detJ_f
        end

    end
    FacesFunctionSpace{dim,fdim,T,comps}(L, E, detJf, normals, q_ref_faceweights, q_ref_facepoints)
end

########### Data Functions
@inline getngeobasefunctions(fs::AbstractScalarFunctionSpace) = size(fs.M, 1)
@inline getnquadpoints(fs::AbstractScalarFunctionSpace) = length(fs.qr_weights)
@inline getnbasefunctions(fs::AbstractScalarFunctionSpace) = size(fs.N,1)
@inline getdetJdV(fs::ScalarFunctionSpace, cell::Int, q_point::Int) = fs.detJ[cell]*fs.qr_weights[q_point]
@inline shape_value(fs::AbstractScalarFunctionSpace, q_point::Int, base_func::Int) = fs.N[base_func, q_point]
@inline shape_gradient(fs::ScalarFunctionSpace, q_point::Int, base_func::Int, cell::Int) = fs.dNdξ[base_func, q_point] ⋅ fs.Jinv[cell]
@inline shape_divergence(fs::ScalarFunctionSpace, q_point::Int, base_func::Int, cell::Int) = sum(fs.dNdξ[base_func, q_point] ⋅ fs.Jinv[cell])
@inline geometric_value(fs::AbstractScalarFunctionSpace, q_point::Int, base_func::Int) = fs.M[base_func, q_point]
@inline getdim(::AbstractScalarFunctionSpace{dim}) where {dim} = dim
@inline reference_coordinate(fs::AbstractScalarFunctionSpace{dim,T},cell::Int, mesh::PolygonalMesh, x::Vec{dim,T}) where {dim,T} = fs.Jinv[cell]⋅(x-mesh.nodes[mesh.cells[cell].nodes[1]].x)
@inline getfiniteelement(fs::AbstractScalarFunctionSpace) = fs.fe
@inline getnlocaldofs(fs::AbstractScalarFunctionSpace) = getnbasefunctions(fs)
@inline getmesh(fs::AbstractScalarFunctionSpace) = fs.mesh

# Face Data
@inline getngeobasefunctions(fs::AbstractFacesFunctionSpace{dim,fdim,T,1}) where {dim,fdim,T} = size(fs.L, 1)
@inline getnfacequadpoints(fs::AbstractFacesFunctionSpace) = length(fs.qr_face_weigths)
@inline getdetJdS(fs::FacesFunctionSpace{dim,fdim,T,1}, cell::Int, face::Int, q_point::Int) where {dim,fdim,T} = fs.detJf[cell,face]*fs.qr_face_weigths[q_point]
@inline shape_value(fs::AbstractFacesFunctionSpace{dim,fdim,T,1}, face::Int, q_point::Int, base_func::Int, orientation::Bool = true) where {dim,fdim,T} = orientation ? fs.E[base_func, q_point][face] : fs.E[base_func, end - q_point+1][face]
@inline geometric_value(fs::AbstractFacesFunctionSpace, face::Int, q_point::Int, base_func::Int) = fs.L[base_func, q_point, face]

"""
    getnormal(fs::ScalarFunctionSpace, cell::Int, face::Int, qp::Int)
Return the normal at the quadrature point `qp` for the face `face` at
cell `cell` of the `ScalarFunctionSpace` object.
"""
@inline get_normal(fs::FacesFunctionSpace, cell::Int, face::Int) = fs.normals[cell, face]
