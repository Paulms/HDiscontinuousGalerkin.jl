abstract type AbstractScalarFunctionSpace{dim,T,FE<:FiniteElement,MM,N1,N2,N3,fdim} <: DiscreteFunctionSpace{dim,T,FE} end

struct ScalarFunctionSpace{dim,T<:Real,FE<:FiniteElement,MM,N1,N2,N3,fdim} <: AbstractScalarFunctionSpace{dim,T,FE,MM,N1,N2,N3,fdim}
    N::Matrix{T}
    dNdx::Matrix{Vec{dim,T}}
    dNdξ::Matrix{Vec{dim,T}}
    detJ::ScalarWrapper{T}
    M::Matrix{T}
    dMdξ::Matrix{Vec{dim,T}}
    qr_weights::Vector{T}
    fe::FE
    mesh::PolygonalMesh{dim,N1,N2,N3,T}
    Jinv::ScalarWrapper{Tensor{2,dim,T,MM}}
    update_face_data::Bool
    L::Array{T,3}
    E :: Matrix{Vector{T}}
    detJf::Vector{T}
    normals::Vector{Vec{dim,T}}
    qr_face_weigths::Vector{T}
    qr_face_points::Vector{Vec{fdim,T}}
end

function ScalarFunctionSpace(mesh::PolygonalMesh, felem::FiniteElement{dim,shape,order,gorder};update_face_data::Bool = true,
    quad_degree::Int = order+1) where {dim, shape, order,gorder}
    quad_rule = QuadratureRule{dim,shape}(DefaultQuad(), quad_degree)
    f_quad_rule = QuadratureRule{dim-1,shape}(DefaultQuad(), quad_degree)
    fs = _scalar_fs(Float64, mesh, quad_rule, f_quad_rule, felem, update_face_data)
end

function _scalar_fs(::Type{T}, mesh::PolygonalMesh{dim,N1,N2,N3,T}, quad_rule::QuadratureRule{dim,shape},
    f_quad_rule::QuadratureRule{fdim,shape},
    felem::FiniteElement{dim,shape,order,1}, update_face_data::Bool) where {dim, fdim,T,shape<:AbstractRefShape,N1,N2,N3,order}
    n_qpoints = length(getpoints(quad_rule))
    n_cells = getncells(mesh)
    n_faces = n_faces_per_cell(mesh)
    n_faceqpoints = length(getpoints(f_quad_rule))
    q_ref_facepoints = getpoints(f_quad_rule)
    q_ref_faceweights = getweights(f_quad_rule)
    face_quad_rule = create_face_quad_rule(f_quad_rule)

    # Function interpolation
    n_func_basefuncs = getnbasefunctions(felem)
    N    = fill(zero(T)          * T(NaN), n_func_basefuncs, n_qpoints)
    dNdx = fill(zero(Vec{dim,T}) * T(NaN), n_func_basefuncs, n_qpoints)
    dNdξ = fill(zero(Vec{dim,T}) * T(NaN), n_func_basefuncs, n_qpoints)

    # Geometry interpolation
    n_geom_basefuncs = getngeombasefunctions(felem)
    M    = fill(zero(T)          * T(NaN), n_geom_basefuncs, n_qpoints)
    dMdξ = fill(zero(Vec{dim,T}) * T(NaN), n_geom_basefuncs, n_qpoints)

    # Face interpolation
    n_func_basefuncs = getnbasefunctions(felem)
    E    = fill(zeros(T,n_faces) * T(NaN), n_func_basefuncs, n_faceqpoints)

    # Geometry face Interpolation
    n_geom_basefuncs = getngeombasefunctions(felem)
    L =    fill(zero(T)          * T(NaN), n_geom_basefuncs, n_faceqpoints, n_faces)
    dLdξ = fill(zero(Vec{dim,T}) * T(NaN), n_geom_basefuncs, n_faceqpoints, n_faces)

    if update_face_data
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
    end

    for (qp, ξ) in enumerate(quad_rule.points)
        for i in 1:n_func_basefuncs
            dNdξ[i, qp], N[i, qp]  = gradient(ξ -> value(felem, i, ξ), ξ, :all)
        end
        for i in 1:n_geom_basefuncs
            dMdξ[i, qp], M[i, qp] = gradient(ξ -> geom_value(felem, i, ξ), ξ, :all)
        end
    end

    detJ = ScalarWrapper(T(NaN))
    detJf = fill(T(NaN), n_faces)
    normals = fill(zero(Vec{dim,T}) * T(NaN), n_faces)
    Jinv = ScalarWrapper(zero(Tensor{2,dim,T,2*dim}) )

    MM = Tensors.n_components(Tensors.get_base(typeof(Jinv[])))

    ScalarFunctionSpace{dim,T,typeof(felem),MM,N1,N2,N3,fdim}(N, dNdx, dNdξ, detJ,
    M, dMdξ, getweights(quad_rule), felem, mesh, Jinv,update_face_data, L, E, detJf, normals,
    q_ref_faceweights, q_ref_facepoints)
end

function reinit!(fs::ScalarFunctionSpace{dim}, x::AbstractVector{Vec{dim,T}}) where {dim,T}
    n_geom_basefuncs = getngeombasefunctions(fs.fe)
    n_func_basefuncs = getnbasefunctions(fs.fe)
    @assert length(x) == n_geom_basefuncs
    fecv_J = zero(Tensor{2,dim})
    for j in 1:n_geom_basefuncs
        fecv_J += x[j] ⊗ fs.dMdξ[j, 1]
    end
    detJ = det(fecv_J)
    detJ > 0.0 || throw(ArgumentError("det(J) is not positive: det(J) = $(detJ)"))
    fs.detJ[] = detJ
    fs.Jinv[] = inv(fecv_J)

    for j in 1:n_func_basefuncs, i in 1:length(fs.qr_weights)
        fs.dNdx[j, i] = fs.dNdξ[j, i] ⋅ fs.Jinv[]
    end
    if fs.update_face_data
        n_faces = n_faces_per_cell(fs.mesh)
        shape = getshape(fs.fe)
        for l in 1:n_faces
            fef_J = zero(Tensor{2,dim})
            for j in 1:n_geom_basefuncs
                fef_J += x[j] ⊗ fs.dLdξ[j, 1, l]
            end
            weight_norm = weighted_normal(fef_J, l, shape, Val{dim})
            detJ_f = norm(weight_norm)
            fs.normals[l] = weight_norm / detJ_f
            detJ_f > 0.0 || throw(ArgumentError("det(Jf) is not positive: det(Jf) = $(detJ_f)"))
            fs.detJf[l] = detJ_f
        end
    end
end

########### Data Functions
@inline getngeobasefunctions(fs::AbstractScalarFunctionSpace) = size(fs.M, 1)
@inline getnquadpoints(fs::AbstractScalarFunctionSpace) = length(fs.qr_weights)
@inline getnbasefunctions(fs::AbstractScalarFunctionSpace) = size(fs.N,1)
@inline getdetJdV(fs::ScalarFunctionSpace, q_point::Int) = fs.detJ[]*fs.qr_weights[q_point]
@inline shape_value(fs::AbstractScalarFunctionSpace, q_point::Int, base_func::Int) = fs.N[base_func, q_point]
@inline shape_gradient(fs::ScalarFunctionSpace, q_point::Int, base_func::Int) = fs.dNdx[base_func, q_point]
@inline shape_divergence(fs::ScalarFunctionSpace, q_point::Int, base_func::Int) = sum(fs.dNdx[base_func, q_point])
@inline geometric_value(fs::AbstractScalarFunctionSpace, q_point::Int, base_func::Int) = fs.M[base_func, q_point]
@inline getdim(::AbstractScalarFunctionSpace{dim}) where {dim} = dim
@inline reference_coordinate(fs::AbstractScalarFunctionSpace{dim,T},cell::Int, mesh::PolygonalMesh, x::Vec{dim,T}) where {dim,T} = fs.Jinv[]⋅(x-mesh.nodes[mesh.cells[cell].nodes[1]].x)
@inline getfiniteelement(fs::AbstractScalarFunctionSpace) = fs.fe
@inline getnlocaldofs(fs::AbstractScalarFunctionSpace) = getnbasefunctions(fs)
@inline getmesh(fs::AbstractScalarFunctionSpace) = fs.mesh

# Face Data
@inline getnfacegeobasefunctions(fs::AbstractScalarFunctionSpace) = size(fs.L, 1)
@inline getnfacequadpoints(fs::AbstractScalarFunctionSpace) = length(fs.qr_face_weigths)
@inline getfacedetJdS(fs::ScalarFunctionSpace, face::Int, q_point::Int) = fs.detJf[face]*fs.qr_face_weigths[q_point]
@inline shape_value(fs::AbstractScalarFunctionSpace, face::Int, q_point::Int, base_func::Int, orientation::Bool = true) = orientation ? fs.E[base_func, q_point][face] : fs.E[base_func, end - q_point+1][face]
@inline geometric_value(fs::AbstractScalarFunctionSpace, face::Int, q_point::Int, base_func::Int) = fs.L[base_func, q_point, face]

"""
    getnormal(fs::ScalarFunctionSpace, cell::Int, face::Int, qp::Int)
Return the normal at the quadrature point `qp` for the face `face` at
cell `cell` of the `ScalarFunctionSpace` object.
"""
@inline get_normal(fs::ScalarFunctionSpace, face::Int) = fs.normals[face]
