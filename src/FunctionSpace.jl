#ScalarFunctionSpace
struct ScalarFunctionSpace{dim,T<:Real,NN,refshape<:AbstractRefShape,fdim,order,M} <: DiscreteFunctionSpace{dim,T,refshape}
    N::Matrix{T}
    dNdξ::Matrix{Vec{dim,T}}
    E :: Matrix{Vector{T}}
    detJ::Matrix{T}
    detJf::Vector{Matrix{T}}
    Jinv::Matrix{Tensor{2,dim,T,M}}
    M::Matrix{T}
    normals::Array{Vec{dim,T},3}
    qr_weights::Vector{T}
    qr_face_weigths::Vector{T}
    qr_face_points::Vector{Vec{fdim,T}}
end

@inline getngeobasefunctions(fs::ScalarFunctionSpace) = size(fs.M, 1)
@inline getnquadpoints(fs::ScalarFunctionSpace) = length(fs.qr_weights)
@inline getnfacequadpoints(fs::ScalarFunctionSpace) = length(fs.qr_face_weigths)
@inline getnbasefunctions(fs::ScalarFunctionSpace) = size(fs.N,1)
@inline getdetJdV(fs::ScalarFunctionSpace{dim,T,2}, cell::Int, q_point::Int) where {dim,T} = fs.detJ[cell,q_point]*fs.qr_weights[q_point]
@inline getdetJdV(fs::ScalarFunctionSpace{dim,T,1}, cell::Int, q_point::Int) where {dim,T} = fs.detJ[cell]*fs.qr_weights[q_point]
@inline getdetJdS(fs::ScalarFunctionSpace{dim,T,2}, cell::Int, face::Int, q_point::Int) where {dim,T} = fs.detJf[cell][face, q_point]*fs.qr_face_weigths[q_point]
@inline getdetJdS(fs::ScalarFunctionSpace{dim,T,1}, cell::Int, face::Int, q_point::Int) where {dim,T} = fs.detJf[cell][face]*fs.qr_face_weigths[q_point]
@inline shape_value(fs::ScalarFunctionSpace, q_point::Int, base_func::Int) = fs.N[base_func, q_point]
@inline face_shape_value(fs::ScalarFunctionSpace, face::Int, q_point::Int, base_func::Int, orientation::Bool = true) = orientation ? fs.E[base_func, q_point][face] : fs.E[base_func, end - q_point+1][face]
@inline shape_gradient(fs::ScalarFunctionSpace{dim,T,2}, q_point::Int, base_func::Int, cell::Int) where {dim,T} = fs.dNdξ[base_func, q_point] ⋅ fs.Jinv[cell,q_point]
@inline shape_gradient(fs::ScalarFunctionSpace{dim,T,1}, q_point::Int, base_func::Int, cell::Int) where {dim,T} = fs.dNdξ[base_func, q_point] ⋅ fs.Jinv[cell]
@inline shape_divergence(fs::ScalarFunctionSpace{dim,T,2}, q_point::Int, base_func::Int, cell::Int) where {dim,T} = sum(fs.dNdξ[base_func, q_point] ⋅ fs.Jinv[cell,q_point])
@inline shape_divergence(fs::ScalarFunctionSpace{dim,T,1}, q_point::Int, base_func::Int, cell::Int) where {dim,T} = sum(fs.dNdξ[base_func, q_point] ⋅ fs.Jinv[cell])
@inline geometric_value(fs::ScalarFunctionSpace, q_point::Int, base_func::Int) = fs.M[base_func, q_point]
@inline getdim(::ScalarFunctionSpace{dim}) where {dim} = dim

"""
    getnormal(fs::ScalarFunctionSpace, cell::Int, face::Int, qp::Int)
Return the normal at the quadrature point `qp` for the face `face` at
cell `cell` of the `ScalarFunctionSpace` object.
"""
@inline get_normal(fs::ScalarFunctionSpace{dim,T,2}, cell::Int, face::Int, qp::Int) where {dim,T} = fs.normals[qp, cell, face, qp]
@inline get_normal(fs::ScalarFunctionSpace{dim,T,1}, cell::Int, face::Int) where {dim,T} = fs.normals[qp, cell, face]

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
    n_faceqpoints = length(getpoints(face_quad_rule))
    q_ref_facepoints = getpoints(face_quad_rule)
    q_ref_faceweights = getweights(face_quad_rule)
    face_quad_rule = create_face_quad_rule(face_quad_rule, func_interpol)

    n_qpoints = length(getpoints(quad_rule))
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
    L =    fill(zero(T)          * T(NaN), n_geom_basefuncs, n_faceqpoints, n_faces)
    dLdξ = fill(zero(Vec{dim,T}) * T(NaN), n_geom_basefuncs, n_faceqpoints, n_faces)

    # Normals
    normals = zeros(Vec{dim,T}, n_cells, n_faces, Jfdim)

    for (qp, ξ) in enumerate(quad_rule.points)
        for i in 1:n_func_basefuncs
            dNdξ[i, qp] = gradient_value(func_interpol, i, ξ)
            N[i, qp] = value(func_interpol, i, ξ)
        end
        for i in 1:n_geom_basefuncs
            dMdξ[i, qp], M[i, qp] = gradient(ξ -> value(geom_interpol, i, ξ), ξ, :all)
        end
    end
    for face in 1:n_faces, (qp, ξ) in enumerate(face_quad_rule[face].points)
        for i in 1:n_geom_basefuncs
            dLdξ[i, qp, face], L[i, qp, face] = gradient(ξ -> value(geom_interpol, i, ξ), ξ, :all)
        end
    end

    for qp in 1:n_faceqpoints
        for i in 1:n_func_basefuncs
            E_f = zeros(T, n_faces)
            for j in 1:n_faces  # TODO: Here I assume all cells have the same number of faces
                #Evaluate shape function on q_point map to edge j
                E_f[j] = value(func_interpol, i, face_quad_rule[j].points[qp])
            end
            E[i, qp] = E_f
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
                fef_J = zero(Tensor{2,dim})
                for j in 1:n_geom_basefuncs
                    fef_J += x[j] ⊗ dLdξ[j, i, l]
                end
                weight_norm = weighted_normal(fef_J, l, shape(), Val{dim})
                normals[k,l,i] = weight_norm / norm(weight_norm)
                detJ_f = norm(weight_norm)
                detJ_f > 0.0 || throw(ArgumentError("det(Jf) is not positive: det(Jf) = $(detJ_f)"))
                detJf_c[l, i] = detJ_f
            end
        end
        detJf[k] = detJf_c
    end
    MM = Tensors.n_components(Tensors.get_base(eltype(Jinv)))
    ScalarFunctionSpace{dim,T,NN,shape,dim1,order,MM}(N, dNdξ, E, detJ, detJf, Jinv,
    M, normals,getweights(quad_rule), q_ref_faceweights, q_ref_facepoints)
end

function weighted_normal(J::Tensor{2,2}, face::Int, ::RefTetrahedron, ::Type{Val{2}})
    @inbounds begin
        face == 1 && return Vec{2}((-(J[2,1] - J[2,2]), J[1,1] - J[1,2]))
        face == 2 && return Vec{2}((-J[2,2], J[1,2]))
        face == 3 && return Vec{2}((J[2,1], -J[1,1]))
    end
    throw(ArgumentError("unknown face number: $face"))
end

# Interpolated Functions
struct InterpolatedFunction{dim,T}
    N::Matrix{T}
end

@inline function_value(ifunc::InterpolatedFunction, cell::Int,q_point::Int) = ifunc.N[cell, q_point]

"""
function spatial_coordinate(fs::ScalarFunctionSpace{dim}, q_point::Int, x::AbstractVector{Vec{dim,T}})
Map coordinates of quadrature point `q_point` of Scalar Function Space `fs`
into domain with vertices `x`
"""
function spatial_coordinate(fs::ScalarFunctionSpace{dim}, q_point::Int, x::AbstractVector{Vec{dim,T}}) where {dim,T}
    n_base_funcs = getngeobasefunctions(fs)
    @assert length(x) == n_base_funcs
    vec = zero(Vec{dim,T})
    @inbounds for i in 1:n_base_funcs
        vec += geometric_value(fs, q_point, i) * x[i]
    end
    return vec
end

"""
function interpolate(f::Function, fs::ScalarFunctionSpace{dim,T}, mesh::PolygonalMesh) where {dim,T}
Interpolation of scalar functions f(x): x ∈ ℝⁿ ↦ ℝ on Scalar Function Space `fs`
"""
function interpolate(f::Function, fs::ScalarFunctionSpace{dim,T}, mesh::PolygonalMesh) where {dim,T}
    n_cells = numcells(mesh)
    n_qpoints = getnquadpoints(fs)
    N = fill(zero(T)          * T(NaN), n_cells, n_qpoints)
    for (k,cell) in enumerate(get_cells(mesh))
        coords = get_coordinates(cell, mesh)
        for i in 1:n_qpoints
            N[k,i] = f(spatial_coordinate(fs, i, coords))
        end
    end
    InterpolatedFunction{dim,T}(N)
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

function face_shape_value(fs::VectorFunctionSpace{dim,T}, face::Int, q_point::Int, base_func::Int, orientation::Bool=true) where {dim, T}
    @assert 1 <= base_func <= fs.n_dof "invalid base function index: $base_func"
    N_comp = zeros(T, dim)
    n = size(fs.ssp.N,1)
    q_p = orientation ? q_point : getnfacequadpoints(fs.ssp)-q_point + 1
    N_comp[div(base_func,n+1)+1] = fs.ssp.E[mod1(base_func,n), q_p][face]
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

@inline getnbasefunctions(fs::ScalarTraceFunctionSpace) = size(fs.N,1)
@inline getnquadpoints(fs::ScalarTraceFunctionSpace) = length(fs.qr_weights)
@inline getdetJdS(fs::ScalarTraceFunctionSpace{dim,T,2}, cell::Int, face::Int, q_point::Int) where {dim,T} = fs.detJ[cell,q_point][face]*fs.qr_weights[q_point]
@inline getdetJdS(fs::ScalarTraceFunctionSpace{dim,T,1}, cell::Int, face::Int, q_point::Int) where {dim,T} = fs.detJ[cell][face]*fs.qr_weights[q_point]
@inline shape_value(fs::ScalarTraceFunctionSpace, q_point::Int, base_func::Int) = fs.N[base_func, q_point]

spatial_coordinate(fs, i, coords, orientation)

"""
function spatial_coordinate(fs::ScalarTraceFunctionSpace{dim}, q_point::Int, x::AbstractVector{Vec{dim,T}}, orientation=true)
Map coordinates of quadrature point `q_point` of Scalar Trace Function Space `fs`
into domain with vertices `x`
"""
function spatial_coordinate(fs::ScalarTraceFunctionSpace{dim}, q_point::Int, x::AbstractVector{Vec{dim,T}}, orientation=true) where {dim,T}
    n_base_funcs = getngeobasefunctions(fs)
    @assert length(x) == n_base_funcs
    vec = zero(Vec{dim,T})
    @inbounds for i in 1:n_base_funcs
        vec += geometric_value(fs, q_point, i) * x[i]
    end
    return vec
end

function ScalarTraceFunctionSpace(psp::ScalarFunctionSpace{dim,T,N,refshape,order,M},
    func_interpol::Interpolation{dim1,refshape,order}) where {dim, T,N,refshape, order,M, dim1}
    @assert getdim(psp) == getdim(func_interpol)+1
    detJ = psp.detJf
    qr_weights = psp.qr_face_weigths
    qr_points = psp.qr_face_points
    ScalarTraceFunctionSpace(Float64, N, func_interpol, detJ, qr_weights, qr_points)
end

function ScalarTraceFunctionSpace(::Type{T}, NN, func_interpol::Interpolation{dim,refshape,order},
    detJ::Vector, qr_weights::Vector, qr_points::Vector) where {dim, refshape, order, T}
    n_qpoints = length(qr_weights)

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
