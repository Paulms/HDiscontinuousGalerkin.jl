# Apply Dirichlet boundary condition
struct Dirichlet{T}  #<: Constraint
    prescribed_dofs::Vector{Int}
    values::Vector{T}
end

function Dirichlet(fs::DiscreteFunctionSpace, mesh::PolygonalMesh, faceset::String,f::Function)
    Dirichlet(fs, mesh, get_faceset(mesh, faceset), f)
end

function Dirichlet(fs::ScalarTraceFunctionSpace{1,T}, mesh::PolygonalMesh, faceset::Set{Int},f::Function) where {T}
    #Project function on faces
    n_dof = getnbasefunctions(fs)
    n_qpoints = getnquadpoints(fs)
    n_faces = length(faceset)
    prescribed_dofs = Vector{Int}(n_faces*n_dof)
    values = Vector{T}(n_faces*n_dof)
    k = 0
    for (face_idx, face) in enumerate(get_faces(mesh))
        if face_idx ∈ faceset
            println(face_idx)
            @assert length(face.cells) == 1 "Face $face_idx is not in boundary"
            cell = mesh.cells[face.cells[1]]
            face_lidx = find(x -> x == face_idx,cell.faces)[1]
            orientation = face_orientation(cell, face_lidx)
            N = zero(T)
            coords = get_coordinates(cell, mesh)
            #Solve system ∫ k ϕiϕj = ∫ f ϕi  we assume basis is orthogonal
            for i in 1:n_dof
                k += 1
                for q_point in 1:n_qpoints
                    N += fs.qr_weights[q_point]*f(spatial_coordinate(fs, face_lidx, q_point, coords, orientation))*shape_value(fs,q_point,i)
                end
                values[k] = N
                prescribed_dofs[k] = face_idx*n_dof-n_dof + i
            end
        end
    end
    return Dirichlet(prescribed_dofs, values)
end

@enum(ApplyStrategy, APPLY_TRANSPOSE, APPLY_INPLACE)

function apply!(KK::Union{SparseMatrixCSC,Symmetric}, f::AbstractVector, dirichlet::Dirichlet;
                strategy::ApplyStrategy=APPLY_TRANSPOSE)
    K = isa(KK, Symmetric) ? KK.data : KK
    @assert length(f) == 0 || length(f) == size(K, 1)
    @boundscheck checkbounds(K, dirichlet.prescribed_dofs, dirichlet.prescribed_dofs)
    @boundscheck length(f) == 0 || checkbounds(f, dirichlet.prescribed_dofs)

    m = meandiag(K) # Use the mean of the diagonal here to not ruin things for iterative solver
    @inbounds for i in 1:length(dirichlet.values)
        d = dirichlet.prescribed_dofs[i]
        v = dirichlet.values[i]

        if v != 0
            for j in nzrange(K, d)
                f[K.rowval[j]] -= v * K.nzval[j]
            end
        end
    end
    zero_out_columns!(K, dirichlet.prescribed_dofs)
    if strategy == APPLY_TRANSPOSE
        K′ = copy(K)
        transpose!(K′, K)
        zero_out_columns!(K′, dirichlet.prescribed_dofs)
        transpose!(K, K′)
    elseif strategy == APPLY_INPLACE
        K[dirichlet.prescribed_dofs, :] = 0
    else
        error("Unknown apply strategy")
    end
    @inbounds for i in 1:length(dirichlet.values)
        d = dirichlet.prescribed_dofs[i]
        v = dirichlet.values[i]
        K[d, d] = m
        if length(f) != 0
            f[d] = v * m
        end
    end
end

# columns need to be stored entries, this is not checked
function zero_out_columns!(K, dofs::Vector{Int}) # can be removed in 0.7 with #24711 merged
    #@debug assert(issorted(dofs))
    for col in dofs
        r = nzrange(K, col)
        K.nzval[r] = 0.0
    end
end

function meandiag(K::AbstractMatrix)
    z = zero(eltype(K))
    for i in 1:size(K, 1)
        z += abs(K[i, i])
    end
    return z / size(K, 1)
end
