# Apply Dirichlet boundary condition
struct Dirichlet # <: Constraint
    f::Function # f(x,t) -> value
    faces::Set{Int}
end
function Dirichlet(fs::DiscreteFunctionSpace, mesh::PolygonalMesh, faceset::String,f::Function)
    Dirichlet(fs, mesh, get_faceset(mesh, faceset), f)
end

function Dirichlet(fs::ScalarTraceFunctionSpace{1,T}, mesh::PolygonalMesh, faceset::Set{Int},f::Function) where {T}
    #Project function on faces
    n_dof = getnbasefunctions(fs)
    n_qpoints = getnquadpoints(fs)
    n_faces = length(faceset)

    for (face_idx, face) in enumerate(get_faces(mesh))
        if face_idx ∈ faceset
            @assert length(face.cells) == 1 "Face is not in boundary"
            cell = mesh.cells(face.cells[1])
            face_lidx = find(x -> x == face_idx,cell.faces)[1]
            orientation = face_orientation(cell, face_lidx)
            N = fill(zero(T)          * T(NaN), n_qpoints)
            coords = get_coordinates(face, mesh)
            for i in 1:n_qpoints
                N[i] = f(spatial_coordinate(fs, i, coords, orientation))
            end
            end
        end
    end
    return Dirichlet(f, faces)
end

function apply!(KK::Union{SparseMatrixCSC,Symmetric}, f::AbstractVector, ch::ConstraintHandler, applyzero::Bool=false;
                strategy::ApplyStrategy=APPLY_TRANSPOSE)
    K = isa(KK, Symmetric) ? KK.data : KK
    @assert length(f) == 0 || length(f) == size(K, 1)
    @boundscheck checkbounds(K, ch.prescribed_dofs, ch.prescribed_dofs)
    @boundscheck length(f) == 0 || checkbounds(f, ch.prescribed_dofs)

    m = meandiag(K) # Use the mean of the diagonal here to not ruin things for iterative solver
    @inbounds for i in 1:length(ch.values)
        d = ch.prescribed_dofs[i]
        v = ch.values[i]

        if !applyzero && v != 0
            for j in nzrange(K, d)
                f[K.rowval[j]] -= v * K.nzval[j]
            end
        end
    end
    zero_out_columns!(K, ch.prescribed_dofs)
    if strategy == APPLY_TRANSPOSE
        K′ = copy(K)
        transpose!(K′, K)
        zero_out_columns!(K′, ch.prescribed_dofs)
        transpose!(K, K′)
    elseif strategy == APPLY_INPLACE
        K[ch.prescribed_dofs, :] = 0
    else
        error("Unknown apply strategy")
    end
    @inbounds for i in 1:length(ch.values)
        d = ch.prescribed_dofs[i]
        v = ch.values[i]
        K[d, d] = m
        # We will only enter here with an empty f vector if we have assured that v == 0 for all dofs
        if length(f) != 0
            vz = applyzero ? zero(eltype(f)) : v
            f[d] = vz * m
        end
    end
end
