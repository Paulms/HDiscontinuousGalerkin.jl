# Apply Dirichlet boundary condition
struct Dirichlet{T}  #<: Constraint
    prescribed_dofs::Vector{Int}
    values::Vector{T}
end

function Dirichlet(u::TrialFunction, mesh::PolygonalMesh, faceset::String,f::Function)
    Dirichlet(getfunctionspace(u), mesh, get_faceset(mesh, faceset), f)
end

function Dirichlet(fs::ScalarTraceFunctionSpace{2,T}, mesh::PolygonalMesh, faceset::Set{Int},f::Function) where {T}
    #Project function on faces
    n_dof = getnbasefunctions(fs)
    n_qpoints = getnquadpoints(fs)
    n_faces = length(faceset)
    prescribed_dofs = Vector{Int}(n_faces*n_dof)
    values = Vector{T}(n_faces*n_dof)
    k = 0
    for (face_idx, face) in enumerate(get_faces(mesh))
        let face_idx::Int = face_idx
        if face_idx ∈ faceset
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
    end
    return Dirichlet(prescribed_dofs, values)
end

function Dirichlet(u::TrialFunction, dh::DofHandler, faceset::String,f::Function)
    Dirichlet(u, dh, get_faceset(dh.mesh, faceset), f)
end

#TODO: only works for nodal basis
function Dirichlet(field::TrialFunction{2,T,refshape}, dh::DofHandler, faceset::Set{Int}, f::Function) where {T,refshape}
    fi = find_field(dh, field)
    fs = getfunctionspace(field)
    #compute total number of dofs
    n_qpoints = getnquadpoints(fs)
    n_max_topology_elements = maximum(keys(get_topology(refshape, Val{2})))
    nelementdofs = Int[]
    ncellelementdofs = Int[]
    # Compute dofs for element and total element dofs for cell
    for n_element in 0:n_max_topology_elements-1
        n_el_dofs_cell = get_topology(get_interpolation(fs))[n_element]
        n_el_cell = get_topology(getrefshape(field), Val{2})[n_element]
        @assert mod(n_el_dofs_cell, n_el_cell) == 0
        push!(nelementdofs,Int(n_el_dofs_cell/n_el_cell))
        push!(ncellelementdofs,n_el_dofs_cell)
    end

    prescribed_dofs = Vector{Int}()
    values = Vector{T}()
    M = _get_nodal_transformation_matrix(get_interpolation(fs))
    for (face_idx, face) in enumerate(get_faces(dh.mesh))
        if face_idx ∈ faceset
            @assert length(face.cells) == 1 "Face $face_idx is not in boundary"
            cell = dh.mesh.cells[face.cells[1]]
            cell_idx = face.cells[1]
            face_lidx = find(x -> x == face_idx,cell.faces)[1]
            l_dof = Int[]
            offset::Int = dh.cell_dofs_offset[cell_idx] - 1 + field_offset(dh, field)
            for j = 1:2 #Add vertex dofs
                local_offset::Int = reference_edge_nodes(refshape,Val{2})[face_lidx][j]
                if !(dh.cell_dofs[offset+local_offset] ∈ prescribed_dofs)
                    push!(prescribed_dofs, dh.cell_dofs[offset+local_offset])
                    push!(l_dof, local_offset)
                end
            end
            for j in 1:nelementdofs[2]  #Add face dofs
                local_offset = ncellelementdofs[1] + nelementdofs[2]*(facel_lidx-1) + j
                if !(dh.cell_dofs[offset+local_offset] ∈ prescribed_dofs)
                    push!(prescribed_dofs, dh.cell_dofs[offset+local_offset])
                    push!(l_dof, local_offset)
                end
            end
            #TODO: We now assume a nodal base...
            orientation = face_orientation(cell, face_lidx)
            coords = get_coordinates(cell, dh.mesh)
            for i in l_dof
                push!(values,f(spatial_nodal_coordinate(get_interpolation(fs),M,i,coords)))
            end
        end
    end
    #now put all in order
    p = sortperm(prescribed_dofs)
    return Dirichlet(prescribed_dofs[p], values[p])
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