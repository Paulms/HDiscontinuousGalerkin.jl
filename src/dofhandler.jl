"""
    DofHandler(grid::Grid)

Construct a `DofHandler` based on the grid `grid`.
"""
struct DofHandler{dim,T}
    variables::Vector{TrialFunction}
    cell_dofs::Vector{Int}
    cell_dofs_offset::Vector{Int}
    closed::ScalarWrapper{Bool}
    mesh::PolygonalMesh{dim,T}
end

function DofHandler(mesh::PolygonalMesh)
    DofHandler(TrialFunction[], Int[], Int[], ScalarWrapper(false), mesh)
end

# macro VarName(arg)
#           string(arg)
# end
# function Base.show(io::IO, dh::DofHandler)
#     println(io, "DofHandler")
#     println(io, "  Fields:")
#     for i in 1:nvariables(dh)
#         println(io, "    ", repr(@Name dh.variables[i]), ", interpolation: ", dh.field_interpolations[i],", dim: ", dh.field_dims[i])
#     end
#     if !isclosed(dh)
#         print(io, "  Not closed!")
#     else
#         println(io, "  Dofs per cell: ", ndofs_per_cell(dh))
#         print(io, "  Total dofs: ", ndofs(dh))
#     end
# end

ndofs(dh::DofHandler) = maximum(dh.cell_dofs)
ndofs_per_cell(dh::DofHandler, cell::Int=1) = dh.cell_dofs_offset[cell+1] - dh.cell_dofs_offset[cell]
isclosed(dh::DofHandler) = dh.closed[]
@inline nvariables(dh::DofHandler) = length(dh.variables)

function find_field(dh::DofHandler, field::TrialFunction)
    j = findfirst(i->i == field, dh.variables)
    j == 0 && error("did not find field $field")
    return j
end

# Calculate the offset to the first local dof of a field
function field_offset(dh::DofHandler, field::TrialFunction)
    offset = 0
    for i in 1:find_field(dh, field)-1
        offset += getnlocaldofs(dh.variables[i])::Int
    end
    return offset
end

"""
    dof_range(dh:DofHandler, field_name)

Return the local dof range for `field_name`. Example:

```jldoctest
julia> grid = generate_grid(Triangle, (3, 3));

julia> dh = DofHandler(grid); push!(dh, :u, 3); push!(dh, :p, 1); close!(dh);

julia> dof_range(dh, :u)
1:9

julia> dof_range(dh, :p)
10:12
```
"""
function dof_range(dh::DofHandler, field::TrialFunction)
    f = find_field(dh, field)
    offset = field_offset(dh, field)
    n_field_dofs = getnlocaldofs(dh.variables[f])::Int
    return (offset+1):(offset+n_field_dofs)
end

function Base.push!(dh::DofHandler, u::TrialFunction)
    @assert !isclosed(dh)
    @assert !in(u, dh.variables)
    push!(dh.variables, u)
    return dh
end

# close the DofHandler and distribute all the dofs
#TODO: This only work for D < 3
function close!(dh::DofHandler{dim}) where {dim}
    @assert !isclosed(dh)

    # `vertexdict` keeps track of the visited vertices. We store the global vertex
    # number and the first dof we added to that vertex.
    vertexdicts = [Dict{Int,Int}() for _ in 1:nvariables(dh)]

    # `edgedict` keeps track of the visited edges, this will only be used for a 3D problem
    # We also need to store the direction
    # of the first edge we encounter and add dofs too. When we encounter the same edge
    # the next time we check if the direction is the same, otherwise we reuse the dofs
    # in the reverse order
    #edgedicts = [Dict{Int,Int}() for _ in 1:nvariables(dh)]

    # `facedict` keeps track of the visited faces. We only need to store the first dof we
    # added to the face; if we encounter the same face again we *always* reverse the order
    # In 2D a face (i.e. a line) is uniquely determined by 2 vertices, and in 3D a
    # face (i.e. a surface) is uniquely determined by 3 vertices.
    facedicts = [Dict{Int,Int}() for _ in 1:nvariables(dh)]

    topologyDicts = Dict(0=>vertexdicts, 1=>facedicts)  #TODO: only valid in 2D

    nextdof = 1 # next free dof to distribute
    push!(dh.cell_dofs_offset, 1) # dofs for the first cell start at 1

    # loop over all the cells, and distribute dofs for all the fields
    for (ci, cell) in enumerate(get_cells(dh.mesh))
        for fi in 1:nvariables(dh)
            n_max_topology_elements = maximum(keys(get_topology(getrefshape(dh.variables[fi]), Val{dim})))
            for n_element in 0:n_max_topology_elements-1
                n_el_dofs_cell = get_topology(get_interpolation(getfunctionspace(dh.variables[fi])))[n_element]
                n_el_cell = get_topology(getrefshape(dh.variables[fi]), Val{dim})[n_element]
                @assert mod(n_el_dofs_cell, n_el_cell) == 0
                nelementdofs = Int(n_el_dofs_cell/n_el_cell)
                if nelementdofs > 0
                    for element in topology_elements(cell,n_element)
                        token = ht_keyindex2!(topologyDicts[n_element][fi], element)
                        if token > 0 # haskey(vertexdicts[fi], vertex) # reuse dofs
                            reuse_dof = topologyDicts[n_element][fi].vals[token] # vertexdicts[fi][vertex]
                            for d in 1:getncomponents(dh.variables[fi])
                                push!(dh.cell_dofs, reuse_dof + (d-1))
                            end
                        else # token <= 0, distribute new dofs
                            for elementdof in 1:nelementdofs
                                Base._setindex!(topologyDicts[n_element][fi], nextdof, element, -token) # vertexdicts[fi][vertex] = nextdof
                                for d in 1:getncomponents(dh.variables[fi])
                                    push!(dh.cell_dofs, nextdof)
                                    nextdof += 1
                                end
                            end
                        end
                    end # vertex loop
                end
            end
            ncelldofs = get_topology(get_interpolation(getfunctionspace(dh.variables[fi])))[dim]
            if ncelldofs > 0 # always distribute new dofs for cell
                for celldof in 1:ncelldofs
                    for d in 1:getncomponents(dh.variables[fi])
                        push!(dh.cell_dofs, nextdof)
                        nextdof += 1
                    end
                end # cell loop
            end
        end # field loop
        # push! the first index of the next cell to the offset vector
        push!(dh.cell_dofs_offset, length(dh.cell_dofs)+1)
    end # cell loop
    dh.closed[] = true
    return dh
end

function celldofs!(global_dofs::Vector{Int}, dh::DofHandler, i::Int)
    @assert isclosed(dh)
    @assert length(global_dofs) == ndofs_per_cell(dh, i)
    unsafe_copy!(global_dofs, 1, dh.cell_dofs, dh.cell_dofs_offset[i], length(global_dofs))
    return global_dofs
end

function celldofs(dh::DofHandler, i::Int)
    @assert isclosed(dh)
    unsafe_copy!(global_dofs, 1, dh.cell_dofs, dh.cell_dofs_offset[i], length(global_dofs))
    return dh.cell_dofs[dh.cell_dofs_offset[i]:dh.cell_dofs, dh.cell_dofs_offset[i]+ndofs_per_cell(dh, i)]
end

# Creates a sparsity pattern from the dofs in a DofHandler.
# Returns a sparse matrix with the correct storage pattern.
"""
    create_sparsity_pattern(dh::DofHandler)

Create the sparsity pattern corresponding to the degree of freedom
numbering in the [`DofHandler`](@ref). Return a `SparseMatrixCSC`
with stored values in the correct places.

See the [Sparsity Pattern](@ref) section of the manual.
"""
@inline create_sparsity_pattern(dh::DofHandler) = _create_sparsity_pattern(dh, false)

"""
    create_symmetric_sparsity_pattern(dh::DofHandler)

Create the symmetric sparsity pattern corresponding to the degree of freedom
numbering in the [`DofHandler`](@ref) by only considering the upper
triangle of the matrix. Return a `Symmetric{SparseMatrixCSC}`.

See the [Sparsity Pattern](@ref) section of the manual.
"""
@inline create_symmetric_sparsity_pattern(dh::DofHandler) = Symmetric(_create_sparsity_pattern(dh, true), :U)

function _create_sparsity_pattern(dh::DofHandler, sym::Bool)
    ncells = getncells(dh.mesh)
    n = ndofs_per_cell(dh)
    N = sym ? div(n*(n+1), 2) * ncells : n^2 * ncells
    N += ndofs(dh) # always add the diagonal elements
    I = Int[]; resize!(I, N)
    J = Int[]; resize!(J, N)
    global_dofs = zeros(Int, n)
    cnt = 0
    for element_id in 1:ncells
        celldofs!(global_dofs, dh, element_id)
        @inbounds for j in 1:n, i in 1:n
            dofi = global_dofs[i]
            dofj = global_dofs[j]
            sym && (dofi > dofj && continue)
            cnt += 1
            if cnt > length(J)
                resize!(I, trunc(Int, length(I) * 1.5))
                resize!(J, trunc(Int, length(J) * 1.5))
            end
            I[cnt] = dofi
            J[cnt] = dofj

        end
    end
    @inbounds for d in 1:ndofs(dh)
        cnt += 1
        if cnt > length(J)
            resize!(I, trunc(Int, length(I) + ndofs(dh)))
            resize!(J, trunc(Int, length(J) + ndofs(dh)))
        end
        I[cnt] = d
        J[cnt] = d
    end
    resize!(I, cnt)
    resize!(J, cnt)
    V = zeros(length(I))
    K = sparse(I, J, V)
    return K
end

# Reconstruct TrialFunctions
function reconstruct!(field::TrialFunction, u::AbstractVector{T}, dh::DofHandler) where {T}
    for cell_idx in 1:getncells(dh.mesh)
        offset = field_offset(dh,field) + dh.cell_dofs_offset[cell_idx]
        n_dofs = offset + getnlocaldofs(field) - 1
        field.m_values[cell_idx,:] = u[dh.cell_dofs[offset:n_dofs]]
    end
end

# dof renumbering
"""
    renumber!(dh::DofHandler, perm)

Renumber the degrees of freedom in the DofHandler according to the
permuation `perm`.

!!! warning
    Remember to do renumbering *before* adding boundary conditions,
    otherwise the mapping for the dofs will be wrong.
"""
function renumber!(dh::DofHandler, perm::AbstractVector{<:Integer})
    @assert isperm(perm) && length(perm) == ndofs(dh)
    cell_dofs = dh.cell_dofs
    for i in eachindex(cell_dofs)
        cell_dofs[i] = perm[cell_dofs[i]]
    end
    return dh
end
