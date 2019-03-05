# this file defines iterators used for looping over a grid
struct UpdateFlags
    nodes::Bool
    coords::Bool
end

UpdateFlags(; nodes::Bool=true, coords::Bool=true) =
    UpdateFlags(nodes, coords)

struct CellIterator{dim,N1,N2,N3,T}
    flags::UpdateFlags
    mesh::PolygonalMesh{dim,N1,N2,N3,T}
    current_cellid::ScalarWrapper{Int}
    nodes::Vector{Int}
    coords::Vector{Vec{dim,T}}
    function CellIterator{dim,N1,N2,N3,T}(mesh::PolygonalMesh{dim,N1,N2,N3,T}, flags::UpdateFlags) where {dim,N1,N2,N3,T}
        cell = ScalarWrapper(0)
        nodes = zeros(Int, N1)
        coords = zeros(Vec{dim,T}, N1)
        return new{dim,N1,N2,N3,T}(flags, mesh, cell, nodes, coords)
    end
end

CellIterator(mesh::PolygonalMesh{dim,N1,N2,N3,T},     flags::UpdateFlags=UpdateFlags()) where {dim,N1,N2,N3,T} =
    CellIterator{dim,N1,N2,N3,T}(mesh, flags)

# iterator interface
function Base.iterate(ci::CellIterator, state = 1)
    if state > getncells(ci.mesh)
        return nothing
    else
        return (reinit!(ci, state), state+1)
    end
end
Base.length(ci::CellIterator)  = getncells(ci.mesh)

Base.IteratorSize(::Type{T})   where {T<:CellIterator} = Base.HasLength() # this is default in Base
Base.IteratorEltype(::Type{T}) where {T<:CellIterator} = Base.HasEltype() # this is default in Base
Base.eltype(::Type{T})         where {T<:CellIterator} = T

# utility
@inline getnodes(ci::CellIterator) = ci.nodes
@inline getcoordinates(ci::CellIterator) = ci.coords
@inline nfaces(ci::CellIterator) = nfaces(eltype(ci.mesh.cells))
@inline cellid(ci::CellIterator) = ci.current_cellid[]
@inline celldofs!(v::Vector, dh::DofHandler, ci::CellIterator) = celldofs!(v, dh, ci.current_cellid[])

function reinit!(ci::CellIterator{dim,N}, i::Int) where {dim,N}
    nodeids = ci.mesh.cells[i].nodes
    ci.current_cellid[] = i
    @inbounds for j in 1:N
        nodeid = nodeids[j]
        ci.flags.nodes  && (ci.nodes[j] = nodeid)
        ci.flags.coords && (ci.coords[j] = ci.mesh.nodes[nodeid].x)
    end
    return ci
end

@inline reinit!(fs::DiscreteFunctionSpace{dim,T,FE}, ci::CellIterator{dim,N1,N2,N3,T}) where {dim,N1,N2,N3,T,FE} = reinit!(fs, ci.coords)
@inline reinit!(u::TrialFunction{dim,T}, ci::CellIterator{dim,N1,N2,N3,T}) where {dim,N1,N2,N3,T} = reinit!(u.fs, ci.coords)
function_value(f::Function, fs::DiscreteFunctionSpace{dim,T,FE}, ci::CellIterator,q_point::Int) where {dim,T,FE} = function_value(f,fs, ci.current_cellid[],q_point)
@inline getnfaces(ci::CellIterator{dim,N}) where {dim,N} = getnfaces(ci.mesh.cells[ci.current_cellid[]])
@inline getnodes(ci::CellIterator{dim,N}) where {dim,N} =  ci.mesh.cells[ci.current_cellid[]].nodes
value(u_h::TrialFunction{dim,T}, node::Int, ci::CellIterator{dim,N}) where {dim, T, N} = value(u_h, node, ci.current_cellid[])
@inline face_orientation(ci::CellIterator{dim,N}, face_idx::Int) where {dim,N} = face_orientation(ci.mesh, ci.current_cellid[], face_idx)
