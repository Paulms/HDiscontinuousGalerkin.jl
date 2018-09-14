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
value(ifunc::InterpolatedFunction, ci::CellIterator,q_point::Int) = value(ifunc, ci.current_cellid[],q_point)
