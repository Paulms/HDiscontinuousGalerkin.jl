# Abstract type for Polygonal Meshes
abstract type AbstractPolygonalMesh end
abstract type AbstractCell{dim,V,F} end

struct Node{dim,T}
    x::Vec{dim, T}
end

Node(x::NTuple{dim,T}) where {dim,T} = Node(Vec{dim,T}(x))

"""
get_coords(node::Node) = node.x
get coordinates of a node
"""
@inline get_coordinates(node::Node) = node.x

struct Cell{dim, N, M}
    nodes::NTuple{N,Int}
    faces::NTuple{M,Int}
end

#Common cell types
const TriangleCell = Cell{2,3,3}
@inline get_cell_name(::TriangleCell) = "Triangle"
@inline getnfaces(cell::Cell{dim,N,M}) where {dim,N,M} = M
function topology_elements(cell::Cell{2},element::Int)
    if element == 0
        return cell.nodes
    elseif element == 1
        return cell.faces
    else
        throw("Topology element of order $element not available for cell type")
    end
end

struct PolygonalMesh{dim,N,M,K,T} <: AbstractPolygonalMesh
    cells::Vector{Cell{dim,N,M}}
    nodes::Vector{Node{dim,T}}
    faces::Matrix{Int}
    facesets::Dict{String,Set{Int}}
end

function face_orientation(mesh::PolygonalMesh{2,3,3}, cell_idx::Int, face_idx::Int)
    k = reference_edge_nodes(RefTetrahedron, Val{2})[face_idx]
    return mesh.cells[cell_idx].nodes[k[2]] > mesh.cells[cell_idx].nodes[k[1]]
end

function get_vertices_matrix(mesh::PolygonalMesh{dim,N,M,K,T}) where {dim,N,M,K,T}
    nodes_m = Matrix{T}(length(mesh.nodes),dim)
    for (k,node) in enumerate(mesh.nodes)
        nodes_m[k,:] = node.x
    end
    nodes_m
end
function get_cells_matrix(mesh::PolygonalMesh{dim,N,M,K,T}) where {dim,N,M,K,T}
    cells_m = Matrix{Int}(getncells(mesh), n_faces_per_cell(mesh))
    for k = 1:getncells(mesh)
        @. cells_m[k,:] = mesh.cells[k].nodes - 1
    end
    cells_m
end
@inline n_faces_per_cell(mesh::PolygonalMesh{dim,N,M}) where {dim,N,M} = M
@inline n_nodes_per_cell(mesh::PolygonalMesh{dim,N,M}) where {dim,N,M} = N
@inline getnfaces(mesh::PolygonalMesh) = size(mesh.faces,1)
@inline getnnodes(mesh::PolygonalMesh) = length(mesh.nodes)
@inline getncells(mesh::PolygonalMesh) = length(mesh.cells)
@inline get_faceset(mesh::PolygonalMesh, set::String) = mesh.facesets[set]
"""
    getcoordinates(cell, mesh::PolygonalMesh)

Return a vector with the coordinates of the vertices of cell number `cell`.
"""
@inline function get_coordinates(cell::Cell, mesh::PolygonalMesh{dim,N,M,K,T}) where {dim,N,M,K,T}
    coords = Vector{Vec{dim,T}}(N)
    for (i,j) in enumerate(cell.nodes)
        coords[i] = mesh.nodes[j].x
    end
    return coords
end

@inline function get_coordinates!(coords::Vector{Vec{dim,T}}, cell::Cell, mesh::PolygonalMesh{dim,N,M,K,T}) where {dim,N,M,K,T}
    @assert length(coords) == N
    for (i,j) in enumerate(cell.nodes)
        coords[i] = mesh.nodes[j].x
    end
    return coords
end

@inline function get_cell_coordinates(cell_idx::Int, mesh::PolygonalMesh{dim,N,M,K,T}) where {dim,N,M,K,T}
    coords = Vector{Vec{dim,T}}(N)
    for (i,j) in enumerate(mesh.cells[cell_idx].nodes)
        coords[i] = mesh.nodes[j].x
    end
    return coords
end

@inline function get_cell_coordinates!(coords::Vector{Vec{dim,T}}, cell_idx::Int, mesh::PolygonalMesh{dim,N,M,K,T}) where {dim,N,M,K,T}
    @assert length(coords) == N
    for (i,j) in enumerate(mesh.cells[cell_idx].nodes)
        coords[i] = mesh.nodes[j].x
    end
    return coords
end
@inline function get_coordinates(face::Int, mesh::PolygonalMesh{dim,N,M,K,T}) where {dim,N,M,K,T}
    return [node.x for node in nodes(face,mesh)]::Vector{Vec{dim,T}}
end
@inline get_cells(mesh::PolygonalMesh) = mesh.cells
@inline get_nodes(mesh::PolygonalMesh) = mesh.nodes
@inline ndims(mesh::PolygonalMesh{dim}) where {dim} = dim
@inline nodes(ele::Cell, mesh::PolygonalMesh) = [mesh.nodes[node] for node in ele.nodes]
@inline nodes(face::Int, mesh::PolygonalMesh{dim,N,M,L}) where {dim,N,M,L} = [mesh.nodes[node] for node in mesh.faces[face,1:L]]
@inline cells(face::Int, mesh::PolygonalMesh{dim,N,M,L}) where {dim,N,M,L} = [mesh.cells[ele] for ele in mesh.faces[face,L+1:end]]
@inline getcell(cell::Int,face::Int, mesh::PolygonalMesh{dim,N,M,L}) where {dim,N,M,L} = mesh.faces[face,L+cell]
@inline getnode(node::Int,face::Int, mesh::PolygonalMesh{dim,N,M,L}) where {dim,N,M,L} = mesh.nodes[mesh.faces[face,node]]

function cell_diameter(mesh::PolygonalMesh{dim,N,M,NN,T}, idx::Int) where {dim,N,M,NN,T}
    K = get_cells(mesh)[idx]
    h = zero(T)
     for k in K.faces
        mσ = norm(get_coordinates(getnode(2, k, mesh)) - get_coordinates(getnode(1, k, mesh)))
        h = max(h, mσ)
    end
    h
end

_check_setname(dict, name) = haskey(dict, name) && throw(ArgumentError("there already exists a set with the name: $name"))
_warn_emptyset(set) = length(set) == 0 && warn("no entities added to set")

function addfaceset!(mesh::PolygonalMesh, name::String, faceid::Set{Int})
    _check_setname(mesh.facesets, name)
    faceset = Set(faceid)
    _warn_emptyset(faceset)
    mesh.facesets[name] = faceset
    mesh
end
