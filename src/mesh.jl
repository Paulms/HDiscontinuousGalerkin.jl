# Abstract type for Polygonal Meshes
abstract type AbstractPolygonalMesh end
abstract type AbstractCell{dim,V,F} end

struct Node{dim,T}
    x::Vec{dim, T}
    ref::Int
end

Node(x::NTuple{dim,T}, ref::Int) where {dim,T} = Node(Vec{dim,T}(x), ref)

"""
get_coords(node::Node) = node.x
get coordinates of a node
"""
@inline get_coordinates(node::Node) = node.x

struct Cell{dim, N, M, T,TI}
    nodes::NTuple{N,TI}
    faces::NTuple{M,TI}
    orientation::Vector{Bool}
    normals::Vector{Vec{dim,T}}
end

#Common cell types
const TriangleCell = Cell{2,3,3}
@inline get_cell_name(::TriangleCell) = "Triangle"

@inline getnfaces(cell::Cell{dim,N,M}) where {dim,N,M} = M
@inline get_normal(cell::Cell, face::Int) = cell.normals[face]
@inline face_orientation(cell::Cell, face::Int) = cell.orientation[face]
function topology_elements(cell::Cell{2},element::Int)
    if element == 0
        return cell.nodes
    elseif element == 1
        return cell.faces
    else
        throw("Topology element of order $element not available for cell type")
    end
end

struct Eddge{T <: Int}
    cells::Vector{T}
    nodes::Vector{T}
    ref::T  #To check if is belongs to the boundary
end

struct Face{T <: Int}
    cells::Vector{T}
    nodes::Vector{T}
    ref::T  #To check if is belongs to the boundary
end

struct PolygonalMesh{dim,N,M,T,TI} <: AbstractPolygonalMesh
    cells::Vector{Cell{dim,N,M,T,TI}}
    nodes::Vector{Node{dim,T}}
    faces::Vector{Face{TI}}
    facesets::Dict{String,Set{TI}}
end

function get_vertices_matrix(mesh::PolygonalMesh{dim,N,M,T,T1}) where {dim,N,M,T,T1}
    nodes_m = Matrix{T}(length(mesh.nodes),dim)
    for (k,node) in enumerate(mesh.nodes)
        nodes_m[k,:] = node.x
    end
    nodes_m
end
function get_cells_matrix(mesh::PolygonalMesh{dim,N,M,T,T1}) where {dim,N,M,T1,T}
    cells_m = Matrix{T1}(getncells(mesh), getncellfaces(mesh))
    for k = 1:getncells(mesh)
        @. cells_m[k,:] = mesh.cells[k].nodes - 1
    end
    cells_m
end
@inline getncellfaces(mesh::PolygonalMesh{dim,N,M}) where {dim,N,M} = M
@inline getnfaces(mesh::PolygonalMesh) = length(mesh.faces)
@inline getnnodes(mesh::PolygonalMesh) = length(mesh.nodes)
@inline get_faceset(mesh::PolygonalMesh, set::String) = mesh.facesets[set]
@inline get_coordinates(cell::Cell, mesh::PolygonalMesh) = [mesh.nodes[j].x for j in cell.nodes]
@inline get_coordinates(face::Face, mesh::PolygonalMesh) = [mesh.nodes[j].x for j in face.nodes]
@inline get_cells(mesh::PolygonalMesh) = mesh.cells
@inline get_nodes(mesh::PolygonalMesh) = mesh.nodes
@inline get_faces(mesh::PolygonalMesh) = mesh.faces
@inline get_faces(cell::Cell, mesh::PolygonalMesh) = [mesh.faces[i] for i in cell.faces]
@inline node(idx::Int, face::Face, mesh::PolygonalMesh) = mesh.nodes[face.nodes[idx]]
@inline node(idx::Int, ele::Cell, mesh::PolygonalMesh) = mesh.nodes[ele.nodes[idx]]
@inline ndims(mesh::PolygonalMesh{dim}) where {dim} = dim
@inline getncells(mesh::PolygonalMesh) = length(mesh.cells)

@inline nodes(ele::Cell, mesh::PolygonalMesh) = [mesh.nodes[node] for node in ele.nodes]
@inline nodes(face::Face, mesh::PolygonalMesh) = [mesh.nodes[node] for node in face.nodes]
@inline faces(ele::Cell, mesh::PolygonalMesh) = [mesh.faces[face] for face in ele.faces]
@inline cells(face::Face, mesh::PolygonalMesh) = [mesh.cells[ele] for ele in face.cells]

function cell_diameter(mesh::PolygonalMesh{dim,N,M,T}, idx::Int) where {dim,N,M,T}
    K = get_cells(mesh)[idx]
    h = zero(T)
     for k in K.faces
        σ = mesh.faces[k]
        mσ = norm(get_coordinates(node(2, σ, mesh)) - get_coordinates(node(1, σ, mesh)))
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
