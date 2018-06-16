# Abstract type for Polygonal Meshes
abstract type AbstractPolygonalMesh end

struct Node{N,T}
    x::Vec{N, T}
    ref::Int
end
Node(x::NTuple{dim,T}, ref::Int) where {dim,T} = Node(Vec{dim,T}(x), ref)

"""
get_coords(node::Node) = node.x
get coordinates of a node
"""
@inline get_coordinates(node::Node) = node.x

struct Cell{T <: Int,N,T2}
    nodes::Vector{T}
    faces::Vector{T}
    orientation::Vector{Bool}
    normals::Vector{Vec{N,T2}}
end
@inline numfaces(cell::Cell) = length(cell.faces)
@inline get_normal(cell::Cell, face::Int) = cell.normals[face]
@inline face_orientation(cell::Cell, face::Int) = cell.orientation[face]
function topology_elements(cell::Cell,element::Int)
    if element == 0
        return cell.nodes
    elseif element == 1
        return cell.faces    #rename as edges
    else
        throw("Topology element of order $element not available for current mesh")
    end
end

struct Face{T <: Int}
    cells::Vector{T}
    nodes::Vector{T}
    ref::Int64  #To check if is belongs to the boundary
end
struct PolygonalMesh{dims,Type} <: AbstractPolygonalMesh
    cells::Vector{Cell}
    nodes::Vector{Node{dims,Type}}
    faces::Vector{Face}
    facesets::Dict{String,Set{Int}}
end
function get_maxnfaces(mesh::PolygonalMesh)
    nmax = 0
    for cell in mesh.cells
        nmax = max(nmax, numfaces(cell))
    end
    nmax
end
@inline numfaces(mesh::PolygonalMesh) = length(mesh.faces)
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
@inline ndims(mesh::PolygonalMesh{dims,Type}) where {dims,Type} = dims
@inline getncells(mesh::PolygonalMesh) = length(mesh.cells)

@inline nodes(ele::Cell, mesh::PolygonalMesh{dims,Type}) where {dims,Type} = [mesh.nodes[node] for node in ele.nodes]
@inline nodes(face::Face, mesh::PolygonalMesh{dims,Type}) where {dims,Type} = [mesh.nodes[node] for node in face.nodes]
@inline faces(ele::Cell, mesh::PolygonalMesh) = [mesh.faces[face] for face in ele.faces]
@inline cells(face::Face, mesh::PolygonalMesh) = [mesh.cells[ele] for ele in face.cells]

function cell_diameter(mesh::PolygonalMesh{N,T}, idx::Int) where {N,T}
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
