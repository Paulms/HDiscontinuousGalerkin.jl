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
@inline numcells(mesh::PolygonalMesh) = length(mesh.cells)

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

# Read mesh from a triangle generated file
function read_line(ln, types)
    m2 = matchall(r"\b((\d*\.)?\d+)\b", ln)
    return (parse(types[i],x) for (i,x) in enumerate(m2))
end

function parse_nodes!(nodes,root_file)
    open(root_file*".node") do f
        first_line = true
        for ln in eachline(f)
            m = match(r"^\s*(?:#|$)", ln)
            if m == nothing
                if (first_line)   #skip first line
                    first_line = false
                else
                    #parse nodes
                    pln = collect(read_line(ln, (Int,Float64,Float64,Int)))
                    node = Node((pln[2],pln[3]),pln[4])
                    push!(nodes,node)
                end
            end
        end
    end
end

function parse_faces!(faces,root_file)
    open(root_file*".edge") do f
        first_line = true
        for ln in eachline(f)
            m = match(r"^\s*(?:#|$)", ln)
            if m == nothing
                if (first_line)   #skip first line
                    first_line = false
                else
                    #parse nodes
                    pln = collect(read_line(ln, (Int,Int,Int,Int)))
                    face = Face(Vector{Int}(),pln[2:3],pln[4])
                    push!(faces, face)
                end
            end
        end
    end
end

function parse_cells!(cells, faces, faces_unorded,facesets, nodes, root_file)
    #read cell nodes
    open(root_file*".ele") do f
        first_line = true
        n_el = 0
        n_faces = 0
        boundary_faces = Set{Int}()
        for ln in eachline(f)
            m = match(r"^\s*(?:#|$)", ln)
            if m == nothing
                if (first_line)   #skip first line
                    first_line = false
                else
                    n_el = n_el + 1
                    #parse nodes
                    pln = collect(read_line(ln, (Int,Int,Int,Int)))
                    el_nodes = pln[2:4]
                    el_faces = [-1,-1,-1]
                    #build faces
                    for (i,fn_id) in enumerate(((2,3),(3,1),(1,2)))
                        c_face = [el_nodes[fn_id[1]],el_nodes[fn_id[2]]]
                        face_found = false
                        for (j,face) in enumerate(faces)
                            if sort(face.nodes) == sort(c_face)
                                el_faces[i] = j
                                if !(n_el in face.cells)
                                    push!(face.cells,n_el)
                                end
                                face_found = true
                            end
                        end
                        if !face_found
                            ref = -1
                            for (j,face) in enumerate(faces_unorded)
                                if sort(face.nodes) == sort(c_face)
                                    ref = face.ref
                                end
                            end
                            face = Face([n_el],c_face,ref)
                            push!(faces, face)
                            n_faces = n_faces + 1
                            el_faces[i] = n_faces
                            if ref > 0
                                push!(boundary_faces, n_faces)
                            end
                        end
                    end
                    orientation = [true,true,true]
                    normals = fill(zero(Vec{2,Float64}) * Float64(NaN), 3)
                    #check consistent cells orientation and compute normals
                    _build_face_data(nodes, el_nodes, el_faces, normals, orientation)

                    #save cell
                    cell = Cell(el_nodes, el_faces, orientation, normals)
                    push!(cells, cell)
                end
            end
        end
        push!(facesets, "boundary" => boundary_faces)
    end
end

"""
function parse_mesh_triangle(root_file)
read mesh generated by triangle with root file name `root_file`
Ex: `parse_mesh_triangle("figure.1")`
"""
function parse_mesh_triangle(root_file)
    nodes = Vector{Node}()
    faces_unorded = Vector{Face}()
    faces = Vector{Face}()
    cells = Vector{Cell}()
    facesets = Dict{String,Set{Int}}()
    parse_nodes!(nodes,root_file)
    parse_faces!(faces_unorded, root_file)
    parse_cells!(cells, faces, faces_unorded,facesets,nodes, root_file)
    PolygonalMesh{size(nodes[1].x,1),eltype(nodes[1].x)}(cells, nodes, faces, facesets)
end
