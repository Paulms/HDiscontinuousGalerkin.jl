# Abstract type for Polygonal Meshes
using Tensors

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
get_coordinates(node::Node) = node.x

struct Cell{T <: Int,N,T2}
    nodes::Vector{T}
    faces::Vector{T}
    orientation::Vector{Bool}
    normals::Vector{Vec{N,T2}}
end
@inline numfaces(cell::Cell) = length(cell.faces)

struct Face{T <: Int}
    cells::Vector{T}
    nodes::Vector{T}
    ref::Int64  #To check if is belongs to the boundary
end
struct PolygonalMesh{dims,Type} <: AbstractPolygonalMesh
    cells::Vector{Cell}
    nodes::Vector{Node{dims,Type}}
    faces::Vector{Face}
end
function get_maxnfaces(mesh::PolygonalMesh)
    nmax = 0
    for cell in mesh.cells
        nmax = max(nmax, numfaces(cell))
    end
    nmax
end
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

nodes(ele::Cell, mesh::PolygonalMesh{dims,Type}) where {dims,Type} = [mesh.nodes[node] for node in ele.nodes]
nodes(face::Face, mesh::PolygonalMesh{dims,Type}) where {dims,Type} = [mesh.nodes[node] for node in face.nodes]
faces(ele::Cell, mesh::PolygonalMesh) = [mesh.faces[face] for face in ele.faces]
cells(face::Face, mesh::PolygonalMesh) = [mesh.cells[ele] for ele in face.cells]

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

function parse_cells!(cells,faces,root_file)
    all_nodes = Vector{Vector{Node}}()
    #read cell nodes
    open(root_file*".ele") do f
        first_line = true
        n_el = 0
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
                    #look for faces
                    for (i,c_face) in enumerate([el_nodes[1:2],el_nodes[2:3],
                        [el_nodes[3],el_nodes[1]]])
                        for (j,face) in enumerate(faces)
                            if sort(face.nodes) == sort(c_face)
                                el_faces[i] = j
                                if !(n_el in face.cells)
                                    push!(face.cells,n_el)
                                end
                            end
                        end
                    end
                    orientation = [true,true,true]
                    normals = fill(zero(Vec{2,Float64}) * Float64(NaN), 3)
                    #save cell
                    cell = Cell(el_nodes, el_faces, orientation, normals)
                    push!(cells, cell)
                end
            end
        end
    end
end

"""
function parse_mesh_triangle(root_file)
read mesh generated by triangle with root file name `root_file`
Ex: `parse_mesh_triangle("figure.1")`
"""
function parse_mesh_triangle(root_file)
    nodes = Vector{Node}()
    faces = Vector{Face}()
    cells = Vector{Cell}()
    parse_nodes!(nodes,root_file)
    parse_faces!(faces,root_file)
    parse_cells!(cells, faces, root_file)
    #check consistent cells orientation and compute normals
    for cell in cells
        coords = [nodes[j].x for j in cell.nodes]
        a = coords[2]-coords[1]
        b = coords[3]-coords[1]
        if (a[1]*b[2]-a[2]*b[1]) < 0
            #swap vertices 2 and 3
            cell.nodes[2],cell.nodes[3] = cell.nodes[3],cell.nodes[2]
            cell.faces = reverse(cell.faces)
        end
        #Compute normals
        for (i,k) in enumerate(((1,2),(2,3),(3,1)))
            v1 =  coords[k[2]] - coords[k[1]]
            n1 = Vec{2}([v1[2], -v1[1]])
            cell.normals[i] = n1/norm(n1)
            #Compute faces orientation
            #Just to standarize jump definitions
            cell.orientation[i] = cell.nodes[k[2]] > cell.nodes[k[1]]
        end
    end
    PolygonalMesh{size(nodes[1].x,1),eltype(nodes[1].x)}(cells, nodes, faces)
end
