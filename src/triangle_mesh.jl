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
Reads mesh generated by triangle with root file name `root_file`
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
    PolygonalMesh{size(nodes[1].x,1),eltype(nodes[1].x),eltype(eltype(values(facesets)))}(cells, nodes, faces, facesets)
end
