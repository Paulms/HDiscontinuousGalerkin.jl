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
                    node = Node((pln[2],pln[3]))
                    push!(nodes,node)
                end
            end
        end
    end
end

function parse_faces!(faces_ref,root_file)
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
                    element = (minimum(pln[2:3]),maximum(pln[2:3]))
                    token = ht_keyindex2!(faces_ref, element)
                    if token < 0
                        Base._setindex!(faces_ref, pln[4], element, -token)
                    end
                end
            end
        end
    end
end

function parse_cells!(cells, faces, faces_ref,facesets, nodes, root_file)
    #read cell nodes
    open(root_file*".ele") do f
        first_line = true
        n_el = 0
        n_faces = 0
        boundary_faces = Set{Int}()
        facesdict = Dict{NTuple{2,Int},Int}()
        el_faces = [0,0,0]
        for ln in eachline(f)
            m = match(r"^\s*(?:#|$)", ln)
            if m == nothing
                if (first_line)   #skip first line
                    first_line = false
                else
                    fill!(el_faces,0)
                    n_el = n_el + 1
                    #parse nodes
                    pln = collect(read_line(ln, (Int,Int,Int,Int)))
                    el_nodes = _check_node_data(nodes, pln[2:4]...)
                    #build faces
                    for (i,fn_id) in enumerate(((2,3),(3,1),(1,2)))
                        v1 = el_nodes[fn_id[1]]; v2 = el_nodes[fn_id[2]]
                        element = (min(v1,v2),max(v1,v2))
                        token = ht_keyindex2!(facesdict, element)
                        if token > 0
                            el_faces[i] = facesdict.vals[token]
                            if !(n_el in faces[facesdict.vals[token]].cells)
                                push!(faces[facesdict.vals[token]].cells,n_el)
                            end
                        else
                            ref = -1
                            token2 = ht_keyindex2!(faces_ref, element)
                            if token2 > 0
                                ref = faces_ref.vals[token2]
                            end
                            face = Face([n_el],(v1,v2))
                            push!(faces, face)
                            n_faces = n_faces + 1
                            el_faces[i] = n_faces
                            Base._setindex!(facesdict, n_faces, element, -token)
                            if ref > 0
                                push!(boundary_faces, n_faces)
                            end
                        end
                    end
                    #save cell
                    cell = Cell{2,3,3}(el_nodes, (el_faces...,))
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
    faces_ref = Dict{NTuple{2,Int},Int}()
    faces = Vector{Face{2}}()
    cells = Vector{TriangleCell}()
    facesets = Dict{String,Set{Int}}()
    parse_nodes!(nodes,root_file)
    parse_faces!(faces_ref, root_file)
    parse_cells!(cells, faces, faces_ref,facesets,nodes, root_file)
    PolygonalMesh{2,3,3,2,eltype(nodes[1].x)}(cells, nodes, faces, facesets)
end
