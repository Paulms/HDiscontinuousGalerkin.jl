function _generate_2d_nodes!(nodes, nx, ny, LL, LR, UR, UL)
      for i in 0:ny-1
        ratio_bounds = i / (ny-1)

        x0 = LL[1] * (1 - ratio_bounds) + ratio_bounds * UL[1]
        x1 = LR[1] * (1 - ratio_bounds) + ratio_bounds * UR[1]

        y0 = LL[2] * (1 - ratio_bounds) + ratio_bounds * UL[2]
        y1 = LR[2] * (1 - ratio_bounds) + ratio_bounds * UR[2]

        for j in 0:nx-1
            ratio = j / (nx-1)
            x = x0 * (1 - ratio) + ratio * x1
            y = y0 * (1 - ratio) + ratio * y1
            push!(nodes, Node(Vec{2}((x, y))))
        end
    end
end

#########################
# Triangle Cells 2D   #
#########################
# Check face orientation consistency
function _check_node_data(nodes, n1,n2,n3)
    a = nodes[n2].x-nodes[n1].x
    b = nodes[n3].x-nodes[n1].x
    if (a[1]*b[2]-a[2]*b[1]) < 0
        #swap vertices 2 and 3
        return (n1,n3,n2)
    end
    return (n1,n2,n3)
end

function _build_cells(cells::Vector{TriangleCell}, el_nodes, el_faces,n_el, faces, facesdict,nodes)
    fill!(el_faces,0)
    #build faces
    for (i,fn_id) in enumerate(reference_edge_nodes(RefTetrahedron, Val{2}))
        v1 = el_nodes[fn_id[1]]; v2 = el_nodes[fn_id[2]]
        element = (min(v1,v2),max(v1,v2))
        token = ht_keyindex2!(facesdict, element)
        if token > 0
                el_faces[i] = facesdict.vals[token]
                if !(n_el in faces[facesdict.vals[token]].cells)
                    push!(faces[facesdict.vals[token]].cells,n_el)
                end
        else
            Base._setindex!(facesdict, length(faces)+1, element, -token)
            face = Face([n_el],(v1,v2))
            push!(faces, face)
            el_faces[i] = length(faces)
        end
    end
    #save cell
    cell = TriangleCell(el_nodes, (el_faces...,))
    push!(cells, cell)
end

"""
rectangle_mesh(::Type{TriangleCell}, nel::NTuple{2,Int}, LL::Vec{2,T}, UR::Vec{2,T})
Generate a rectangular mesh with triangular cells, where `LL` is the low left vertex
and `UR` is the upper right one. `nel` is a tuple with the number of partions to be
used in each dimension.
"""
@inline rectangle_mesh(::Type{TriangleCell}, nel::NTuple{2,Int}, LL::Vec{2,T}, UR::Vec{2,T}) where {T} =
        rectangle_mesh(RefTetrahedron, Val{2}, nel, LL, UR)

"""
rectangle_mesh(::Type{RefTetrahedron}, ::Type{Val{2}}, nel::NTuple{2,Int}, LL::Vec{2,T}, UR::Vec{2,T})
Generate a rectangular mesh with triangular cells, where `LL` is the low left vertex
and `UR` is the upper right one. `nel` is a tuple with the number of partions to be
used in each dimension.
"""
function rectangle_mesh(::Type{RefTetrahedron}, ::Type{Val{2}}, nel::NTuple{2,Int}, LL::Vec{2,T}, UR::Vec{2,T}) where {T}
    LR = Vec{2}((UR[1],LL[2]))
    UL = Vec{2}((LL[1],UR[2]))
    nel_x = nel[1]; nel_y = nel[2]; nel_tot = 2*nel_x*nel_y
    n_nodes_x = nel_x + 1; n_nodes_y = nel_y + 1
    n_nodes = n_nodes_x * n_nodes_y

    # Generate nodes
    nodes = Node{2,T}[]
    _generate_2d_nodes!(nodes, n_nodes_x, n_nodes_y, LL, LR, UR, UL)

    faces = Vector{Face{2}}()

    # Generate cells
    node_array = reshape(collect(1:n_nodes), (n_nodes_x, n_nodes_y))
    cells = TriangleCell[]
    facesdict = Dict{NTuple{2,Int},Int}()
    el_faces = [0,0,0]
    n_el = 0
    for j in 1:nel_y, i in 1:nel_x
        n_el = n_el + 1
        el_nodes = _check_node_data(nodes, node_array[i,j], node_array[i+1,j], node_array[i,j+1])
        _build_cells(cells, el_nodes, el_faces,n_el, faces, facesdict,nodes) # ◺

        n_el = n_el + 1
        el_nodes = _check_node_data(nodes, node_array[i+1,j], node_array[i+1,j+1], node_array[i,j+1])
        _build_cells(cells, el_nodes, el_faces,n_el, faces, facesdict,nodes) # ◹
    end

    # Add faces sets
    bottomSet = Set{Int}()
    rightSet = Set{Int}()
    topSet = Set{Int}()
    leftSet = Set{Int}()
    for k in 1:length(faces)
        ref = 0
        if length(faces[k].cells) == 1
            if (nodes[faces[k].nodes[1]].x[2] == LL[2] && nodes[faces[k].nodes[2]].x[2] == LL[2])
                push!(bottomSet, k)
            elseif (nodes[faces[k].nodes[1]].x[1] == UR[1] && nodes[faces[k].nodes[2]].x[1] == UR[1])
                push!(rightSet, k)
            elseif (nodes[faces[k].nodes[1]].x[2] == UR[2] && nodes[faces[k].nodes[2]].x[2] == UR[2])
                push!(topSet, k)
            elseif (nodes[faces[k].nodes[1]].x[1] == LL[1] && nodes[faces[k].nodes[2]].x[1] == LL[1])
                push!(leftSet, k)
            else
                throw("Face $k belongs to one cell but is not in boundary")
            end
        end
    end
    facesets = Dict("bottom"=>bottomSet, "right"=>rightSet,"left"=>leftSet,
                    "top"=>topSet,"boundary"=>union(bottomSet,rightSet,leftSet,topSet))
    return PolygonalMesh{2,3,3,2,T}(cells, nodes, faces, facesets)
end
