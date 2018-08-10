#########################
# All RefTetrahedron   #
#########################
struct RefTetrahedron <: AbstractRefShape end
@inline get_num_faces(::Type{RefTetrahedron}, ::Type{Val{dim}}) where {dim} = dim + 1
@inline get_num_vertices(::Type{RefTetrahedron}, ::Type{Val{dim}}) where {dim} = dim + 1

function reference_coordinates(::Type{RefTetrahedron}, ::Type{Val{1}})
    return [Vec{1, Float64}((0.0,)),Vec{1, Float64}((1.0,))]
end
function gettopology(::Type{RefTetrahedron}, ::Type{Val{1}})
    return Dict(0=>2,1=>1)
end
function reference_coordinates(::Type{RefTetrahedron}, ::Type{Val{2}})
    [Vec{2, Float64}((0.0, 0.0)),
     Vec{2, Float64}((1.0, 0.0)),
     Vec{2, Float64}((0.0, 1.0))]
end
function reference_edges(::Type{RefTetrahedron},::Type{Val{2}})
    [[Vec{2, Float64}((1.0, 0.0)),Vec{2, Float64}((0.0, 1.0))],
     [Vec{2, Float64}((0.0, 1.0)),Vec{2, Float64}((0.0, 0.0))],
     [Vec{2, Float64}((0.0, 0.0)),Vec{2, Float64}((1.0, 0.0))]]
end
@inline reference_edge_nodes(::Type{RefTetrahedron},::Type{Val{2}}) = ((2,3),(3,1),(1,2))

function gettopology(::Type{RefTetrahedron}, ::Type{Val{2}})
    return Dict(0=>3,1=>3,2=>1)
end

"""
get_nodal_points(shp::RefTetrahedron, dim, order)
get points for a nodal basis of order `order` on a `dim`
    dimensional simplex
"""
function get_nodal_points(::Type{RefTetrahedron}, ::Type{Val{1}}, order)
    points = Vector{Vec{1,Float64}}()
    vertices = reference_coordinates(RefTetrahedron, Val{1})
    topology = Dict{Int, Int}()
    append!(points, vertices)
    push!(topology, 0=>length(points))
    append!(points, _interior_points(vertices, order))
    push!(topology, 1=>length(points)-topology[0])
    points, topology
end

function get_nodal_points(::Type{RefTetrahedron}, ::Type{Val{2}}, order)
    points = Vector{Vec{2,Float64}}()
    vertices = reference_coordinates(RefTetrahedron, Val{2})
    topology = Dict{Int, Int}()
    append!(points, vertices)
    push!(topology, 0=>length(points))
    [append!(points, _interior_points(verts, order)) for verts in reference_edges(RefTetrahedron, Val{2})]
    push!(topology, 1=>length(points)-topology[0])
    append!(points, _interior_points(vertices, order))
    push!(topology, 2=>length(points)-topology[0]-topology[1])
    points, topology
end

function _interior_points(verts, order)
    n = length(verts)
    ls = [(verts[i] - verts[1])/order for i in 2:n]
    m = length(ls)
    grid_indices =  []
    if m == 1
        grid_indices = [[i] for i in 1:order-1]
    elseif m == 2 && order > 2
        grid_indices = [[i,j] for i in 1:order-1 for j in 1:order-i-1]
    end
    pts = Vector{typeof(verts[1])}()
    for indices in grid_indices
        res = verts[1]
        for (i,ii) in enumerate(indices)
            res += (ii) * ls[m - i+1]
        end
        push!(pts,res)
    end
    pts
end

function weighted_normal(J::Tensor{2,2}, face::Int, ::Type{RefTetrahedron}, ::Type{Val{2}})
    @inbounds begin
        face == 1 && return Vec{2}((-(J[2,1] - J[2,2]), J[1,1] - J[1,2]))
        face == 2 && return Vec{2}((-J[2,2], J[1,2]))
        face == 3 && return Vec{2}((J[2,1], -J[1,1]))
    end
    throw(ArgumentError("unknown face number: $face"))
end

""" Compute volume of a simplex spanned by vertices `verts` """
function volume(verts::Vector{Vec{N, T}}) where {N,T}
    # Volume of reference simplex element is 1/n!
    n = length(verts) - 1
    ref_verts = reference_coordinates(RefTetrahedron, Val{n})
    A, b = get_affine_map(ref_verts, verts)
    F = svd(A)
    return *(F.S...)/factorial(n)
end
