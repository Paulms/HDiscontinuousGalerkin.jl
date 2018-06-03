struct RefTetrahedron <: AbstractRefShape end

function reference_coordinates(::RefTetrahedron, ::Type{Val{1}})
    return [Vec{1, Float64}((0.0,)),Vec{1, Float64}((1.0,))]
end
function reference_coordinates(::RefTetrahedron, ::Type{Val{2}})
    [Vec{2, Float64}((0.0, 0.0)),
     Vec{2, Float64}((1.0, 0.0)),
     Vec{2, Float64}((0.0, 1.0))]
end

function reference_edges(::RefTetrahedron,::Type{Val{2}})
    [[Vec{2, Float64}((1.0, 0.0)),Vec{2, Float64}((0.0, 1.0))],
     [Vec{2, Float64}((0.0, 1.0)),Vec{2, Float64}((0.0, 0.0))],
     [Vec{2, Float64}((0.0, 0.0)),Vec{2, Float64}((1.0, 0.0))]]

end

"""
get_nodal_points(shp::RefTetrahedron, dim, order)
get points for a nodal basis of order `order` on a `dim`
    dimensional simplex
"""
function get_nodal_points(shp::RefTetrahedron, ::Type{Val{1}}, order)
    points = Vector{Vec{1,Float64}}()
    vertices = reference_coordinates(shp, Val{1})
    append!(points, vertices)
    append!(points, _interior_points(vertices, order))
    points
end

function get_nodal_points(shp::RefTetrahedron, ::Type{Val{2}}, order)
    points = Vector{Vec{2,Float64}}()
    vertices = reference_coordinates(shp, Val{2})
    append!(points, vertices)
    [append!(points, _interior_points(verts, order)) for verts in reference_edges(shp, Val{2})]
    append!(points, _interior_points(vertices, order))
    points
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

""" Compute volume of a simplex spanned by vertices `verts` """
function volume(verts::Vector{Vec{N, T}}) where {N,T}
    # Volume of reference simplex element is 1/n!
    n = length(verts) - 1
    ref_verts = reference_coordinates(RefTetrahedron(), Val{n})
    A, b = get_affine_map(ref_verts, verts)
    F = svd(A)
    return *(F[2]...)/factorial(n)
end
