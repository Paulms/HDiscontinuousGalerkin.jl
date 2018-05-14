function interior_points(verts, order)
    n = length(verts)
    ls = [(verts[i] - verts[1])/order for i in 2:n]
    m = length(ls)
    grid_indices = []
    if m == 1
        grid_indices = [[i] for i in a:b]
    elseif m == 2
        grid_indices = [[i,j] for i in a:b for j in a:b-i+1]
    pts = Vector{typeof(verts[1])}()
    for indices in grid_indices
        res = verts[1]
        for (i,ii) in enumerate(indices)
            res += (ii-1) * ls[m - i+1]
        end
        push!(pts,res)
    end
    pts
end
