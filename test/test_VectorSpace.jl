using HDiscontinuousGalerkin

# Load mesh
root_file = "mesh/figure2.1"
mesh = parse_mesh_triangle(root_file)

# ### Trial and test functions
dim = 2
Vh = VectorFunctionSpace(mesh, Dubiner{dim,RefTetrahedron,1}())
@test sum(Vh.ssp.detJdV) ≈ 1.0

# Integral u⋅v dΩ
n_basefuncs = 6
Me = zeros(n_basefuncs, n_basefuncs)
for cell_idx in 1:numcells(mesh)
    fill!(Me, 0)
    for q_point in 1:getnquadpoints(Vh)
        dΩ = getdetJdV(Vh, cell_idx, q_point)
        for i in 1:n_basefuncs
            vh  = shape_value(Vh, q_point, i)
            for j in 1:n_basefuncs
                sigma = shape_value(Vh, q_point, j)
                Me[i, j] += (sigma ⋅ vh) * dΩ
            end
        end
    end
    @test Me ≈ 0.5*one(Me)
end
