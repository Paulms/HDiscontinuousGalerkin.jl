using HDiscontinuousGalerkin
root_file = "../src/mesh/figura.1"
mesh = parse_mesh_triangle(root_file)

@test cell_diameter(mesh,1) == sqrt(2)/2
