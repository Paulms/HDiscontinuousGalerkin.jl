using HDiscontinuousGalerkin
using Tensors
root_file = "mesh/figura.1"
mesh = parse_mesh_triangle(root_file)

# General checks
@test numcells(mesh) == 8
for cell in mesh.cells
    @test volume(get_coordinates(cell,  mesh)) ≈ 1/8
end
# Check expected data for cell 1
@test mesh.cells[1].nodes == [8,3,5]
@test mesh.cells[1].faces == [1,2,3]
@test mesh.cells[1].orientation == [false,true,true]
@test mesh.cells[1].normals ≈ [Vec{2}([1.0,0.0]), Vec{2}([-sqrt(2)/2,sqrt(2)/2]), Vec{2}([0.0,-1.0])]
@test cell_diameter(mesh,1) == sqrt(2)/2
