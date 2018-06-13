using HDiscontinuousGalerkin
using Tensors
root_file = "mesh/figure2.1"
mesh = parse_mesh_triangle(root_file)

# General checks
@test numcells(mesh) == 4
for cell in mesh.cells
    @test volume(get_coordinates(cell,  mesh)) ≈ 1/4
end
# Check expected data for cell 1
@test mesh.cells[1].nodes == [2,3,5]
@test mesh.cells[1].faces == [1,2,3]
@test mesh.cells[1].orientation == [true,false,true]
@test mesh.cells[1].normals ≈ [Vec{2}([-sqrt(2)/2,sqrt(2)/2]), Vec{2}([-sqrt(2)/2,-sqrt(2)/2]), Vec{2}([1.0,0.0])]
@test cell_diameter(mesh,1) == 1.0
@test get_faceset(mesh, "boundary") == Set([3,6,7,8])
