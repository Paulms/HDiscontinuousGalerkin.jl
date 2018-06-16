using HDiscontinuousGalerkin
using Tensors

# Test read external mesh in Triangle format
root_file = "mesh/figure2.1"
mesh = parse_mesh_triangle(root_file)

# General checks
@test getncells(mesh) == 4
@test getnnodes(mesh) == 5
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
@test get_coordinates(mesh.faces[1], mesh) == [Vec{2}([1.0, 1.0]), Vec{2}([0.5, 0.5])]

#Test generated mesh
mesh = rectangle_mesh(TriangleCell, (2,2), Vec{2}((0.0,0.0)), Vec{2}((1.0,1.0)))
@test getncells(mesh) == 8
for cell in mesh.cells
    @test volume(get_coordinates(cell,  mesh)) ≈ 1/8
end
# Check expected data for cell 1
@test mesh.cells[1].nodes == [1,2,4]
@test mesh.cells[1].faces == [1,2,3]
@test mesh.cells[1].orientation == [true,false,true]
@test mesh.cells[1].normals ≈ [Vec{2}([sqrt(2)/2,sqrt(2)/2]), Vec{2}([-1.0,0.0]), Vec{2}([0.0,-1.0])]
@test cell_diameter(mesh,1) == sqrt(2)/2
@test get_faceset(mesh, "boundary") == Set([3,7,9,16,2,11,12,15])
