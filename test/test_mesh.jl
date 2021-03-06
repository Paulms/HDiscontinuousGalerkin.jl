@testset "Test mesh reader" begin

using HDiscontinuousGalerkin
using Tensors

# Test read external mesh in Triangle format
root_file = "mesh/figure2.1"
mesh = parse_mesh_triangle(root_file)

# General checks
@test getncells(mesh) == 4
@test getnnodes(mesh) == 5
@test n_faces_per_cell(mesh) == 3
for cell in mesh.cells
    @test volume(get_coordinates(cell,  mesh)) ≈ 1/4
end
@test getcells_matrix(mesh) == [2 3 5; 4 1 5; 5 3 4; 1 2 5]
@test get_vertices_matrix(mesh) == [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0; 0.5 0.5]
# Check expected data for cell 1
@test mesh.cells[1].nodes == (2,3,5)
@test mesh.cells[1].faces == (1,2,3)
@test [face_orientation(mesh,1,i) for i in 1:3] == [true,false,true]
@test cell_diameter(mesh,1) == 1.0
@test getfaceset(mesh, "boundary") == Set([3,6,7,8])
@test get_coordinates(1, mesh) == [Vec{2}([1.0, 1.0]), Vec{2}([0.5, 0.5])]
@test getnfaces(mesh.cells[1]) == 3

#Test generated mesh
mesh = rectangle_mesh(TriangleCell, (2,2), Vec{2}((0.0,0.0)), Vec{2}((1.0,1.0)))
@test getncells(mesh) == 8
@test getnnodes(mesh) == 9
@test n_faces_per_cell(mesh) == 3
for cell in mesh.cells
    @test volume(get_coordinates(cell,  mesh)) ≈ 1/8
end
@test getcells_matrix(mesh) == [1 2 4; 2 5 4; 2 3 5; 3 6 5; 4 5 7; 5 8 7; 5 6 8; 6 9 8]
@test get_vertices_matrix(mesh) == [0.0 0.0; 0.5 0.0; 1.0 0.0; 0.0 0.5; 0.5 0.5; 1.0 0.5; 0.0 1.0; 0.5 1.0; 1.0 1.0]
# Check expected data for cell 1
@test mesh.cells[1].nodes == (1,2,4)
@test mesh.cells[1].faces == (1,2,3)
@test [face_orientation(mesh,1,i) for i in 1:3] == [true,false,true]
@test cell_diameter(mesh,1) == sqrt(2)/2
@test getfaceset(mesh, "boundary") == Set([3,7,9,16,2,11,12,15])
@test getnfaces(mesh.cells[1]) == 3

end
