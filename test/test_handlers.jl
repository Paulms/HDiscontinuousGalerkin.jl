using HDiscontinuousGalerkin
mesh = rectangle_mesh(TriangleCell, (2,2), Vec{2}((0.0,0.0)), Vec{2}((1.0,1.0)))
dim = 2
Wh = ScalarFunctionSpace(mesh, Lagrange{dim,RefTetrahedron,1}())
u_h = TrialFunction(Wh, mesh)

dh = DofHandler(mesh)
push!(dh, u_h)
close!(dh);

@test dh.cell_dofs == [1, 2, 3, 2, 4, 3, 2, 5, 4, 5, 6, 4, 3, 4, 7, 4, 8, 7, 4, 6, 8, 6, 9, 8]
@test dh.cell_dofs_offset == [1, 4, 7, 10, 13, 16, 19, 22, 25]
