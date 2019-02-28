@testset "Test Scalar Function Spaces" begin

using HDiscontinuousGalerkin
using Tensors

root_file = "mesh/figure2.1"
mesh = parse_mesh_triangle(root_file)

finiteElement = ContinuousLagrange{2,RefTetrahedron,1}()
Wh = ScalarFunctionSpace(mesh, finiteElement)

# Basic Test
sq2 = sqrt(2)
sq3 = sqrt(3)
invs = (Tensor{2,2}([1.0 1.0;-2.0 0.0]),
        Tensor{2,2}([-1.0 -1.0;2.0 0.0]),
        Tensor{2,2}([1.0 1.0;-1.0 1.0]),
        Tensor{2,2}([1.0 -1.0;0.0 2.0]))
detsJf = ([sq2/2,sq2/2,1],
          [sq2/2,sq2/2,1],
          [1,sq2/2,sq2/2],
          [sq2/2,sq2/2,1])
@test getnlocaldofs(Wh) == 3
@inbounds for (cellcount, cell) in enumerate(CellIterator(mesh))
    reinit!(Wh, cell)
    @test Wh.detJ[] ≈ 0.5
    @test getdetJdV(Wh,1)/Wh.qr_weights[1] ≈ 0.5
    @test Wh.Jinv[] ≈ invs[cellcount]
    # Face Data
    @test Wh.detJf ≈ detsJf[cellcount]
    if cellcount == 1
        @test Wh.normals ≈ [Vec{2}([-sq2/2,sq2/2]), Vec{2}([-sq2/2,-sq2/2]), Vec{2}([1.0,0.0])]
    end
end
end
