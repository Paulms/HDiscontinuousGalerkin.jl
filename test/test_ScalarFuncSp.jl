@testset "Test Scalar Function Spaces" begin

using HDiscontinuousGalerkin
using Tensors

root_file = "mesh/figure2.1"
mesh = parse_mesh_triangle(root_file)

finiteElement = ContinuousLagrange{2,RefTetrahedron,1}()
Wh, Whf = ScalarFunctionSpace(mesh, finiteElement)

# Basic Test
@test getnlocaldofs(Wh) == 3
@test sum(Wh.detJ) ≈ 2.0
for i in 1:4
    @test getdetJdV(Wh,i,1)/Wh.qr_weights[1] ≈ 0.5
end
@test Wh.Jinv[1,1] ≈ Tensor{2,2}([1.0 1.0;-2.0 0.0])
@test Wh.Jinv[2,1] ≈ Tensor{2,2}([-1.0 -1.0;2.0 0.0])
@test Wh.Jinv[3,1] ≈ Tensor{2,2}([1.0 1.0;-1.0 1.0])
@test Wh.Jinv[4,1] ≈ Tensor{2,2}([1.0 -1.0;0.0 2.0])

# Face Data
sq2 = sqrt(2)
sq3 = sqrt(3)
@test Whf.detJf[1,:,1] ≈ [sq2/2,sq2/2,1]
@test Whf.detJf[2,:,1] ≈ [sq2/2,sq2/2,1]
@test Whf.detJf[3,:,1] ≈ [1,sq2/2,sq2/2]
@test Whf.detJf[4,:,1] ≈ [sq2/2,sq2/2,1]
@test Whf.normals[1,:]≈ [Vec{2}([-sq2/2,sq2/2]), Vec{2}([-sq2/2,-sq2/2]), Vec{2}([1.0,0.0])]

end
