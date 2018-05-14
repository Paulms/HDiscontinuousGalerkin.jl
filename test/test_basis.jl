using HDiscontinuousGalerkin
using Tensors

const rr = 0.5
const ss = 0.5
interpolation = Dubiner{2,RefTetrahedron,4}()
for j = 1:15
    @test dubiner_basis(rr,ss,j) ≈ value(interpolation, j, (rr, ss))
    @test ∇dubiner_basis(rr,ss,j) ≈ gradient_value(interpolation, j, (rr, ss))
end

function ref_value(i::Int, ξ::Vec{2})
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    i == 1 && return ξ_x
    i == 2 && return ξ_y
    i == 3 && return 1. - ξ_x - ξ_y
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

interpolation = Lagrange{2,RefTetrahedron,1}()
const ξ_ref = Vec{2}((0.5,0.5))
for j = 1:3
    @test ref_value(j,ξ_ref) ≈ value(interpolation,j,ξ_ref)
    @test gradient(ξ -> ref_value(j, ξ), ξ_ref) ≈ gradient_value(interpolation,j,ξ_ref)
end
