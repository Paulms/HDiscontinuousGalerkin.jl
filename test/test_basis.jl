using HDiscontinuousGalerkin
using Tensors

#Test Dubiner base
interpolation = Dubiner{2,RefTetrahedron,4}()
quad_rule = QuadratureRule{2,RefTetrahedron}(Strang(),5)
for j = 1:15
    @test integrate(x->(dubiner_basis(x[1],x[2],j) - value(interpolation, j, x))^2,quad_rule) < eps()
    @test integrate(x->sum((∇dubiner_basis(x[1],x[2],j) - gradient_value(interpolation, j, x)).^2),quad_rule) < eps()
end

# Test nodal points for Lagrange base
@test get_nodal_points(RefTetrahedron(), Val{2}, 1) == [Vec{2, Float64}((0.0, 0.0)),
                                            Vec{2, Float64}((1.0, 0.0)),
                                            Vec{2, Float64}((0.0, 1.0))]

@test get_nodal_points(RefTetrahedron(), Val{2}, 2) == [Vec{2, Float64}((0.0, 0.0)),
                                            Vec{2, Float64}((1.0, 0.0)),
                                            Vec{2, Float64}((0.0, 1.0)),
                                            Vec{2, Float64}((0.5, 0.0)),
                                            Vec{2, Float64}((0.5, 0.5)),
                                            Vec{2, Float64}((0.0, 0.5))]
@test length(get_nodal_points(RefTetrahedron(), Val{2}, 3)) == 10
@test get_nodal_points(RefTetrahedron(), Val{2}, 3)[10] ≈ Vec{2, Float64}((1/3, 1/3))
# Test Lagrange base
function ref_value(i::Int, ξ::Vec{2})
    ξ_x = ξ[1]
    ξ_y = ξ[2]
    i == 2 && return ξ_x
    i == 3 && return ξ_y
    i == 1 && return 1. - ξ_x - ξ_y
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end

interpolation = Lagrange{2,RefTetrahedron,1}()
const ξ_ref = Vec{2}((0.5,0.5))
for j = 1:3
    @test abs(ref_value(j,ξ_ref) - value(interpolation,j,ξ_ref)) < eps(Float64)
    @test norm(gradient(ξ -> ref_value(j, ξ), ξ_ref) - gradient_value(interpolation,j,ξ_ref), Inf) < eps(Float64)
end
