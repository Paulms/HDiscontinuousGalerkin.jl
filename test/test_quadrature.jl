using HDiscontinuousGalerkin

# Test GrundmannMoeller Quadrature
quad_rule = QuadratureRule{2,RefTetrahedron}(GrundmannMoeller(),0)
@test getweights(quad_rule) ≈ [0.5]
@test getpoints(quad_rule)[1] ≈ [1//3,1//3]
quad_rule = QuadratureRule{2,RefTetrahedron}(GrundmannMoeller(),1)
@test getweights(quad_rule) ≈ 0.5*[0.520833333333333 ,0.520833333333333,0.520833333333333 ,-0.562500000000000]
@test getpoints(quad_rule)[1] ≈ [1//5,1//5]
@test getpoints(quad_rule)[2] ≈ [3//5,1//5]
@test getpoints(quad_rule)[3] ≈ [1//5,3//5]
@test getpoints(quad_rule)[4] ≈ [1//3,1//3]

for i in 0:5
    quad_rule = QuadratureRule{2,RefTetrahedron}(GrundmannMoeller(),i)
    @test sum(getweights(quad_rule)) ≈ 0.5
end

# Test Strang Quadrature
for i in 1:5
    quad_rule = QuadratureRule{2,RefTetrahedron}(Strang(),i)
    @test sum(getweights(quad_rule)) ≈ 0.5
end
