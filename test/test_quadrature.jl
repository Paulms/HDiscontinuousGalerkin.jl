using HDiscontinuousGalerkin

quad_rule = QuadratureRule{2,RefSimplex}(GrundmannMoeller(),0)
@test getweights(quad_rule) ≈ [1.0]
@test getpoints(quad_rule)[1] ≈ [1//3,1//3]
quad_rule = QuadratureRule{2,RefSimplex}(GrundmannMoeller(),1)
@test getweights(quad_rule) ≈ [0.520833333333333 ,0.520833333333333,0.520833333333333 ,-0.562500000000000]
@test getpoints(quad_rule)[1] ≈ [1//5,1//5]
@test getpoints(quad_rule)[2] ≈ [3//5,1//5]
@test getpoints(quad_rule)[3] ≈ [1//5,3//5]
@test getpoints(quad_rule)[4] ≈ [1//3,1//3]
