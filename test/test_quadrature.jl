using HDiscontinuousGalerkin

quad_rule = QuadratureRule{2,RefSimplex,GrundmannMoeller}(0)
@test quad_rule.weights ≈ [1.0]
@test quad_rule.points[1] ≈ [1//3,1//3]
quad_rule = QuadratureRule{2,RefSimplex,GrundmannMoeller}(1)
@test quad_rule.weights ≈ [0.520833333333333 ,0.520833333333333,0.520833333333333 ,-0.562500000000000]
@test quad_rule.points[1] ≈ [1//5,1//5]
@test quad_rule.points[2] ≈ [3//5,1//5]
@test quad_rule.points[3] ≈ [1//5,3//5]
@test quad_rule.points[4] ≈ [1//3,1//3]
