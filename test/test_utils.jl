using HDiscontinuousGalerkin
#Test affine maps
A,b = get_affine_map([[-1.0,-1.0],[1.0,-1.0],[-1.0,1.0]],[[0.0,0.0],[1.0,0.0],[0.0,1.0]])
@test A*[0.0,0.0]+b ≈ [0.5,0.5]
