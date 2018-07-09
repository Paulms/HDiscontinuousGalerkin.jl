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
nodal_points, topology = get_nodal_points(RefTetrahedron, Val{2}, 1)
@test  nodal_points == [Vec{2, Float64}((0.0, 0.0)),
                        Vec{2, Float64}((1.0, 0.0)),
                        Vec{2, Float64}((0.0, 1.0))]
@test topology == Dict(0=>3,1=>0,2=>0)
nodal_points, topology = get_nodal_points(RefTetrahedron, Val{2}, 2)
@test nodal_points == [Vec{2, Float64}((0.0, 0.0)),
                        Vec{2, Float64}((1.0, 0.0)),
                        Vec{2, Float64}((0.0, 1.0)),
                        Vec{2, Float64}((0.5, 0.5)),
                        Vec{2, Float64}((0.0, 0.5)),
                        Vec{2, Float64}((0.5, 0.0))]
@test topology == Dict(0=>3,1=>3,2=>0)
nodal_points, topology = get_nodal_points(RefTetrahedron, Val{2}, 3)
@test length(nodal_points) == 10
@test topology == Dict(0=>3,1=>6,2=>1)
@test nodal_points[10] ≈ Vec{2, Float64}((1/3, 1/3))
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
ξ_ref = Vec{2}((0.5,0.5))
for j = 1:3
    @test abs(ref_value(j,ξ_ref) - value(interpolation,j,ξ_ref)) < eps(Float64)
    @test norm(gradient(ξ -> ref_value(j, ξ), ξ_ref) - gradient_value(interpolation,j,ξ_ref), Inf) < eps(Float64)
end

function ref_value1(i::Int, ξ::Vec{1})
    ξ_x = ξ[1]
    i == 1 && return 1-ξ_x
    i == 2 && return ξ_x
    throw(ArgumentError("no shape function $i for interpolation $ip"))
end
ξ_ref = Vec{1}((0.5,))
interpolation = Lagrange{1,RefTetrahedron,1}()
for j = 1:2
    @test abs(ref_value1(j,ξ_ref) - value(interpolation,j,ξ_ref)) < eps(Float64)
    @test norm(gradient(ξ -> ref_value1(j, ξ), ξ_ref) - gradient_value(interpolation,j,ξ_ref), Inf) < eps(Float64)
end

#Test Legendre basis
function ref_value(i::Int, x::Real)
    @assert i <= 3
    if i == 0
        return 1.0
    elseif i == 1
        return sqrt(3)*(2*x - 1)
    elseif i == 2
        return sqrt(5)*(6*x^2 - 6*x + 1)
    else
        return sqrt(7)*(2*x - 1)*(10*x^2 - 10*x + 1)
    end
end
function ref_dvalue(i::Int, x::Real)
    @assert i <= 3
    if i == 0
        return 0
    elseif i == 1
        return sqrt(3)*2
    elseif i == 2
        return sqrt(5)*(12*x - 6)
    else
        return sqrt(7)*12*(5*x^2 - 5*x + 1)
    end
end

interpolation = Legendre{1,RefTetrahedron,3}()
quad_rule = QuadratureRule{1,RefTetrahedron}(GaussLegendre(),3)
for i = 0:3
    @test integrate(x->(ref_value(i,x[1]) - value(interpolation, i+1, x))^2,quad_rule) < eps()
    @test integrate(x->(ref_dvalue(i,x[1]) - gradient_value(interpolation, i+1, x)[1])^2,quad_rule) < eps()
end
