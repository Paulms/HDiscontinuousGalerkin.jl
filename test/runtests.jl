using HDiscontinuousGalerkin
using Base.Test

@time @testset "Test mesh reader" begin include("test_mesh.jl") end
@time @testset "Test quadrature rules" begin include("test_quadrature.jl") end
@time @testset "Test basis" begin include("test_basis.jl") end
@time @testset "Test utils" begin include("test_utils.jl") end
@time @testset "Test FunctionSpaces" begin include("test_FunctionSpace.jl") end
