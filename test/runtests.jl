using HDiscontinuousGalerkin
using Test

@time @testset "Test mesh reader" begin include("test_mesh.jl") end
@time @testset "Test quadrature rules" begin include("test_quadrature.jl") end
@time @testset "Test basis" begin include("test_basis.jl") end
@time @testset "Test Finite Elements" begin include("test_fe.jl") end
@time @testset "Test utils" begin include("test_utils.jl") end
@time @testset "Test Scalar Function Spaces" begin include("test_ScalarFuncSp.jl") end
@time @testset "Test FunctionSpaces" begin include("test_FunctionSpace.jl") end
@time @testset "Test handlers" begin include("test_handlers.jl") end
@time @testset "Test CG poisson example" begin include("test_CGExample.jl") end
