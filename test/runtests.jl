using Test
@time begin
    @time include("test_mesh.jl")
    @time include("test_quadrature.jl")
    @time include("test_basis.jl")
    @time include("test_fe.jl")
    @time include("test_utils.jl")
    @time include("test_ScalarFuncSp.jl")
    @time include("test_FunctionSpace.jl")
    @time include("test_handlers.jl")
    @time include("test_CGExample.jl")
end
