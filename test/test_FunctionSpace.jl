using HDiscontinuousGalerkin
using Tensors

# Load mesh
root_file = "mesh/figure2.1"
mesh = parse_mesh_triangle(root_file)

# ### Trial and test functions
dim = 2
Vh = VectorFunctionSpace(mesh, Dubiner{dim,RefTetrahedron,1}())
Wh = ScalarFunctionSpace(mesh, Dubiner{dim,RefTetrahedron,1}())
# Basic Test
@test sum(Vh.ssp.detJ) ≈ 2.0
for i in 1:4
    @test getdetJdV(Wh,i,1)/Wh.qr_weights[1] ≈ 0.5
end
@test Wh.Jinv[1,1] ≈ Tensor{2,2}([1.0 1.0;-2.0 0.0])
@test Wh.Jinv[2,1] ≈ Tensor{2,2}([-1.0 -1.0;2.0 0.0])
@test Wh.Jinv[3,1] ≈ Tensor{2,2}([1.0 1.0;-1.0 1.0])
@test Wh.Jinv[4,1] ≈ Tensor{2,2}([1.0 -1.0;0.0 2.0])

n_basefuncs = 6 #vectorial base
n_basefuncs_s = 3 #scalar
const sq2 = sqrt(2)
Ce_ex=Vector{Matrix{Float64}}(4)
Ce_ex[1] = [0 0 0; 0 0 0; -3*sq2 0 0; 0.0 0 0; sqrt(6) 0 0; 0 0 0]
Ce_ex[2] = [0 0 0; 0 0 0; 3*sq2 0 0; 0.0 0 0; -sqrt(6) 0 0; 0 0 0]
Ce_ex[3] = [0 0 0; sqrt(6)/2 0 0; -3/2*sq2 0 0; 0 0 0; 3/2*sqrt(6) 0 0; 3/2*sq2 0 0]
Ce_ex[4] = [0 0 0; sqrt(6) 0 0; 0 0 0; 0.0 0 0; 0 0 0; 3*sq2 0 0]
Me = zeros(n_basefuncs, n_basefuncs)
Ce = zeros(n_basefuncs, n_basefuncs_s)

for cell_idx in 1:numcells(mesh)
    fill!(Me, 0)
    fill!(Ce, 0)
    for q_point in 1:getnquadpoints(Vh)
        dΩ = getdetJdV(Vh, cell_idx, q_point)
        for i in 1:n_basefuncs
            vh  = shape_value(Vh, q_point, i)
            div_vh = shape_divergence(Vh, q_point, i, cell_idx)
            for j in 1:n_basefuncs
                sigma = shape_value(Vh, q_point, j)
                # Integral sigma⋅v dΩ
                Me[i, j] += (sigma ⋅ vh) * dΩ
            end
            for j in 1:n_basefuncs_s
                u = shape_value(Wh, q_point, j)
                # Integral u*∇⋅v dΩ
                Ce[i,j] += (u*div_vh) * dΩ
            end
        end
    end
    @test Me ≈ 0.5*one(Me)
    @test Ce ≈ Ce_ex[cell_idx]
end
