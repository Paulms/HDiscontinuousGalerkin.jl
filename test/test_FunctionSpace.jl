using HDiscontinuousGalerkin
using Tensors

# Load mesh
root_file = "mesh/figure2.1"
mesh = parse_mesh_triangle(root_file)

# ### Trial and test functions
dim = 2
Vh = VectorFunctionSpace(mesh, Dubiner{dim,RefTetrahedron,1}())
Wh = ScalarFunctionSpace(mesh, Dubiner{dim,RefTetrahedron,1}())
Mh = ScalarTraceFunctionSpace(Wh, Legendre{dim-1,RefTetrahedron,1}())
# Basic Test
@test sum(Vh.ssp.detJ) ≈ 2.0
for i in 1:4
    @test getdetJdV(Wh,i,1)/Wh.qr_weights[1] ≈ 0.5
end
@test Wh.Jinv[1,1] ≈ Tensor{2,2}([1.0 1.0;-2.0 0.0])
@test Wh.Jinv[2,1] ≈ Tensor{2,2}([-1.0 -1.0;2.0 0.0])
@test Wh.Jinv[3,1] ≈ Tensor{2,2}([1.0 1.0;-1.0 1.0])
@test Wh.Jinv[4,1] ≈ Tensor{2,2}([1.0 -1.0;0.0 2.0])

const sq2 = sqrt(2)
const sq3 = sqrt(3)
@test Wh.detJf[1] ≈ [sq2/4,sq2/4,1/2]
@test Wh.detJf[2] ≈ [sq2/4,sq2/4,1/2]
@test Wh.detJf[3] ≈ [1/2,sq2/4,sq2/4]
@test Wh.detJf[4] ≈ [sq2/4,sq2/4,1/2]

n_basefuncs = 6 #vectorial base
n_basefuncs_s = 3 #scalar
n_basefuncs_t = 2 #trace
Ce_ex=Vector{Matrix{Float64}}(4)
Ce_ex[1] = [0 0 0; 0 0 0; -3*sq2 0 0; 0.0 0 0; sqrt(6) 0 0; 0 0 0]
Ce_ex[2] = [0 0 0; 0 0 0; 3*sq2 0 0; 0.0 0 0; -sqrt(6) 0 0; 0 0 0]
Ce_ex[3] = [0 0 0; sqrt(6)/2 0 0; -3/2*sq2 0 0; 0 0 0; 3/2*sqrt(6) 0 0; 3/2*sq2 0 0]
Ce_ex[4] = [0 0 0; sqrt(6) 0 0; 0 0 0; 0.0 0 0; 0 0 0; 3*sq2 0 0]
Se_ex=Vector{Matrix{Float64}}(4)
Se_ex[1] = [2*sq2+2 0 2-2*sq2; 0 4*sq2+4 0; 2-2*sq2 0 4*sq2+4]
Se_ex[2] = [2*sq2+2 0 2-2*sq2; 0 4*sq2+4 0; 2-2*sq2 0 4*sq2+4]
Se_ex[3] = [2*sq2+2 sqrt(6)-sqrt(3) sq2-1; sqrt(6)-sqrt(3)  4*sq2+4 0; sq2-1 0 4*sq2+4]
Se_ex[4] = [2*sq2+2 0 2-2*sq2; 0 4*sq2+4 0; 2-2*sq2 0 4*sq2+4]
Ee_ex=Vector{Matrix{Float64}}(4)
Ee_ex[1] = -1/sq2*[sq2/2 0 sq2/2 0 -sq2 0; sq3/2 -1/2 -sq3/2 1/2 0 -2; 1/2 sq3/2 1/2 sq3/2 2 0;
           -sq2/2 0 sq2/2 0 0 0; -sq3/2 1/2 -sq3/2  1/2 0 0; -1/2 -sq3/2 1/2 sq3/2 0 0]
Ee_ex[2] = -1/sq2*[-sq2/2 0 -sq2/2 0 sq2 0; -sq3/2 1/2 sq3/2 -1/2 0 -2; -1/2 -sq3/2 -1/2 -sq3/2 -2 0;
          sq2/2 0 -sq2/2 0 0 0; sq3/2 -1/2 sq3/2  -1/2 0 0; 1/2 sq3/2 -1/2 -sq3/2 0 0]
Ee_ex[3] = -1/sq2*[0 0 sq2/2 0 -sq2/2 0; 0 0 -sq3/2  -1/2 0 1; 0 0 1/2 -sq3/2 1 0;
            -sq2 0 sq2/2 0 sq2/2 0; -sq3 1 -sq3/2 -1/2 0 -1; -1 -sq3 1/2 -sq3/2 -1 0]
Ee_ex[4] = -1/sq2*[-sq2/2 0 sq2/2 0 0 0; -sq3/2 1/2 -sq3/2 1/2 0 0; -1/2 -sq3/2 1/2 sq3/2 0 0;
           -sq2/2 0 -sq2/2 0 sq2 0; -sq3/2 1/2 sq3/2  -1/2 0 2; -1/2 -sq3/2 -1/2 -sq3/2 -2 0]


Me = zeros(n_basefuncs, n_basefuncs)
Ce = zeros(n_basefuncs, n_basefuncs_s)
Se = zeros(n_basefuncs_s, n_basefuncs_s)
Ee = zeros(n_basefuncs,3*n_basefuncs_t)
Fe = zeros(n_basefuncs_s,3*n_basefuncs_t)

for cell_idx in 1:numcells(mesh)
    fill!(Me, 0)
    fill!(Ce, 0)
    fill!(Se, 0)
    fill!(Ee, 0)
    #Cell integrals
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
    #Face integrals
    for face_idx in 1:numfaces(mesh.cells[cell_idx])
        for q_point in 1:getnfacequadpoints(Wh)
            dS = getdetJdS(Wh, cell_idx, face_idx, q_point)
            for i in 1:n_basefuncs_s
                w = face_shape_value(Wh, face_idx, q_point, i)
                for j in 1:n_basefuncs_s
                    u = face_shape_value(Wh, face_idx, q_point, j)
                    # Integral_∂T u ⋅ w dS
                    Se[i,j] += (u*w) * dS
                end
            end
        end
    end
    @test Me ≈ 0.5*one(Me)
    @test Ce ≈ Ce_ex[cell_idx]
    @test Se ≈ Se_ex[cell_idx]
    # Third equation matrices
    for face_idx in 1:numfaces(mesh.cells[cell_idx])
        for q_point in 1:getnfacequadpoints(Wh)
            dS = getdetJdS(Mh, cell_idx, face_idx, q_point)
            orientation = face_orientation(mesh.cells[cell_idx], face_idx)
            for i in 1:n_basefuncs
                u = face_shape_value(Vh, face_idx, q_point, i, orientation)
                n = get_normal(mesh.cells[cell_idx], face_idx)
                for j in 1:n_basefuncs_t
                    w = shape_value(Mh, q_point, j)
                    # Integral_∂T u ⋅ w dS
                    Ee[i,n_basefuncs_t*(face_idx-1)+j] += (w*(u⋅n)) * dS
                end
            end
        end
    end
    @test Ee ≈ Ee_ex[cell_idx]
end
