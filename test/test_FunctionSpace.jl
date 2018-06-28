# Test using Poisson problem
# -Δu = f  in Ω
#   u = 0  on Γ = ∂Ω
# where f = sin(πx)⋅sin(πy)

using HDiscontinuousGalerkin
using Tensors
using BlockArrays

# Load mesh
root_file = "mesh/figure2.1"
mesh = parse_mesh_triangle(root_file)

# ### Trial and test functions
dim = 2
Vh = VectorFunctionSpace(mesh, Dubiner{dim,RefTetrahedron,1}())
Wh = ScalarFunctionSpace(mesh, Dubiner{dim,RefTetrahedron,1}())
Mh = ScalarTraceFunctionSpace(Wh, Legendre{dim-1,RefTetrahedron,1}())

# Variables
û_h = TrialFunction(Mh)
σ_h = TrialFunction(Vh)
u_h = TrialFunction(Wh)

# Basic Test
@test getnlocaldofs(Vh) == 6
@test getnlocaldofs(Wh) == 3
@test getnlocaldofs(Mh) == 6
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
@test Wh.E[1,:] ≈ [[sq2, sq2, sq2],[sq2, sq2,sq2]]
@test Wh.detJf[1,:,1] ≈ [sq2/2,sq2/2,1]
@test Wh.detJf[2,:,1] ≈ [sq2/2,sq2/2,1]
@test Wh.detJf[3,:,1] ≈ [1,sq2/2,sq2/2]
@test Wh.detJf[4,:,1] ≈ [sq2/2,sq2/2,1]
@test Wh.normals[1,:,:]≈ [Vec{2}([-sq2/2,sq2/2]), Vec{2}([-sq2/2,-sq2/2]), Vec{2}([1.0,0.0])]

# ### Boundary conditions
dbc = Dirichlet(û_h, mesh, "boundary", x -> 0)

# RHS function
f(x::Vec{dim}) = 2*π^2*sin(π*x[1])*sin(π*x[2])
ff = interpolate(f, Wh)
@test errornorm(ff,f) <= eps(Float64)

Be_ex=Vector{Matrix{Float64}}(4)
Be_ex[1] = [0 0 0; 0 0 0; -3*sq2 0 0; 0.0 0 0; sqrt(6) 0 0; 0 0 0]
Be_ex[2] = [0 0 0; 0 0 0; 3*sq2 0 0; 0.0 0 0; -sqrt(6) 0 0; 0 0 0]
Be_ex[3] = [0 0 0; sqrt(6)/2 0 0; -3/2*sq2 0 0; 0 0 0; 3/2*sqrt(6) 0 0; 3/2*sq2 0 0]
Be_ex[4] = [0 0 0; sqrt(6) 0 0; 0 0 0; 0.0 0 0; 0 0 0; 3*sq2 0 0]
Ce_ex=Vector{Matrix{Float64}}(4)
Ce_ex[1] = [2*sq2+2 0 2-2*sq2; 0 4*sq2+4 0; 2-2*sq2 0 4*sq2+4]
Ce_ex[2] = [2*sq2+2 0 2-2*sq2; 0 4*sq2+4 0; 2-2*sq2 0 4*sq2+4]
Ce_ex[3] = [2*sq2+2 sqrt(6)-sqrt(3) sq2-1; sqrt(6)-sqrt(3)  4*sq2+4 0; sq2-1 0 4*sq2+4]
Ce_ex[4] = [2*sq2+2 0 2-2*sq2; 0 4*sq2+4 0; 2-2*sq2 0 4*sq2+4]
Ee_ex=Vector{Matrix{Float64}}(4)
Ee_ex[1] = -[sq2/2 0 sq2/2 0 -sq2 0; sq3/2 -1/2 -sq3/2 1/2 0 -2; 1/2 sq3/2 1/2 sq3/2 2 0;
           -sq2/2 0 sq2/2 0 0 0; -sq3/2 1/2 -sq3/2  1/2 0 0; -1/2 -sq3/2 1/2 sq3/2 0 0]
Ee_ex[2] = -[-sq2/2 0 -sq2/2 0 sq2 0; -sq3/2 1/2 sq3/2 -1/2 0 -2; -1/2 -sq3/2 -1/2 -sq3/2 -2 0;
          sq2/2 0 -sq2/2 0 0 0; sq3/2 -1/2 sq3/2  -1/2 0 0; 1/2 sq3/2 -1/2 -sq3/2 0 0]
Ee_ex[3] = -[0 0 sq2/2 0 -sq2/2 0; 0 0 -sq3/2  -1/2 0 1; 0 0 1/2 -sq3/2 1 0;
            -sq2 0 sq2/2 0 sq2/2 0; -sq3 1 -sq3/2 -1/2 0 -1; -1 -sq3 1/2 -sq3/2 -1 0]
Ee_ex[4] = -[-sq2/2 0 sq2/2 0 0 0; -sq3/2 1/2 -sq3/2 1/2 0 0; -1/2 -sq3/2 1/2 sq3/2 0 0;
           -sq2/2 0 -sq2/2 0 sq2 0; -sq3/2 1/2 sq3/2  -1/2 0 2; -1/2 -sq3/2 -1/2 -sq3/2 -2 0]
He_ex=Vector{Matrix{Float64}}(4)
He_ex[1] = [sq2/2 0 0 0 0 0 ;0 sq2/2 0 0 0 0;0 0 sq2/2 0 0 0;0 0 0 sq2/2 0 0; 0 0 0 0 1 0; 0 0 0 0 0 1]
He_ex[3] = [1 0 0 0 0 0 ;0 1 0 0 0 0;0 0 sq2/2 0 0 0;0 0 0 sq2/2 0 0; 0 0 0 0 sq2/2 0; 0 0 0 0 0 sq2/2]
He_ex[2] = He_ex[1]
He_ex[4] = He_ex[1]

function doassemble(Vh, Wh, Mh, τ = 1)
    # Allocate Matrices
    n_basefuncs = getnbasefunctions(Vh)
    @test n_basefuncs == 6
    n_basefuncs_s = getnbasefunctions(Wh)
    @test n_basefuncs_s == 3
    n_basefuncs_t = getnbasefunctions(Mh)
    @test n_basefuncs_t == 2
    Ae = zeros(n_basefuncs, n_basefuncs)
    Be = zeros(n_basefuncs, n_basefuncs_s)
    Ce = zeros(n_basefuncs_s, n_basefuncs_s)
    Ee = zeros(n_basefuncs,3*n_basefuncs_t)
    Fe = zeros(n_basefuncs_s, 3*n_basefuncs_t)
    be = zeros(n_basefuncs_s,1)
    He = zeros(3*n_basefuncs_t, 3*n_basefuncs_t)

    # create a matrix assembler and rhs vector
    assembler = start_assemble(getnfaces(mesh)*n_basefuncs_t)
    rhs = Array{Float64}(getnfaces(mesh)*n_basefuncs_t)
    fill!(rhs,0)
    K_element = Array{AbstractMatrix{Float64}}(getncells(mesh))
    b_element = Array{AbstractVector{Float64}}(getncells(mesh))

    for cell_idx in 1:getncells(mesh)
        fill!(Ae, 0)
        fill!(Be, 0)
        fill!(Ce, 0)
        fill!(Ee, 0)
        fill!(Fe, 0)
        fill!(He, 0)
        fill!(be, 0)
        #Cell integrals
        for q_point in 1:getnquadpoints(Vh)
            dΩ = getdetJdV(Vh, cell_idx, q_point)
            for i in 1:n_basefuncs
                vh  = shape_value(Vh, q_point, i)
                div_vh = shape_divergence(Vh, q_point, i, cell_idx)
                for j in 1:n_basefuncs
                    σ = shape_value(Vh, q_point, j)
                    # Integral σ⋅v dΩ
                    Ae[i, j] += (σ ⋅ vh) * dΩ
                end
                for j in 1:n_basefuncs_s
                    u = shape_value(Wh, q_point, j)
                    # Integral u*∇⋅v dΩ
                    Be[i,j] += (u*div_vh) * dΩ
                end
            end
        end
        @test Ae ≈ 0.5*one(Ae)
        @test Be ≈ Be_ex[cell_idx]
        #RHS
        for q_point in 1:getnquadpoints(Wh)
            dΩ = getdetJdV(Wh, cell_idx, q_point)
            for i in 1:n_basefuncs_s
                w  = shape_value(Wh, q_point, i)
                fh = value(ff, cell_idx,q_point)
                # Integral f*u dΩ
                be[i] += fh*w*dΩ
            end
        end
        #Face integrals
        for face_idx in 1:getnfaces(mesh.cells[cell_idx])
            for q_point in 1:getnfacequadpoints(Wh)
                dS = getdetJdS(Wh, cell_idx, face_idx, q_point)
                orientation = face_orientation(mesh, cell_idx, face_idx)
                for i in 1:n_basefuncs_s
                    w = face_shape_value(Wh, face_idx, q_point, i)
                    for j in 1:n_basefuncs_s
                        u = face_shape_value(Wh, face_idx, q_point, j)
                        # Integral_∂T τ*u ⋅ w dS
                        Ce[i,j] += τ*(u*w) * dS
                    end
                    w = face_shape_value(Wh, face_idx, q_point, i, orientation)
                    for j in 1:n_basefuncs_t
                        û = shape_value(Mh, q_point, j)
                        # Integral_∂T τ*û*w  dS
                        Fe[i,n_basefuncs_t*(face_idx-1)+j] += (τ*(û*w)) * dS
                    end
                end
                dS = getdetJdS(Mh, cell_idx, face_idx, q_point)
                n = get_normal(Vh, cell_idx, face_idx)
                for i in 1:n_basefuncs
                    v = face_shape_value(Vh, face_idx, q_point, i, orientation)
                    for j in 1:n_basefuncs_t
                        û = shape_value(Mh, q_point, j)
                        # Integral_∂T û(v⋅n)  dS
                        Ee[i,n_basefuncs_t*(face_idx-1)+j] += (û*(v⋅n)) * dS
                    end
                end
                for i in 1:n_basefuncs_t
                    μ = shape_value(Mh, q_point, i)
                    for j in 1:n_basefuncs_t
                        û = shape_value(Mh, q_point, j)
                        # Integral_∂T û*μ  dS
                        He[n_basefuncs_t*(face_idx-1)+i,n_basefuncs_t*(face_idx-1)+j] += (û*μ) * dS
                    end
                end
            end
        end
        @test Ce ≈ Ce_ex[cell_idx]
        @test Ee ≈ Ee_ex[cell_idx]
        @test He ≈ He_ex[cell_idx]
        #Assamble Ke
        Me = BlockArray{Float64}(undef, [n_basefuncs,n_basefuncs_s], [n_basefuncs,n_basefuncs_s])
        setblock!(Me, Ae, 1, 1)
        setblock!(Me, Be', 2, 1)
        setblock!(Me, -Be, 1, 2)
        setblock!(Me, Ce, 2, 2)
        Mei = factorize(Array(Me))
        EFe = BlockArray{Float64}(undef, [n_basefuncs,n_basefuncs_s], [3*n_basefuncs_t])
        setblock!(EFe, -Ee, 1, 1)
        setblock!(EFe, Fe, 2, 1)
        Ge = BlockArray{Float64}(undef, [n_basefuncs,n_basefuncs_s], [3*n_basefuncs_t])
        setblock!(Ge, Ee, 1, 1)
        setblock!(Ge, Fe, 2, 1)
        Ge = Ge'
        Bte = BlockArray{Float64}(undef, [n_basefuncs,n_basefuncs_s],[1])
        setblock!(Bte, zeros(n_basefuncs,1),1,1)
        setblock!(Bte, be,2,1)
        K_element[cell_idx]=Mei\EFe
        Ate = Ge*K_element[cell_idx]-He
        b_element[cell_idx]=(Mei\Bte)[:,1]
        bte = -Ge*b_element[cell_idx]
        #Assemble
        gdof = Vector{Int}()
        for fidx in mesh.cells[cell_idx].faces
            for j in 1:n_basefuncs_t
                push!(gdof, fidx*n_basefuncs_t-(n_basefuncs_t-j))
            end
        end
        assemble!(assembler, gdof, Ate)
        assemble!(rhs, gdof, bte[:,1])
    end
    return end_assemble(assembler), rhs, K_element, b_element
end

### Solution of the system
K, b, K_e, b_e = doassemble(Vh,Wh,Mh);

# To account for the boundary conditions we use the `apply!` function.
# This modifies elements in `K` and `f` respectively, such that
# we can get the correct solution vector `u` by using `\`.
apply!(K,b,dbc)
û = K \ b;
#Now we recover original variables from skeleton û
function get_uσ!(σ_h,u_h,û_h,û, K_e, b_e, mesh)
    n_cells = getncells(mesh)
    n_basefuncs = getnbasefunctions(σ_h)
    n_basefuncs_t = getnbasefunctions(Mh)
    for cell_idx in 1:getncells(mesh)
        #get dofs
        û_e = Vector{Float64}()
        for (k,face) in enumerate(mesh.cells[cell_idx].faces)
            push!(û_e, û[n_basefuncs_t*face-1:n_basefuncs_t*face]...)
            û_h.m_values[cell_idx,:,k] = û[n_basefuncs_t*face-1:n_basefuncs_t*face]
        end
        dof = K_e[cell_idx]*û_e+b_e[cell_idx]
        σ_h.m_values[cell_idx,:] = dof[1:n_basefuncs]
        u_h.m_values[cell_idx,:] = dof[n_basefuncs+1:end]
    end
end

get_uσ!(σ_h, u_h,û_h,û, K_e, b_e, mesh)
#Compute errors
u_ex(x::Vec{dim}) = sin(π*x[1])*sin(π*x[2])
Etu_h = errornorm(u_h, u_ex)
@test Etu_h <= 0.12
