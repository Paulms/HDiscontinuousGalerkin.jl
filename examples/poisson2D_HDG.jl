# Here we solve the equation on a unit square, with a uniform internal source.
# The strong form of the (linear) heat equation is given by
#
# ```math
#  -\nabla \cdot (k \nabla u) = f  \quad x \in \Omega,
# ```
#
# where $u$ is the unknown temperature field, $k$ the heat conductivity,
# $f$ the heat source and $\Omega$ the domain. For simplicity we set $f = 1$
# and $k = 1$. We will consider homogeneous Dirichlet boundary conditions such that
# ```math
# u(x) = 0 \quad x \in \partial \Omega,
# ```
# where $\partial \Omega$ denotes the boundary of $\Omega$.
#
# The HDG weak form is given by
# ```math
#(\vec{\sigma}_{h},\vec{v_{h}})_{\tau_{h}}-(u_{h},\nabla_{h}\cdot\vec{v}_{h})_{\tau_{h}}+\left\langle \hat{u}_{h},\vec{v}_{h}\cdot\vec{n}_{h}\right\rangle _{\partial\tau_{h}} & =0\quad\forall\vec{v}_{h}\in\vec{V}_{h},\\
#(w_{h},\nabla_{h}\cdot\vec{\sigma}_{h})_{\tau_{h}}+\left\langle (\hat{\sigma}_{h}-\vec{\sigma}_{h})\cdot\vec{n},w_{h}\right\rangle _{\partial\tau_{h}} & =(f,w_{h})_{\tau_{h}}\quad\forall w_{h}\in W_{h},\\
#\left\langle \vec{\sigma}_{h},\xi_{h}\right\rangle _{\partial\tau_{h}\backslash\Gamma} & =0\quad\forall\xi_{h}\in M_{h}^{o},\\
#\left\langle \hat{u}_{h},\xi_{h}\right\rangle _{\Gamma} & =\left\langle g,\xi_{h}\right\rangle _{\Gamma}\quad\forall\xi_{h}\in M_{h}^{\partial},\\
#\hat{\sigma}_{h}\cdot\vec{n} & =\vec{\sigma}_{h}\cdot\vec{n}+\tau(u_{h}-\hat{u}_{h}).
# ```
#-

using HDiscontinuousGalerkin
using Tensors
using BlockArrays
using LinearAlgebra

# Load mesh
#To load mesh from triangle
#root_file = "mesh/figure.1"
#@time mesh = parse_mesh_triangle(root_file)

#or use internal mesh
mesh = rectangle_mesh(TriangleCell, (10,10), Vec{2}((0.0,0.0)), Vec{2}((1.0,1.0)))

# ### Trial and test functions
dim = 2
finiteElement = GenericFiniteElement(Dubiner{dim,RefTetrahedron,1}())
Wh = ScalarFunctionSpace(mesh, finiteElement)
Vh = VectorFunctionSpace(mesh, finiteElement)
Mh = ScalarTraceFunctionSpace(Wh, GenericFiniteElement(Legendre{dim-1,RefTetrahedron,1}()))

# Variables
û_h = TrialFunction(Mh)
σ_h = TrialFunction(Vh)
u_h = TrialFunction(Wh)

# ### Boundary conditions
dbc = Dirichlet(û_h, mesh, "boundary", x -> 0)

# RHS function
f(x::Vec{dim}) = 2*π^2*sin(π*x[1])*sin(π*x[2])
# ### Assembling the linear system
# Now we have all the pieces needed to assemble the linear system, $K û = f$.
function doassemble(Vh, Wh, Mh, τ = 1.0)
    # Allocate Matrices
    n_basefuncs = getnbasefunctions(Vh)
    n_basefuncs_s = getnbasefunctions(Wh)
    n_basefuncs_t = getnbasefunctions(Mh)
    Ae = zeros(n_basefuncs, n_basefuncs)
    Be = zeros(n_basefuncs, n_basefuncs_s)
    Ce = zeros(n_basefuncs_s, n_basefuncs_s)
    Ee = zeros(n_basefuncs,3*n_basefuncs_t)
    Fe = zeros(n_basefuncs_s, 3*n_basefuncs_t)
    be = zeros(n_basefuncs_s,1)
    He = zeros(3*n_basefuncs_t, 3*n_basefuncs_t)

    # create a matrix assembler and rhs vector
    assembler = start_assemble(getnfaces(mesh)*n_basefuncs_t)
    rhs = zeros(Float64, getnfaces(mesh)*n_basefuncs_t)
    K_element = Array{AbstractMatrix{Float64}}(undef, getncells(mesh))
    b_element = Array{AbstractVector{Float64}}(undef, getncells(mesh))

    @inbounds for (cell_idx, cell) in enumerate(CellIterator(mesh))
        reinit!(Wh, cell)
        reinit!(Vh, cell)
        fill!(Ae, 0)
        fill!(Be, 0)
        fill!(Ce, 0)
        fill!(Ee, 0)
        fill!(Fe, 0)
        fill!(He, 0)
        fill!(be, 0)
        #Cell integrals
        for q_point in 1:getnquadpoints(Vh)
            dΩ = getdetJdV(Vh, q_point)
            for i in 1:n_basefuncs
                vh  = shape_value(Vh, q_point, i)
                div_vh = shape_divergence(Vh, q_point, i)
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
        #RHS
        for q_point in 1:getnquadpoints(Wh)
            dΩ = getdetJdV(Wh, q_point)
            for i in 1:n_basefuncs_s
                w  = shape_value(Wh, q_point, i)
                fh = function_value(f, Wh, cell, q_point)
                # Integral f*u dΩ
                be[i] += fh*w*dΩ
            end
        end
        #Face integrals
        for face_idx in 1:getnfaces(cell)
            for q_point in 1:getnfacequadpoints(Wh)
                dS = getfacedetJdS(Wh, face_idx, q_point)
                orientation = face_orientation(cell, face_idx)
                for i in 1:n_basefuncs_s
                    w = shape_value(Wh, face_idx, q_point, i)
                    for j in 1:n_basefuncs_s
                        u = shape_value(Wh, face_idx, q_point, j)
                        # Integral_∂T τ*u ⋅ w dS
                        Ce[i,j] += τ*(u*w) * dS
                    end
                    w = shape_value(Wh, face_idx, q_point, i, orientation)
                    for j in 1:n_basefuncs_t
                        û = shape_value(Mh, q_point, j)
                        # Integral_∂T τ*û*w  dS
                        Fe[i,n_basefuncs_t*(face_idx-1)+j] += (τ*(û*w)) * dS
                    end
                end
                dS = getfacedetJdS(Mh, face_idx, q_point)
                n = get_normal(Vh, face_idx)
                for i in 1:n_basefuncs
                    v = shape_value(Vh, face_idx, q_point, i, orientation)
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
Etu_h <= 0.00006

### Plot Solution
# TODO: Turn into a recipe
using Makie, CairoMakie
#AbstractPlotting.__init__()
using ColorSchemes
connectivity = getcells_matrix(mesh)
coordinates = get_vertices_matrix(mesh)

# We avg since u_h is discontinuous
color = nodal_avg(u_h)
scene = Scene(resolution = (400, 200), colorrange = (0.0, 1.0), colormap = ColorSchemes.viridis.colors)
#poly!(scene, coordinates, connectivity, color = color, strokecolor = (:black, 0.6), strokewidth = 1)
for row in 1:size(connectivity)[1]
	#read coordinates
	points = node(:poly, Point2f0[coordinates[i,:] for i in connectivity[row,:]])
	colors = get(ColorSchemes.viridis, [color[i] for i in connectivity[row,:]])
    #colors = get(ColorSchemes.viridis, [value(u_h, node, row) for node in connectivity[row,:]])
	poly!(scene, points, strokewidth = 1, strokecolor = (:black,0.6), color = colors, show_axis = true, scale_plot = false)
end
display(scene)
