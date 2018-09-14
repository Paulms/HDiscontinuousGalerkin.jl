# # Heat Equation
#
# ## Introduction
#
# The heat equation is the "Hello, world!" equation of finite elements.
# Here we solve the equation on a unit square, with a uniform internal source.
# The strong form of the (linear) heat equation is given by
#
# ```math
#  -\nabla \cdot (k \nabla u) = f  \quad x \in \Omega,
# ```
#
# where $u$ is the unknown temperature field, $k$ the heat conductivity,
# $f$ the heat source and $\Omega$ the domain. For simplicity we set
# $k = 1$. We will consider homogeneous Dirichlet boundary conditions such that
# ```math
# u(x) = 0 \quad x \in \partial \Omega,
# ```
# where $\partial \Omega$ denotes the boundary of $\Omega$.
#
# The resulting weak form is given by
# ```math
# \int_{\Omega} \nabla v \cdot \nabla u \ d\Omega = \int_{\Omega} f\cdot v \ d\Omega,
# ```
# where $v$ is a suitable test function.
#-
# ## Commented Program
using HDiscontinuousGalerkin
using Tensors
using SparseArrays

# We start  generating a simple grid with 20x20 quadrilateral elements
# using `generate_grid`. The generator defaults to the unit square,
# so we don't need to specify the corners of the domain.
mesh = rectangle_mesh(TriangleCell, (10,10), Vec{2}((0.0,0.0)), Vec{2}((1.0,1.0)));

# ### Initiate function Spaces
dim = 2
Wh = ScalarFunctionSpace(mesh, ContinuousLagrange{dim,RefTetrahedron,1}(); update_face_data = false)

# Declare variables
u_h = TrialFunction(Wh)

# ### Degrees of freedom
# Next we need to define a `DofHandler`, which will take care of numbering
# and distribution of degrees of freedom for our approximated fields.
# We create the `DofHandler` and then add a single field called `u`.
# Lastly we `close!` the `DofHandler`, it is now that the dofs are distributed
# for all the elements.
dh = DofHandler([u_h], mesh);

# Now that we have distributed all our dofs we can create our tangent matrix,
# using `create_sparsity_pattern`. This function returns a sparse matrix
# with the correct elements stored.
K = create_sparsity_pattern(dh);

# We can inspect the pattern using the `spy` function from `UnicodePlots.jl`.
# By default the stored values are set to $0$, so we first need to
# fill the stored values, e.g. `K.nzval` with something meaningful.
using UnicodePlots
fill!(K.nzval, 1.0);
spy(K; height = 15)

# ### Boundary conditions
dbc = Dirichlet(u_h, dh, "boundary", [0.0])

# ### RHS function
f(x::Vec{dim}) = 2*π^2*sin(π*x[1])*sin(π*x[2])

# ### Assembling the linear system
# Now we have all the pieces needed to assemble the linear system, $K u = f$.
function doassemble(Wh, K::SparseMatrixCSC, dh::DofHandler)
    # We allocate the element stiffness matrix and element force vector
    # just once before looping over all the cells instead of allocating
    # them every time in the loop.
    n_basefuncs = getnbasefunctions(Wh)
    Ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)
    b = zeros(ndofs(dh))
    cell_dofs = Vector{Int}(undef, ndofs_per_cell(dh))

    # Next we define the global force vector `f` and
    # create an assembler. The assembler
    # is just a thin wrapper around `f` and `K` and some extra storage
    # to make the assembling faster.
    assembler = start_assemble(K, b)
    ff = interpolate(f, Wh)

    # It is now time to loop over all the cells in our grid.
    @inbounds for (cellcount, cell) in enumerate(CellIterator(mesh))
        # Always remember to reset the element stiffness matrix and
        # force vector since we reuse them for all elements.
        fill!(Ke, 0)
        fill!(fe, 0)
        reinit!(Wh, cell)

        # It is now time to loop over all the quadrature points in the cell and
        # assemble the contribution to `Ke` and `fe`. The integration weight
        # can be queried from `cellvalues` by `getdetJdV`.
        for q_point in 1:getnquadpoints(Wh)
            dΩ = getdetJdV(Wh, q_point)
            fh = value(ff, cell, q_point)
            # For each quadrature point we loop over all the (local) shape functions.
            # We need the value and gradient of the testfunction `v` and also the gradient
            # of the trial function `u`. We get all of these from `cellvalues`.
            for i in 1:n_basefuncs
                v  = shape_value(Wh, q_point, i)
                ∇v = shape_gradient(Wh, q_point, i)
                fe[i] += fh*v * dΩ
                for j in 1:n_basefuncs
                    ∇u = shape_gradient(Wh, q_point, j)
                    Ke[i, j] += (∇v ⋅ ∇u) * dΩ
                end
            end
        end
        # The last step in the element loop is to assemble `Ke` and `fe`
        # into the global `K` and `f` with `assemble!`.
        celldofs!(cell_dofs, dh, cell)
        assemble!(assembler, cell_dofs, fe, Ke)
    end
    return K, b
end
#md nothing # hide

# ### Solution of the system
# The last step is to solve the system. First we call `doassemble`
# to obtain the global stiffness matrix `K` and force vector `f`.
K, b = doassemble(Wh, K, dh);

# To account for the boundary conditions we use the `apply!` function.
# This modifies elements in `K` and `f` respectively, such that
# we can get the correct solution vector `u` by using `\`.
apply!(K,b,dbc);
u = K \ b;

# reconstruct!(u_h, u, dh)
# #
# # # ### Compute errors
# u_ex(x::Vec{dim}) = sin(π*x[1])*sin(π*x[2])
# Etu_h = errornorm(u_h, u_ex)
# Etu_h <= 0.0002
#
# # ### Plot Solution
# #Plot mesh
# using PyCall
# using PyPlot
# @pyimport matplotlib.tri as mtri
# m_nodes = get_vertices_matrix(mesh)
# triangles = get_cells_matrix(mesh)
# triang = mtri.Triangulation(m_nodes[:,1], m_nodes[:,2], triangles)
# PyPlot.triplot(triang, "ko-")
#
# nodalu_h = Vector{Float64}(length(mesh.nodes))
# share_count = zeros(Int,length(mesh.nodes))
# fill!(nodalu_h,0)
# for (k,cell) in enumerate(mesh.cells)
#     for node in cell.nodes
#         u = value(u_h, k, mesh.nodes[node].x)
#         nodalu_h[node] += u
#         share_count[node] += 1
#     end
# end
# nodalu_h = nodalu_h./share_count
# u_ex_i = sin.(π*m_nodes[:,1]).*sin.(π*m_nodes[:,2])
# nodalu_h
# nuh = [abs(x) < eps() ? 0.0 : x for x in nodalu_h]
# PyPlot.tricontourf(triang, nuh)
