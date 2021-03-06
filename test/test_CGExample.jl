@testset "Test CG poisson example" begin

using HDiscontinuousGalerkin
using Tensors
using SparseArrays

mesh = rectangle_mesh(TriangleCell, (10,10), Vec{2}((0.0,0.0)), Vec{2}((1.0,1.0)))

# ### Initiate function Spaces
dim = 2
Wh = ScalarFunctionSpace(mesh, ContinuousLagrange{dim,RefTetrahedron,1}(); update_face_data = false)

# Declare variables
u_h = TrialFunction(Wh)

# ### Degrees of freedom
dh = DofHandler([u_h],mesh)
K = create_sparsity_pattern(dh);

# ### Boundary conditions
dbc = Dirichlet(u_h, dh, "boundary", x -> 0)

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
            fh = function_value(f, Wh, cell, q_point)
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

apply!(K,b,dbc)
u = K \ b;

reconstruct!(u_h, u, dh)

# ### Compute errors
u_ex(x::Vec{dim}) = sin(π*x[1])*sin(π*x[2])
Etu_h = errornorm(u_h, u_ex)
@test Etu_h <= 0.0002

end
