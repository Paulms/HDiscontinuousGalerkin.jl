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

# Load mesh
root_file = "mesh/figure2.1"
mesh = parse_mesh_triangle(root_file)

# ### Trial and test functions
dim = 2

Vh = VectorFunctionSpace(mesh, Dubiner{dim,RefTetrahedron,1}())
Wh = ScalarFunctionSpace(mesh, Dubiner{dim,RefTetrahedron,1}())
Mh = ScalarFunctionSpace(mesh, Legendre{dim-1,RefTetrahedron,1}())

# dh = DofHandler(mesh)
# push!(dh, :u, Wh)
# push!(dh, :sigma, Vh)
# push!(dh, :ut, Mh)
# close!(dh);

# ### Boundary conditions
# ch = ConstraintHandler(dh);
# ∂Ω = union(getfaceset.(grid, ["boundary"])...);
# dbc = Dirichlet(:u, ∂Ω, (x, t) -> 0)
# add!(ch, dbc);
# close!(ch)
# update!(ch, 0.0);

# ### Assembling the linear system
# Now we have all the pieces needed to assemble the linear system, $K u = f$.
#function doassemble(dh::DofHandler)
    # Allocate Matrices
    n_basefuncs = 6 #getnbasefunctions(dh, :sigma)
    Me = zeros(n_basefuncs, n_basefuncs)
    #fe = zeros(n_basefuncs)

    # Next we define the global force vector `f` and use that and
    # the stiffness matrix `K` and create an assembler.
    #f = zeros(ndofs(dh))
    #assembler = start_assemble(K, f)

    #@inbounds for cell_idx in 1:numcells(mesh)
        # Always remember to reset the element stiffness matrix and
        # force vector since we reuse them for all elements.
        fill!(Me, 0)
        #fill!(fe, 0)
        cell_idx = 1
        for q_point in 1:getnquadpoints(Vh)
            dΩ = getdetJdV(Vh, cell_idx, q_point)
            # For each quadrature point we loop over all the (local) shape functions.
            # We need the value and gradient of the testfunction `v` and also the gradient
            # of the trial function `u`.
            for i in 1:n_basefuncs
                vh  = shape_value(Vh, q_point, i)
                #fe[i] += v * dΩ
                for j in 1:n_basefuncs
                    sigma = shape_value(Vh, q_point, j)
                    Me[i, j] += (sigma ⋅ vh) * dΩ
                end
            end
        end

        # The last step in the element loop is to assemble `Ke` and `fe`
        # into the global `K` and `f` with `assemble!`.
        # assemble!(assembler, celldofs(cell), fe, Ke)
        println(Me)
    #end
    #return K, f
#end
#md nothing # hide

# ### Solution of the system
K, f = doassemble(cellvalues, K, dh);

# To account for the boundary conditions we use the `apply!` function.
# This modifies elements in `K` and `f` respectively, such that
# we can get the correct solution vector `u` by using `\`.
apply!(K, f, ch)
u = K \ f;

# ### Exporting to VTK
# To visualize the result we export the grid and our field `u`
# to a VTK-file, which can be viewed in e.g. [ParaView](https://www.paraview.org/).
vtk_grid("heat_equation", dh) do vtk
    vtk_point_data(vtk, dh, u)
end

## test the result                #src
using Base.Test                   #src
@test norm(u) ≈ 3.307743912641305 #src
