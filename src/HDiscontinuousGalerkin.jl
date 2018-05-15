module HDiscontinuousGalerkin

using Tensors
using PolynomialBases

abstract type AbstractRefShape end
abstract type AbstractQuadratureRule end

abstract type Interpolation{dim,shape,order} end

include("utils.jl")
include("mesh.jl")
include("shapes.jl")
include("basis.jl")
include("quadrature.jl")
include("GrundmannMoellerQuad.jl")

# Function exports
# mesh
export parse_mesh_triangle, cell_diameter, get_coordinates
export nodes, faces, cells, get_cells, get_faces, get_nodes, node
export Cell, Node, Face, PolygonalMesh

# Shapes
export get_nodal_points

# Quadratures
export QuadratureRule
export RefTetrahedron
export GrundmannMoeller
export getpoints, getweights

# basis
export Dubiner, Lagrange
export dubiner_basis, ∇dubiner_basis
export value, derivative, gradient_value
export getnbasefunctions

#utils
export get_affine_map

end # module
