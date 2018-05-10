module HDiscontinuousGalerkin

using Tensors
using PolynomialBases

abstract type AbstractRefShape end
abstract type AbstractQuadratureRule end

struct RefTetrahedron <: AbstractRefShape end

abstract type Interpolation{dim,shape,order} end

include("mesh.jl")
include("basis.jl")
include("quadrature.jl")
include("GrundmannMoellerQuad.jl")

# Function exports
# mesh
export parse_mesh_triangle, element_diameter, get_coordinates
export nodes, faces, elements, get_elements, get_faces, get_nodes, node
export Element, Node, Face, PolygonalMesh

# Quadratures

export QuadratureRule
export RefTetrahedron
export GrundmannMoeller
export getpoints, getweights

# basis
export Dubiner
export dubiner_basis, ∇dubiner_basis
export value, derivative, gradient_value
end # module
