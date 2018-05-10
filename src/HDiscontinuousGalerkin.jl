module HDiscontinuousGalerkin

using StaticArrays
using PolynomialBases

abstract type AbstractRefShape end
abstract type AbstractQuadratureRule end

struct RefSimplex <: AbstractRefShape end
struct GrundmannMoeller <: AbstractQuadratureRule end

struct QuadratureRule{dim,shape,T}
    weights::Vector{T}
    points::Vector{SVector{dim,T}}
end

abstract type Interpolation{dim,shape,order} end

include("mesh.jl")
include("basis.jl")
include("GrundmannMoellerQuad.jl")

# Function exports
# mesh
export parse_mesh_triangle, element_diameter, get_coordinates
export nodes, faces, elements, get_elements, get_faces, get_nodes, node
export Element, Node, Face, PolygonalMesh

# Quadratures

export QuadratureRule
export RefSimplex
export GrundmannMoeller

# basis
export Dubiner
export dubiner_basis, ∇dubiner_basis
export value, derivative, gradient_value
end # module
