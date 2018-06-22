__precompile__()

module HDiscontinuousGalerkin

using Tensors
using FastGaussQuadrature
import Base:@propagate_inbounds

abstract type AbstractRefShape end
abstract type AbstractQuadratureRule end

abstract type Interpolation{dim,shape,order} end
abstract type DiscreteFunctionSpace{dim,T,refshape} end

include("shapes.jl")
include("mesh.jl")
include("generate_mesh.jl")
include("triangle_mesh.jl")
include("basis.jl")
include("quadrature.jl")
include("utils.jl")
include("GrundmannMoellerQuad.jl")
include("StrangQuad.jl")
include("FunctionSpace.jl")
include("assembler.jl")
include("DiscreteFunctions.jl")
include("dofhandler.jl")
include("boundary.jl")


# Function exports
# mesh
export parse_mesh_triangle, cell_diameter, get_coordinates
export nodes, faces, cells, get_cells, get_faces, get_nodes, node
export Cell, Node, Face, PolygonalMesh
export getncells, getnfaces, n_faces_per_cell, get_normal
export face_orientation, get_faceset
export rectangle_mesh
export TriangleCell
export reference_edge_nodes, getnnodes, get_cell_name
export get_vertices_matrix, get_cells_matrix

# Boundaries
export Dirichlet
export apply!

# Shapes
export get_nodal_points, volume
export get_num_faces
export get_num_vertices

# Quadratures
export QuadratureRule
export RefTetrahedron
export DefaultQuad, GrundmannMoeller, Strang, GaussLegendre
export getpoints, getweights

# basis
export Dubiner, Lagrange, jacobi, Legendre
export dubiner_basis, ∇dubiner_basis
export value, derivative, gradient_value
export getnbasefunctions
export get_default_geom_interpolator
export getorder, getlowerdiminterpol
export get_topology, get_interpolation

#utils
export get_affine_map, integrate, zcheck!, zcheck

#Assembler
export start_assemble, assemble!, end_assemble

#FunctionSpaces
export VectorFunctionSpace, ScalarFunctionSpace, ScalarTraceFunctionSpace
export getnquadpoints, getdetJdV, shape_value
export shape_gradient, shape_divergence
export getnfacequadpoints, getdetJdS
export face_shape_value
export InterpolatedFunction, function_value, interpolate
export spatial_coordinate, reference_coordinate
export getnlocaldofs

#Discrete Functions
export TrialFunction
export errornorm

#Handlers
export DofHandler, ndofs, ndofs_per_cell, celldofs!
export close!
export create_sparsity_pattern, reconstruct!
export dof_range

end # module
