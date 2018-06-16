# Interpolated Functions
struct InterpolatedFunction{dim,T}
    N::Matrix{T}
    fs::DiscreteFunctionSpace{dim,T}
end

@inline value(ifunc::InterpolatedFunction, cell::Int,q_point::Int) = ifunc.N[cell, q_point]

"""
function spatial_coordinate(fs::ScalarFunctionSpace{dim}, q_point::Int, x::AbstractVector{Vec{dim,T}})
Map coordinates of quadrature point `q_point` of Scalar Function Space `fs`
into domain with vertices `x`
"""
function spatial_coordinate(fs::ScalarFunctionSpace{dim}, q_point::Int, x::AbstractVector{Vec{dim,T}}) where {dim,T}
    n_base_funcs = getngeobasefunctions(fs)
    @assert length(x) == n_base_funcs
    vec = zero(Vec{dim,T})
    @inbounds for i in 1:n_base_funcs
        vec += geometric_value(fs, q_point, i) * x[i]
    end
    return vec
end

"""
function interpolate(f::Function, fs::ScalarFunctionSpace{dim,T}, mesh::PolygonalMesh) where {dim,T}
Interpolation of scalar functions f(x): x ∈ ℝⁿ ↦ ℝ on Scalar Function Space `fs`
"""
function interpolate(f::Function, fs::ScalarFunctionSpace{dim,T}, mesh::PolygonalMesh) where {dim,T}
    n_cells = getncells(mesh)
    n_qpoints = getnquadpoints(fs)
    N = fill(zero(T)          * T(NaN), n_cells, n_qpoints)
    for (k,cell) in enumerate(get_cells(mesh))
        coords = get_coordinates(cell, mesh)
        for i in 1:n_qpoints
            N[k,i] = f(spatial_coordinate(fs, i, coords))
        end
    end
    InterpolatedFunction{dim,T}(N,fs)
end

# Trial Functions
struct TrialFunction{dim,T,refshape,N}
    fs::DiscreteFunctionSpace{dim,T,refshape}
    m_values::Array{T,N}
    f_node::Vector{Vec{dim,T}}
    components::Int
end

@inline getnbasefunctions(u::TrialFunction) = getnbasefunctions(u.fs)
@inline getfunctionspace(u::TrialFunction) = u.fs
@inline getnlocaldofs(u::TrialFunction) = getnlocaldofs(getfunctionspace(u))
@inline getrefshape(u::TrialFunction{dim,T,refshape}) where {dim,T,refshape} = refshape
@inline getncomponents(u::TrialFunction) = u.components

function TrialFunction(fs::ScalarTraceFunctionSpace{dim,T}, mesh::PolygonalMesh) where {dim,T}
    m_values = fill(zero(T) * T(NaN), getncells(mesh), getnbasefunctions(fs), get_maxnfaces(mesh))
    f_node = Vector{Vec{dim,T}}(getncells(mesh))
    for (k,cell) in enumerate(mesh.cells)
        f_node[k] = mesh.nodes[cell.nodes[1]].x
    end
    return TrialFunction(fs, m_values, f_node, 1)
end

function TrialFunction(fs::ScalarFunctionSpace{dim,T}, mesh::PolygonalMesh) where {dim,T}
    m_values = fill(zero(T) * T(NaN), getncells(mesh), getnbasefunctions(fs))
    f_node = Vector{Vec{dim,T}}(getncells(mesh))
    for (k,cell) in enumerate(mesh.cells)
        f_node[k] = mesh.nodes[cell.nodes[1]].x
    end
    return TrialFunction(fs, m_values, f_node, 1)
end

function TrialFunction(fs::VectorFunctionSpace{dim,T}, mesh::PolygonalMesh) where {dim,T}
    m_values = fill(zero(T) * T(NaN), getncells(mesh), getnbasefunctions(fs))
    f_node = Vector{Vec{dim,T}}(getncells(mesh))
    for (k,cell) in enumerate(mesh.cells)
        f_node[k] = mesh.nodes[cell.nodes[1]].x
    end
    return TrialFunction(fs, m_values, f_node, dim)
end

function TrialFunction(fs::DiscreteFunctionSpace{dim}, components::Int, m_values::Array{T,N}, mesh::PolygonalMesh) where {dim,T,N}
    @assert size(m_values,1) == getncells(mesh)
    @assert size(m_values,2) == getnbasefunctions(fs)
    f_node = Vector{Vec{dim,T}}(getncells(mesh))
    for (k,cell) in enumerate(mesh.cells)
        f_node[k] = mesh.nodes[cell.nodes[1]].x
    end
    return TrialFunction(fs, m_values, f_node, components)
end

"""
function value(u_h::TrialFunction, cell::Int, q_point::Int)
    get trial function value on cell `cell` at quadrature point
        `q_point`
"""
function value(u_h::TrialFunction{dim,T}, cell::Int, q_point::Int) where {dim,T}
    u = zero(T)
    for i in 1:getnbasefunctions(u_h.fs)
        u  += u_h.m_values[cell, i]*shape_value(u_h.fs, q_point, i)
    end
    return u
end

@inline reference_coordinate(fs::ScalarFunctionSpace{dim,T},cell::Int,
x_ref::Vec{dim,T}, x::Vec{dim,T}) where {dim,T} =
fs.Jinv[cell]⋅(x-x_ref)
"""
function value(u_h::TrialFunction{dim,T}, cell::Int, x::Vec{dim,T})
    get trial function value on cell `cell` at point `x`
"""
function value(u_h::TrialFunction{dim,T}, cell::Int, x::Vec{dim,T}) where {dim,T}
    u = zero(T)
    ξ = reference_coordinate(u_h.fs, cell, u_h.f_node[cell], x)
    for i in 1:getnbasefunctions(u_h.fs)
        u  += u_h.m_values[cell, i]*value(get_interpolation(u_h.fs), i, ξ)
    end
    return u
end

function errornorm(u_h::TrialFunction{dim,T}, u_ex::Function, mesh, norm_type::String="L2") where {dim,T}
    Etu_h = zero(T)
    if norm_type == "L2"
        n_basefuncs_s = getnbasefunctions(u_h)
        for (k,cell) in enumerate(get_cells(mesh))
            Elu_h = zero(T)
            coords = get_coordinates(cell, mesh)
            for q_point in 1:getnquadpoints(u_h.fs)
                dΩ = getdetJdV(u_h.fs, k, q_point)
                u = value(u_h, k, q_point)
                # Integral (u_h - u_ex) dΩ
                Elu_h += (u-u_ex(spatial_coordinate(u_h.fs, q_point, coords)))^2*dΩ
            end
            Etu_h += Elu_h
        end
    else
        throw("Norm $norm_type not available")
    end
    return Etu_h
end

function errornorm(u_h::InterpolatedFunction{dim,T}, u_ex::Function, mesh, norm_type::String="L2") where {dim,T}
    Etu_h = zero(T)
    if norm_type == "L2"
        n_basefuncs_s = getnbasefunctions(u_h.fs)
        for (k,cell) in enumerate(get_cells(mesh))
            Elu_h = zero(T)
            coords = get_coordinates(cell, mesh)
            for q_point in 1:getnquadpoints(u_h.fs)
                dΩ = getdetJdV(u_h.fs, k, q_point)
                u = value(u_h, k, q_point)
                # Integral (u_h - u_ex) dΩ
                Elu_h += (u-u_ex(spatial_coordinate(u_h.fs, q_point, coords)))^2*dΩ
            end
            Etu_h += Elu_h
        end
    else
        throw("Norm $norm_type not available")
    end
    return Etu_h
end
