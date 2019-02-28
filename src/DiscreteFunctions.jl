"""
function function_value(f::Function, fs::ScalarFunctionSpace{dim,T}, cell::Int,q_point::Int) where {dim,T}
Evaluate function f(x): x ∈ ℝⁿ ↦ ℝ with x given by cell number `cell` and quadrature point with index `qpoint`
    using fs FunctionSpace data
"""
function function_value(f::Function, fs::ScalarFunctionSpace{dim,T}, cell::Int,q_point::Int) where {dim,T}
    coords = get_cell_coordinates(cell, fs.mesh)
    return f(spatial_coordinate(fs, q_point, coords))
end

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

# Trial Functions
struct TrialFunction{dim,T,shape,N1}
    fs::DiscreteFunctionSpace
    m_values::Array{T,N1}
    components::Int
end

@inline getnbasefunctions(u::TrialFunction) = getnbasefunctions(u.fs)
@inline getfunctionspace(u::TrialFunction) = u.fs
@inline getnlocaldofs(u::TrialFunction) = getnlocaldofs(getfunctionspace(u))
@inline getncomponents(u::TrialFunction) = u.components

function TrialFunction(fs::ScalarTraceFunctionSpace{dim,T}) where {dim,T}
    mesh = getmesh(fs)
    m_values = fill(zero(T) * T(NaN), getncells(mesh), getnbasefunctions(fs), n_faces_per_cell(mesh))
    return TrialFunction{dim,T,getshape(getfiniteelement(fs)),3}(fs, m_values, 1)
end

function TrialFunction(fs::ScalarFunctionSpace{dim,T}) where {dim,T}
    mesh = getmesh(fs)
    m_values = fill(zero(T) * T(NaN), getncells(mesh), getnbasefunctions(fs))
    return TrialFunction{dim,T,getshape(getfiniteelement(fs)),2}(fs, m_values, 1)
end

function TrialFunction(fs::VectorFunctionSpace{dim,T}) where {dim,T}
    mesh = getmesh(fs)
    m_values = fill(zero(T) * T(NaN), getncells(mesh), getnbasefunctions(fs))
    return TrialFunction{dim,T,getshape(getfiniteelement(fs)),2}(fs, m_values, dim)
end

function TrialFunction(fs::DiscreteFunctionSpace{dim}, components::Int, m_values::Array{T,N}) where {dim,T,N}
    mesh = getmesh(fs)
    @assert size(m_values,1) == getncells(mesh)
    @assert size(m_values,2) == getnbasefunctions(fs)
    return TrialFunction{dim,T,getshape(getfiniteelement(fs)),N}(fs, m_values, components)
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
    mesh = getmesh(u_h.fs)
    ξ = reference_coordinate(u_h.fs, cell, mesh.nodes[mesh.cells[cell].nodes[1]].x, x)
    for i in 1:getnbasefunctions(u_h.fs)
        u  += u_h.m_values[cell, i]*value(getfiniteelement(u_h.fs), i, ξ)
    end
    return u
end

function errornorm(u_h::TrialFunction{dim,T}, u_ex::Function, norm_type::String="L2") where {dim,T}
    mesh = getmesh(u_h.fs)
    Etu_h = zero(T)
    if norm_type == "L2"
        n_basefuncs_s = getnbasefunctions(u_h)
        for (k,cell) in enumerate(get_cells(mesh))
            Elu_h = zero(T)
            coords = get_coordinates(cell, mesh)
            reinit!(u_h.fs, coords)
            for q_point in 1:getnquadpoints(u_h.fs)
                dΩ = getdetJdV(u_h.fs, q_point)
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
