"""
get_affine_map(x, ξ)
Get (A,b) such that ξ = A * x + b is the affine
mapping from the simplex with vertices x to the simplex of vertices ξ.
"""
function get_affine_map(x, ξ)
    @assert length(x[1]) == length(ξ[1]) "dimension mismatch"
    n = length(x[1])
    T = eltype(x[1])

    B = zeros(T, n*(n+1), n*(n+1))
    L = zeros(T, n*(n+1))

    for i in 1:length(x)
        for j in 1:n
            row = (i-1) * n + j
            B[row, n*(j-1)+1:n*j] = x[i]
            L[row] = ξ[i][j]
            B[row, n * n + j] = one(T)
        end
    end
    X = B\L
    return reshape(X[1:n*n], (n, n)), X[n*n+1:end]
end

"""
function integrate(f::function,qr::QuadratureRule)
integrate function f on reference shape of quadrature qr
"""
function integrate(f::Function,qr::QuadratureRule)
    p = getpoints(qr); w = getweights(qr)
    int_val = zero(typeof(f(p[1])))
    for (i,x) in enumerate(p)
        int_val += f(x)*w[i]
    end
    int_val
end
