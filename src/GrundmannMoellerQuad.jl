for dim in (2,3)
    @eval begin
        function QuadratureRule(quad_type::GrundmannMoeller, s::Int) where {$dim,RefTetrahedron}
            points, weigths = _compute_p_w(s, $dim)
            return QuadratureRule{$dim,RefTetrahedron,Float64}(weigths, points)
        end
    end
end

function _compute_p_w(s, n, T = Float64)
    d = 2 * s + 1
    k = 0
    nnodes = binomial(n + s + 1, s)
    weights = Vector{T}(undef, nnodes)
    kk = nnodes
    points = Vector{Vec{n,T}}(undef, nnodes)
    exponentials = _get_all_exponentials(s, n+1)
    for i = 0:s
        w::T = ((-1)^i * 2.0^(-2*s) * (d+n-2*i)^d)/(factorial(i) * factorial(d+n-i))
        for p in _get_all_exponentials(s-i, n+1)
            k += 1
            points[k] = Vec{n}(j -> (2*p[j+1] + 1)/(d+n-2*i))
            weights[k] = w
        end
    end
    return points, weights./sum(2*weights)
end

function _get_all_exponentials(n,k)
    a = Vector{Int32}(undef, k)
    exponentials = Vector{typeof(a)}(undef,binomial(n+2,k-1))
    t::Int32 = n
    h = 0
    a[1] = n
    a[2:k] .= 0
    exponentials[1] = copy(a)
    idx = 2
    while a[k] != n
        if ( 1 < t ); h = 0; end
        h = h + 1
        t = a[h]
        a[h] = 0
        a[1] = t - 1
        a[h+1] = a[h+1] + 1
        exponentials[idx] = copy(a)
        idx +=1
    end
    return exponentials
end
