for dim in (2,3)
    @eval begin
        function (::Type{QuadratureRule{$dim,RefSimplex,GrundmannMoeller}})(s::Int)
            p, weigths = _compute_p_w(s, $dim)
            points = Vector{SVector{$dim,Float64}}(size(p,1))
            for i in 1:size(p,1)
                points[i] = SVector{$dim}(p[i,:])
            end
            return QuadratureRule{$dim,RefSimplex,Float64}(weigths, points)
        end
    end
end

function _compute_p_w(s, n, T = Float64)
    d = 2 * s + 1
    k = 0
    nnodes = binomial(n + s + 1, s)
    weights = Vector{T}(nnodes)
    kk = nnodes
    points = Matrix{Rational{Int64}}(nnodes, n)
    exponentials = _get_all_exponentials(s, n+1)
    for i = 0:s
        w = ((-1)^i * 2.0^(-2*s) * (d+n-2*i)^d)/(factorial(i) * factorial(d+n-i))
        for p in _get_all_exponentials(s-i, n+1)
            k += 1
            for j in 2:(n+1)
                points[k,j-1] = (2*p[j] + 1)//(d+n-2*i)
            end
            weights[k] = w
        end
    end
    return points, weights./sum(weights)
end

function _get_all_exponentials(n,k)
    a = Vector{Int32}(k)
    exponentials = Vector{typeof(a)}(binomial(n+2,k-1))
    t = n
    h = 0
    a[1] = n
    a[2:k] = 0
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
