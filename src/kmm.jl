using LinearAlgebra: I, diagm
using Statistics: mean, median
using Flux: CuArrays, Zygote

safe_diagm(mat, a) = a * I
safe_diagm(mat::CuArrays.CuArray, a::T) where {T} = CuArrays.CuArray{T}(diagm(0 => fill(a, size(mat, 1))))

Zygote.@nograd safe_diagm

function euclidsq(X, Y)
    XiXj = transpose(X) * Y
    x² = sum(X.^2; dims=1)
    y² = sum(Y.^2; dims=1)
    return transpose(x²) .+ y² - 2XiXj
end

function euclidsq(X)
    XiXj = transpose(X) * X
    x² = sum(X.^2; dims=1)
    return transpose(x²) .+ x² - 2XiXj
end

function get_heuristic_σ(d²_dede, d²_denu, d²_nunu)
    # A heuristic is to use the median of pairwise distances as σ, suggested by Sugiyama's book
    σ = sqrt(median(vcat(vec.([d²_dede, d²_denu, d²_nunu]))))
    @info "train" heuristic_σ=σ
    return σ
end

Zygote.@nograd get_heuristic_σ

gaussian_gramian(esq, σ) = exp.(-esq ./ 2σ^2)

function ratioof(Kdede, Kdenu, λ::T) where {T<:AbstractFloat}
    !iszero(λ) && (Kdede += safe_diagm(Kdede, λ))
    n_de, n_nu = size(Kdenu)
    return T(n_de / n_nu) * (Kdede \ vec(sum(Kdenu, dims=2)))
end

mmd²of(Kdede, Kdenu, Knunu) = mean(Kdede) - 2mean(Kdenu) + mean(Knunu)

### Helpers for efficient computation for a list of σ and post-processing

const EPS_RATIO = 1f-3
const CLIP_LOWER, CLIP_UPPER = 0f0, 1f9

clip(x, isclip=true) = isclip ? clamp.(x, CLIP_LOWER, CLIP_UPPER) : x

function prepare2(x_de, x_nu, σs; verbose=false)
    d²_dede, d²_denu = euclidsq(x_de), euclidsq(x_de, x_nu)
    isempty(σs) && (σs = (get_heuristic_σ(d²_dede, d²_denu),))
    return d²_dede, d²_denu, σs
end

function prepare3(x_de, x_nu, σs; verbose=false)
    d²_dede, d²_denu, d²_nunu = euclidsq(x_de), euclidsq(x_de, x_nu), euclidsq(x_nu)
    isempty(σs) && (σs = (get_heuristic_σ(d²_dede, d²_denu, d²_nunu),))
    return d²_dede, d²_denu, d²_nunu, σs
end

function compute_mmd(x_de, x_nu, σs)
    d²_dede, d²_denu, d²_nunu, σs = prepare3(x_de, x_nu, σs)
    function f(σ)
        Kdede, Kdenu, Knunu = gaussian_gramian.((d²_dede, d²_denu, d²_nunu), σ)
        return mmd²of(Kdede, Kdenu, Knunu)
    end
    mmd² = mapreduce(f, +, σs)
    return sqrt(clip(mmd²))
end

function estimate_ratio(x_de, x_nu, σs)
    d²_dede, d²_denu, σs = prepare2(x_de, x_nu, σs)
    function f(σ)
        Kdede, Kdenu = gaussian_gramian.((d²_dede, d²_denu), σ)
        return ratioof(Kdede, Kdenu, EPS_RATIO)
    end
    ratio = mapreduce(f, +, σs)
    return clip(ratio) / length(σs)
end

function estimate_ratio_compute_mmd(x_de, x_nu, σs; isclip_ratio=true)
    d²_dede, d²_denu, d²_nunu, σs = prepare3(x_de, x_nu, σs)
    function f(σ)
        Kdede, Kdenu, Knunu = gaussian_gramian.((d²_dede, d²_denu, d²_nunu), σ)
        return ratioof(Kdede, Kdenu, EPS_RATIO), mmd²of(Kdede, Kdenu, Knunu)
    end
    ratio, mmd² = f(first(σs))
    for σ in Base.tail(σs)
        res1, res2 = f(σ)
        ratio += res1
        mmd² += res2
    end
    return (ratio=(clip(ratio, isclip_ratio) / length(σs)), mmd=sqrt(clip(mmd²)))
end
