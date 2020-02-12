using Statistics: mean, median
using DensityRatioEstimation: euclidsq, gaussian_gramian_by_euclidsq, _densratio_kmm

function prepare(x_de, x_nu, σs; verbose=false)
    esq_dede, esq_denu, esq_nunu = euclidsq(x_de), euclidsq(x_de, x_nu), euclidsq(x_nu)
    if isempty(σs)
        σ = sqrt(median(vcat(vec.([esq_dede, esq_denu, esq_nunu]))))
        #@info "Heuristic σ (the median of pairwise distances) is $σ."   # improve this log
        σs = (σ,)
    end
    return esq_dede, esq_denu, esq_nunu, σs
end

_mmd_sq(Kdede, Kdenu, Knunu) = mean(Kdede) - 2mean(Kdenu) + mean(Knunu)

const EPS_RATIO = 1f-3
const CLIP_LOWER, CLIP_UPPER = 0f0, 1f9
clip(x::AbstractFloat) = clamp(x, CLIP_LOWER, CLIP_UPPER)
clip(x::AbstractArray) = clamp.(x, CLIP_LOWER, CLIP_UPPER)

function compute_mmd(x_de, x_nu, σs)
    esq_dede, esq_denu, esq_nunu, σs = prepare(x_de, x_nu, σs)
    function f(σ)
        Kdede = gaussian_gramian_by_euclidsq(esq_dede, σ)
        Kdenu = gaussian_gramian_by_euclidsq(esq_denu, σ)
        Knunu = gaussian_gramian_by_euclidsq(esq_nunu, σ)
        return _mmd_sq(Kdede, Kdenu, Knunu)
    end
    mmd_sq = mapreduce(f, +, σs)
    return sqrt(clip(mmd_sq))
end

function estimate_ratio(x_de, x_nu, σs)
    esq_dede, esq_denu, esq_nunu, σs = prepare(x_de, x_nu, σs)
    function f(σ)
        Kdede = gaussian_gramian_by_euclidsq(esq_dede, σ)
        Kdenu = gaussian_gramian_by_euclidsq(esq_denu, σ)
        return _densratio_kmm(Kdede, Kdenu, EPS_RATIO)
    end
    ratio = mapreduce(f, +, σs)
    return clip(ratio) / length(σs)
end

function estimate_ratio_compute_mmd(x_de, x_nu, σs)
    esq_dede, esq_denu, esq_nunu, σs = prepare(x_de, x_nu, σs)
    function f(σ)
        Kdede = gaussian_gramian_by_euclidsq(esq_dede, σ)
        Kdenu = gaussian_gramian_by_euclidsq(esq_denu, σ)
        Knunu = gaussian_gramian_by_euclidsq(esq_nunu, σ)
        return _densratio_kmm(Kdede, Kdenu, EPS_RATIO), _mmd_sq(Kdede, Kdenu, Knunu)
    end
    ratio, mmd_sq = f(first(σs))
    for σ in Base.tail(σs)
        res1, res2 = f(σ)
        ratio += res1
        mmd_sq += res2
    end
    return (ratio=(clip(ratio) / length(σs)), mmd=sqrt(clip(mmd_sq)))
end
