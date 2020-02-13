using Statistics: mean, median
using DensityRatioEstimation: euclidsq, gaussian_gramian_by_euclidsq, _densratio_kmm

function prepare(x_de, x_nu, σs; verbose=false)
    d²_dede, d²_denu, d²_nunu = euclidsq(x_de), euclidsq(x_de, x_nu), euclidsq(x_nu)
    if isempty(σs)
        σ = sqrt(median(vcat(vec.([d²_dede, d²_denu, d²_nunu]))))
        #@info "Heuristic σ (the median of pairwise distances) is $σ."   # improve this log
        σs = (σ,)
    end
    return d²_dede, d²_denu, d²_nunu, σs
end

mmd²of(Kdede, Kdenu, Knunu) = mean(Kdede) - 2mean(Kdenu) + mean(Knunu)

const EPS_RATIO = 1f-3
const CLIP_LOWER, CLIP_UPPER = 0f0, 1f9
clip(x) = clamp.(x, CLIP_LOWER, CLIP_UPPER)

function compute_mmd(x_de, x_nu, σs)
    d²_dede, d²_denu, d²_nunu, σs = prepare(x_de, x_nu, σs)
    function f(σ)
        Kdede = gaussian_gramian_by_euclidsq(d²_dede, σ)
        Kdenu = gaussian_gramian_by_euclidsq(d²_denu, σ)
        Knunu = gaussian_gramian_by_euclidsq(d²_nunu, σ)
        return mmd²of(Kdede, Kdenu, Knunu)
    end
    mmd² = mapreduce(f, +, σs)
    return sqrt(clip(mmd²))
end

function estimate_ratio(x_de, x_nu, σs)
    d²_dede, d²_denu, d²_nunu, σs = prepare(x_de, x_nu, σs)
    function f(σ)
        Kdede = gaussian_gramian_by_euclidsq(d²_dede, σ)
        Kdenu = gaussian_gramian_by_euclidsq(d²_denu, σ)
        return _densratio_kmm(Kdede, Kdenu, EPS_RATIO)
    end
    ratio = mapreduce(f, +, σs)
    return clip(ratio) / length(σs)
end

function estimate_ratio_compute_mmd(x_de, x_nu, σs; isclip_ratio=true)
    d²_dede, d²_denu, d²_nunu, σs = prepare(x_de, x_nu, σs)
    function f(σ)
        Kdede = gaussian_gramian_by_euclidsq(d²_dede, σ)
        Kdenu = gaussian_gramian_by_euclidsq(d²_denu, σ)
        Knunu = gaussian_gramian_by_euclidsq(d²_nunu, σ)
        return _densratio_kmm(Kdede, Kdenu, EPS_RATIO), mmd²of(Kdede, Kdenu, Knunu)
    end
    ratio, mmd² = f(first(σs))
    for σ in Base.tail(σs)
        res1, res2 = f(σ)
        ratio += res1
        mmd² += res2
    end
    return (ratio=((isclip_ratio ? clip(ratio) : ratio) / length(σs)), mmd=sqrt(clip(mmd²)))
end
