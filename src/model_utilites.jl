using Zygote: Zygote, @nograd
using DensityRatioEstimation: euclidsq, gaussian_gramian_by_euclidsq, KMM, _densratio_kmm

heuristic_σ(args...) = sqrt(median(vcat(vec.([args...]))))

function multi_run(f_run, x_de, x_nu, σs, verbose)
    dsq_dede, dsq_denu, dsq_nunu = euclidsq(x_de), euclidsq(x_de, x_nu), euclidsq(x_nu)
    
    if isempty(σs)
        σ = heuristic_σ(dsq_dede, dsq_denu, dsq_nunu)
        verbose && multi_run_info(σ)
        σs = [σ]
    end
    
    local res = nothing
    for σ in σs
        res_current = f_run(dsq_dede, dsq_denu, dsq_nunu, σ)
        if isnothing(res)
            res = res_current
        else
            res = res .+ res_current
        end
    end
    
    return res
end

multi_run_info(σ) = @info "Heuristic σ (the median of pairwise distances) is $σ."
@nograd multi_run_info

_mmd_sq(Kdede, Kdenu, Knunu) = mean(Kdede) - 2mean(Kdenu) + mean(Knunu)

const EPS_MMD = 1f-6

function compute_mmd(x_de, x_nu; σs=[], verbose=false, is_clip=false)
    function f(dsq_dede, dsq_denu, dsq_nunu, σ)
        Kdede = gaussian_gramian_by_euclidsq(dsq_dede, σ)
        Kdenu = gaussian_gramian_by_euclidsq(dsq_denu, σ)
        Knunu = gaussian_gramian_by_euclidsq(dsq_nunu, σ)
        return _mmd_sq(Kdede, Kdenu, Knunu)
    end
    mmd_sq = multi_run(f, x_de, x_nu, σs, verbose)
    if is_clip
        return sqrt(relu(mmd_sq))
    else
        return sqrt(mmd_sq + EPS_MMD)
    end
end

const EPS_RATIO = 1f-3

function estimate_ratio(x_de, x_nu; σs=[], verbose=false)
    function f(dsq_dede, dsq_denu, dsq_nunu, σ)
        Kdede = gaussian_gramian_by_euclidsq(dsq_dede, σ)
        Kdenu = gaussian_gramian_by_euclidsq(dsq_denu, σ)
        return _densratio_kmm(Kdede, Kdenu, EPS_RATIO)
    end
    return multi_run(f, x_de, x_nu, σs, verbose) / convert(Float32, length(σs))
end

function estimate_ratio_compute_mmd(x_de, x_nu; σs=[], verbose=false)
    function f(dsq_dede, dsq_denu, dsq_nunu, σ)
        Kdede = gaussian_gramian_by_euclidsq(dsq_dede, σ)
        Kdenu = gaussian_gramian_by_euclidsq(dsq_denu, σ)
        Knunu = gaussian_gramian_by_euclidsq(dsq_nunu, σ)
        return (_densratio_kmm(Kdede, Kdenu, EPS_RATIO), _mmd_sq(Kdede, Kdenu, Knunu))
    end
    ratio, mmd_sq = multi_run(f, x_de, x_nu, σs, verbose)
    return (
        ratio=(ratio / convert(Float32, length(σs))), 
        mmd=sqrt(mmd_sq + EPS_MMD)
    )
end