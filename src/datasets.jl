function get_dataset(name; kwargs...)
    args = (60_000,)
    if name == "ring"
        n_mixtures, distance, var = 8, 2f0, 2f-1
        @assert args == ()
        args = (args..., n_mixtures, distance, var)
    end
    return Dataset(name, args...; kwargs...)
end

function plot!(d::Dataset, g::NeuralSampler, f::Projector)
    rng = MersenneTwister(1)
    Xdata = d.train
    Xgen = rand(rng, g, last(size(Xdata)))
    fXdata = f(Xdata) |> cpu
    fXgen = f(Xgen) |> cpu
    plt.scatter(fXdata[1,:], fXdata[2,:], marker=".", label="data", alpha=0.5)
    plt.scatter(fXgen[1,:],  fXgen[2,:],  marker=".", label="gen",  alpha=0.5)
    autoset_lim!(fXdata)
    plt.legend(fancybox=true, framealpha=0.5)
end
