module GRAM

hello() = "Welcome to the Julia implementation of [G]enerative [Ra]tio [M]atching networks!"

using Random, Statistics, Flux, MLToolkit.Plots, MLToolkit.Datasets, MLToolkit.Neural
using MLToolkit.Datasets: flatten
using MLToolkit.Neural: Neural, Trainable
using Flux: @functor, Optimise

### Neural sampler

using Distributions: Distributions
using Random: GLOBAL_RNG

struct NeuralSampler
    base
    f
end

@functor NeuralSampler

Distributions.rand(rng::AbstractRNG, g::NeuralSampler, n::Int) = g.f(rand(rng, g.base, n))
Distributions.rand(g::NeuralSampler, n::Int) = rand(GLOBAL_RNG, g, n)

export NeuralSampler

###

include("kmm.jl")   # MMD computation and density ratio estimation via KMM
include("gramnet.jl")
include("mmdgan.jl")
include("mmdnet.jl")
include("gan.jl")
export GRAMNet, MMDGAN, MMDNet, GAN

evaluate(g, ds) = evaluate(g, nothing, ds)
function evaluate(g, f, ds; rng=MersenneTwister(1))
    nd_half = div(ds.n_display, 2)

    # Generator
    x_data, x_gen = gpu(ds.Xt[:,1:nd_half]), rand(rng, g, nd_half)
    fig_g = ds.vis((data=flatten(cpu(x_data)), gen=flatten(cpu(x_gen))))
    
    # Projector
    if !isnothing(f)
        fx_data, fx_gen = f(x_data), f(x_gen)
        if size(fx_data, 1) in [2, 3]   # only visualise projector if its output dim is 2
            fig_f = plot(Scatter((data=cpu(fx_data), gen=cpu(fx_gen))))
            return (fig_gen=fig_g, fig_proj=fig_f)
        end
    end
    return (fig_gen=fig_g,)
end

export evaluate

export GRAM

end # module