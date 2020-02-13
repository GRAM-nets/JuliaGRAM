module GRAM

hello() = "Welcome to the Julia implementation of [G]enerative [Ra]tio [M]atching networks!"

using Random, Statistics, Flux, MLToolkit.Plots, MLToolkit.Datasets, MLToolkit.Neural
using MLToolkit.Datasets: flatten
using MLToolkit.Neural: Neural, Trainable
using Flux: @functor, Optimise

### Modules

using Distributions: Distributions
using Random: GLOBAL_RNG

# Neural sampler

struct NeuralSampler
    base
    f
end

@functor NeuralSampler

Distributions.rand(rng::AbstractRNG, g::NeuralSampler, n::Int) = g.f(rand(rng, g.base, n))
Distributions.rand(g::NeuralSampler, n::Int) = rand(GLOBAL_RNG, g, n)

# Projector

struct Projector
    f
end

@functor Projector

(p::Projector)(x) = p.f(x)

# Conv out for CIFAR10

using MLToolkit.Neural: optional_BatchNorm

function build_convnet_outcifar(Din::Int, σs; isnorm::Bool=false)
    @assert length(σs) == 4 "Length of `σs` must be `4` for `build_convnet_outcifar10`"
    return Chain(
        #     Din x B
        Dense(Din,  2048), 
        # -> 2048 x B
        x -> reshape(x, 4, 4, 128, size(x, 2)), 
        optional_BatchNorm(128, σs[1], isnorm; momentum=9f-1),
        # ->    4 x  4 x 128 x B
        ConvTranspose((4, 4), 128 => 64; stride=(2, 2), pad=(1, 1)), 
        optional_BatchNorm(64, σs[2], isnorm; momentum=9f-1),
        # ->    8 x  8 x  64 x B        
        ConvTranspose((4, 4),  64 => 32; stride=(2, 2), pad=(1, 1)), 
        optional_BatchNorm(32, σs[3], isnorm; momentum=9f-1),
        # ->   16 x 16 x  64 x B
        ConvTranspose((4, 4),  32 =>  3; stride=(2, 2), pad=(1, 1)), x -> σs[4].(x)
        # ->   32 x 32 x   3 x B
    )
end

build_convnet_outcifar(Din::Int, σ::Function, σlast::Function; kwargs...) = 
    build_convnet_outcifar(Din, (σ, σ, σ, σlast); kwargs...)

export NeuralSampler, Projector, build_convnet_outcifar

###

include("mmd_utilities.jl")

include("gramnet.jl")
export GRAMNet
include("mmdgan.jl")
export MMDGAN
include("mmdnet.jl")
export MMDNet
include("gan.jl")
export GAN

###

function evaluate(g, ds)
    rng = MersenneTwister(1)
    nd_half = div(ds.n_display, 2)
    x_data, x_gen = gpu(ds.Xt[:,1:nd_half]), rand(rng, g, nd_half)
    fig_g = ds.vis((data=flatten(cpu(x_data)), gen=flatten(cpu(x_gen))))
    return (fig_gen=fig_g,)
end

function evaluate(g, f::Projector, ds)
    rng = MersenneTwister(1)
    nd_half = div(ds.n_display, 2)
    # Generator
    x_data, x_gen = gpu(ds.Xt[:,1:nd_half]), rand(rng, g, nd_half)
    fig_g = ds.vis((data=flatten(cpu(x_data)), gen=flatten(cpu(x_gen))))
    # Projector
    fx_data, fx_gen = f(x_data), f(x_gen)
    if size(fx_data, 1) in [2, 3]   # only visualise projector if its output dim is 2
        fig_f = plot(Scatter((data=cpu(fx_data), gen=cpu(fx_gen))))
        return (fig_gen=fig_g, fig_proj=fig_f)
    else
        return (fig_gen=fig_g,)
    end
end

export evaluate

export GRAM

end # module