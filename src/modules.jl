using Flux: Flux, gpu, cpu, @functor
using MLToolkit.Neural: optional_BatchNorm
using Distributions: Distributions
using Random: GLOBAL_RNG

### Neural sampler

struct NeuralSampler
    base
    f
end

@functor NeuralSampler

Distributions.rand(rng::AbstractRNG, g::NeuralSampler, n::Int) = g.f(rand(rng, g.base, n))
Distributions.rand(g::NeuralSampler, n::Int) = rand(GLOBAL_RNG, g, n)

### Projector

struct Projector
    f
end

@functor Projector

(p::Projector)(x) = p.f(x)

###

# Flux.trainable(c::Conv) = (c.weight,)
# Flux.trainable(ct::ConvTranspose) = (ct.weight,)

function build_convnet_outmnist(Din::Int, σs; isnorm::Bool=false)
    @assert length(σs) == 4 "Length of `σs` must be `4` for `build_convnet_outmnist`"
    return Chain(
        #     Din x B
        Dense(Din,  1024), optional_BatchNorm(1024, σs[1], isnorm),
        #    1024 x B
        Dense(1024, 6272),
        # -> 6272 x B
        x -> reshape(x, 7, 7, 128, last(size(x))), optional_BatchNorm(128, σs[2], isnorm),
        # ->    7 x  7 x 128 x B
        ConvTranspose((4, 4), 128 => 64; stride=(2, 2), pad=(1, 1)), optional_BatchNorm(64, σs[3], isnorm),
        # ->   14 x 14 x  64 x B
        ConvTranspose((4, 4),  64 =>  1; stride=(2, 2), pad=(1, 1)), x -> σs[4].(x)
        # ->   28 x 28 x 1 x B
    )
end

build_convnet_outmnist(Din::Int, σ::Function, σlast::Function; kwargs...) = 
    build_convnet_outmnist(Din, (σ, σ, σ, σlast); kwargs...)

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
