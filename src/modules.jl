using MLToolkit.Neural: IntIte, optional_BatchNorm

# Flux.trainable(c::Conv) = (c.weight,)
# Flux.trainable(ct::ConvTranspose) = (ct.weight,)

function build_conv_outmnist(Din::Int, σs; isnorm::Bool=false)
    @assert length(σs) == 4 "Length of `σs` must be `4` for `build_conv_outmnist`"
    return Chain(
        #     Din x B
        Dense(Din,  1024), optional_BatchNorm(1024, σs[1], isnorm),
        #    1024 x B
        Dense(1024, 6272),
        # -> 6272 x B
        x -> reshape(x, 7, 7, 128, last(size(x))), optional_BatchNorm(128, σs[2], isnorm),
        # ->  7 x  7 x 128 x B
        ConvTranspose((4, 4), 128 => 64; stride=(2, 2), pad=(1, 1)), optional_BatchNorm(64, σs[3], isnorm),
        # -> 14 x 14 x  64 x B
        ConvTranspose((4, 4),  64 =>  1; stride=(2, 2), pad=(1, 1)), x -> σs[4].(x)
        # -> 28 x 28 x 1 x B
    )
end

build_conv_outmnist(Din::Int, σ::Function, σlast::Function; kwargs...) = 
    build_conv_outmnist(Din, (σ, σ, σ, σlast); kwargs...)

function build_convnet_outcifar(Din::Int, σs; isnorm::Bool=false)
    @assert length(σs) == 4 "Length of `σs` must be `4` for `build_convnet_outcifar10`"
    return Chain(
        #     Din x B
        Dense(Din,  2048), 
        # -> 2048 x B
        x -> reshape(x, 4, 4, 128, size(x, 2)), 
        optional_BatchNorm(128, σs[1], isnorm; momentum=9f-1),
        # ->  4 x  4 x 128 x B
        ConvTranspose((4, 4), 128 => 64; stride=(2, 2), pad=(1, 1)), 
        optional_BatchNorm(64, σs[2], isnorm; momentum=9f-1),
        # ->  8 x  8 x  64 x B        
        ConvTranspose((4, 4),  64 => 32; stride=(2, 2), pad=(1, 1)), 
        optional_BatchNorm(32, σs[3], isnorm; momentum=9f-1),
        # -> 16 x 16 x  64 x B
        ConvTranspose((4, 4),  32 =>  3; stride=(2, 2), pad=(1, 1)), x -> σs[4].(x)
        # -> 32 x 32 x   3 x B
    )
end

build_convnet_outcifar(Din::Int, σ::Function, σlast::Function; kwargs...) = 
    build_convnet_outcifar(Din, (σ, σ, σ, σlast); kwargs...)

# TODO: unify building conv archs
function NeuralSampler(
    base, 
    Dz::Int, 
    Dhs::Union{IntIte,String}, 
    Dx::Union{Int,Tuple{Int,Int,Int}}, 
    σ::Function, 
    σlast::Function, 
    isnorm::Bool, 
    n_default::Int
)
    size(base) != (Dz,) && throw(DimensionMismatch("size(base) ($(size(base))) != (Dz,) ($((Dz,)))"))
    if Dx == (32, 32, 3) && Dhs == "conv"
        f = build_convnet_outcifar(Dz, σ, σlast; isnorm=isnorm)
    elseif Dx == (28, 28, 1) && Dhs == "conv"
        f = build_conv_outmnist(Dz, σ, σlast; isnorm=isnorm)
    else
        f = DenseNet(Dz, Dhs, Dx, σ, σlast; isnorm=isnorm)
    end
    return NeuralSampler(base, f, n_default)
end

### Projector

function DenseProjector(Dx::Int, Dhs::IntIte, Df::Int, σ::Function, isnorm::Bool)
    return Projector(DenseNet(Dx, Dhs, Df, σ, identity; isnorm=isnorm), Df)
end

function ConvProjector(Dx::Union{Int,Tuple{Int,Int,Int}}, Df::Int, σ::Function, isnorm::Bool)
    if Dx == 28 * 28 || Dx == (28, 28, 1)
        return Projector(ConvNet((28, 28, 1), Df, σ, identity; isnorm=isnorm), Df)
    elseif Dx == (32, 32, 3)
        return Projector(ConvNet((32, 32, 3), Df, σ, identity; isnorm=isnorm), Df)
    else
        @error "[ConvProjector] Only MNIST-like or CIFAR-like data is supported."
    end
end

### Discriminator

function DenseDiscriminator(Dx::Int, Dhs::IntIte, σ::Function, isnorm::Bool)
    return Discriminator(DenseNet(Dx, Dhs, 1, σ, sigmoid; isnorm=isnorm))
end

function ConvDiscriminator(Dx::Union{Int,Tuple{Int,Int,Int}}, σ::Function, isnorm::Bool)
    if Dx == 28 * 28
        return Discriminator(ConvNet((28, 28, 1), 1, σ, sigmoid; isnorm=isnorm))
    elseif Dx == 32 * 32 * 3
        return Discriminator(ConvNet((32, 32, 3), 1, σ, sigmoid; isnorm=isnorm))
    else
        @error "[ConvDiscriminator] Only MNIST-like or CIFAR-like data is supported."
    end
end
