module Models

using Random, Flux, MLToolkit.Plots, MLToolkit.Datasets, MLToolkit.Neural
using MLToolkit.Neural: Neural, Trainable

include("modules.jl")
export NeuralSampler, Projector, build_convnet_outmnist, build_convnet_outcifar

###

include("model_utilites.jl")

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
    fig_g = ds.vis((data=ds.Xt[:,nd_half], gen=rand(rng, g, nd_half)))
    return (gen=fig_g,)
end

# Only visualise projector if its output dim is 2
function evaluate(g, f::Projector, ds)
    rng = MersenneTwister(1)
    nd_half = div(ds.n_display, 2)
    # Generator
    x, x_gen = ds.Xt[:,nd_half], rand(rng, g, nd_half)
    fig_g = ds.vis((data=x, gen=x_gen))
    # Projector
    fx, fx_gen = f(x), f(x_gen)
    if size(fx, 1) in [2, 3]
        fig_f = plot(Scatter((data=fx, gen=fx_gen)))
        return (gen=fig_g, proj=fig_f)
    else
        return (gen=fig_g,)
    end
end

export evaluate

end # module
