module GRAM

using Statistics, LinearAlgebra, StatsFuns, Distributions, Humanize, Dates
using MLDatasets: CIFAR10, MNIST
using Random: MersenneTwister
using Reexport: @reexport

@reexport using MLToolkit
import MLToolkit.Neural: evaluate, update!
import MLToolkit: parse_toml, process_argdict, NeuralSampler, plot!

### Scripting

function parse_toml(hyperpath::String, dataset::String, modelname::String)
    return parse_toml(hyperpath, (:dataset => dataset, :modelname => modelname))
end

function process_argdict(argdict; override=NamedTuple(), suffix="")
    return process_argdict(
        argdict; 
        override=override,
        nameexclude=[:dataset, :modelname, :n_epochs, :actlast],
        nameinclude_last=:seed,
        suffix=suffix
    )
end

export parse_toml, process_argdict

###

include("modules.jl")
export NeuralSampler, DenseProjector, ConvProjector, DenseDiscriminator, ConvDiscriminator
include("models.jl")
export GAN, MMDNet, RMMMDNet, train!, evaluate

###

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


parse_csv(T, l) = map(x -> parse(T, x), split(l, ","))
parse_op(op::String) = eval(Symbol(op))
parse_op(op) = op

function get_model(args::NamedTuple, dataset::Dataset)
    module_path = pathof(@__MODULE__) |> splitdir |> first |> splitdir |> first
    logdir = "$(dataset.name)/$(args.modelname)/$(args.expname)/$(Dates.format(now(), DATETIME_FMT))"
    logger = TBLogger("$module_path/logs/$logdir")
    if args.opt == "adam"
        opt = ADAM(args.lr, (args.beta1, 999f-4))
    elseif args.opt == "rmsprop"
        opt = RMSProp(args.lr)
    end
    if args.base == "uniform"
        base = UniformNoise(args.Dz)
    elseif args.base == "gaussian"
        base = GaussianNoise(args.Dz)
    end
    if args.Dhs_g == "conv"
        if :act in keys(args)
            @warn "args.act is ignored"
        end
        if :actlast in keys(args)
            @warn "args.actlast is ignored"
        end
        actlast = dataset.name == "cifar10" ? tanh : sigmoid
        g = NeuralSampler(base, args.Dz, "conv", dim(dataset), relu, actlast, args.norm, args.batchsize_g)
    else
        Dhs_g = parse_csv(Int, args.Dhs_g)
        act = parse_op(args.act)
        actlast = parse_op(args.actlast)
        g = NeuralSampler(base, args.Dz, Dhs_g, dim(dataset), act, actlast, args.norm, args.batchsize_g)
    end
    if args.modelname == "gan"
        if args.Dhs_d == "conv"
            d = ConvDiscriminator(dim(dataset), act, args.norm)
        else
            Dhs_d = parse_csv(Int, args.Dhs_d)
            d = DenseDiscriminator(dim(dataset), Dhs_d, act, args.norm)
        end
        m = GAN(logger, opt, g, d)
    else
        sigma = args.sigma == "median" ? [] : parse_csv(Float32, args.sigma)
        if args.modelname == "mmdnet"
            m = MMDNet(logger, opt, sigma, g)
        elseif args.modelname == "mmdgan"
            @assert args.Dhs_f != "conv"
            Dhs_f = parse_csv(Int, args.Dhs_f)
            Dx = first(dim(dataset))
            fenc = DenseProjector(Dx, Dhs_f, args.Df, act, args.norm)
            fdec = DenseProjector(args.Df, Dhs_f, Dx, act, args.norm)
            m = MMDGAN(logger, opt, sigma, g, fenc, fdec)
        elseif args.modelname == "rmmmdnet"
            if args.Dhs_f == "conv"
                act = x -> leakyrelu(x, 2f-1)
                if :act in keys(args)
                    @warn "args.act is ignored"
                end
                f = ConvProjector(dim(dataset), args.Df, act, args.norm)
            else
                Dhs_f = parse_csv(Int, args.Dhs_f)
                f = DenseProjector(dim(dataset), Dhs_f, args.Df, act, args.norm)
            end
            m = RMMMDNet(logger, opt, sigma, g, f)
        end
    end
    @info "Init $(args.modelname) with $(nparams(m) |> Humanize.digitsep) parameters" logdir
    return m |> gpu
end

function run_exp(args; model=nothing, initonly=false)
    seed!(args.seed)
    
    data = get_data(args.dataset)
    dataloader = DataLoader(data, args.batchsize)

    model = isnothing(model) ? get_model(args, data) : model
    
    if !initonly
        with_logger(model.logger) do
            train!(model, dataloader, args.n_epochs; evalevery=50)
        end
    end
    
    evaluate(model, dataloader)
    
    modelpath = savemodel(model)
    
    return dataloader, model, modelpath
end

export get_data, get_model, run_exp

end # module
