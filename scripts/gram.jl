using DrWatson
@quickactivate "GRAM"
using MLToolkit.Scripting

###

if length(ARGS) == 1
    argdict_master = load(projectdir("_research", "tmp", first(ARGS)))
else
    argdict_master = Dict(
        :dataset => "mnist",
        :model   => "gramnet",
    )
end

@assert :dataset in keys(argdict_master)
@assert :model in keys(argdict_master)

args_ignored = (
    n_epochs    = 35,
    evalevery   = 100,
    is_continue = false,
    nowandb     = false,
    notes       = "",
    nosave      = true,
)

args = (args_ignored...,
    seed        = 1,
    dataset     = argdict_master[:dataset],
    model       = argdict_master[:model],
    batch_size  = 200,
)

@assert args.dataset in ["gaussian", "2dring", "3dring", "mnist", "cifar10"]
@assert args.model in ["gramnet", "mmdgan", "mmdnet", "gan"]

# Load pre-defined arguments
include(scriptsdir("predefined_args.jl"))
args = concat_predefined_args(args)

# Add keyword arguments
kwargs = args.model == "gramnet" ? (isclip_ratio=true,) : NamedTuple()
length(kwargs) > 0 && (args = union(args, kwargs))

# Overwrite arguments from master
args = overwirte(args, argdict_master)

@info "Arguments" args...

###

using MLToolkit.Datasets

function get_dataset(name; kwargs...)
    args = (60_000,)
    if name in ("2dring", "3dring")
        n_mixtures, distance, var = 8, 2f0, 2f-1
        if name == "2dring"
            args = (args..., n_mixtures, distance, var)
        end
        if name == "3dring"
            args = (args..., Float32(pi / 3), 1f-1, n_mixtures, distance, var)
        end
    end
    if name == "mnist"
        kwargs = (kwargs..., is_flatten=true, alpha=0f0, is_link=false)
    end
    return Dataset(name, args...; kwargs...)
end

dataset = get_dataset(args.dataset; seed=args.seed)

###

using Logging, WeightsAndBiasLogger, Flux, MLToolkit.Neural, GRAM

# Freeze bias for CIFAR10
if args.dataset == "cifar10"
    @warn "Julia implementation for CIFAR10 is not maintained.
    To reproduce the results from our paper,
    please use the TensorFlow implementation, 
    which is available at https://github.com/GRAM-nets/GRAMFlow."
    Flux.trainable(c::Conv) = (c.weight,)
    Flux.trainable(ct::ConvTranspose) = (ct.weight,)
end

include(scriptsdir("model_by_args.jl"))

function get_cbeval(model, dataset)
    function cbeval()
        return evaluate(model, dataset)
    end
    return cbeval
end

###

if args.nowandb
    logger = NullLogger()
else
    logger = WBLogger(; project=projectname(), notes=args.notes)
    config!(logger, args; ignores=keys(args_ignored))
end

if args.opt == "adam"
    opt = ADAM(args.lr, (args.beta1, 999f-4))
elseif args.opt == "rmsprop"
    opt = RMSProp(args.lr)
end

model = get_model(args, dataset) |> gpu

cbeval = get_cbeval(model, dataset)

###

modeldir = datadir("results", args.dataset, savename(args; ignores=[keys(args_ignored)..., :dataset]))

args.is_continue && loadparams!(model, joinpath(modeldir, "model.bson"))

with(logger) do
    train!(
        opt, model, dataset.X, args.n_epochs, args.batch_size; kwargs...,
        evalevery=args.evalevery, cbeval=cbeval, 
        savedir=(args.nosave ? nothing : modeldir)
    )
end
