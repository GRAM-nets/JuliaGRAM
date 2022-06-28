using DrWatson
@quickactivate "GRAM"
using MLToolkit.Scripting

###

if length(ARGS) == 1 || length(ARGS) == 2
    argdict_master = load(projectdir("_research", "tmp", ARGS[1]))["params"]
    if length(ARGS) == 2
        gpu_id = parse(Int, ARGS[2])
        using Flux.CuArrays: device!
        device!(gpu_id)
    end
else
    argdict_master = Dict(
        :dataset => "cifar10",
        :model   => "gramnet",
    )
end

@assert :dataset in keys(argdict_master)
@assert :model in keys(argdict_master)

args_ignored = (
    n_epochs    = 300,
    evalevery   = 100,
    is_continue = false,
    nowandb     = false,
    wandb_name  = "$(projectname())-private",   # change this to your W&B project
    notes       = "",
    nosave      = true,
    modelpath   = "",   # change this to load from specific path
)

args = (args_ignored...,
    seed        = 1,
    dataset     = argdict_master[:dataset],
    model       = argdict_master[:model],
    batch_size  = 100,
)

@assert args.dataset in ["gaussian", "2dring", "3dring", "mnist", "cifar10"]
@assert args.model in ["gramnet", "mmdgan", "mmdnet", "gan"]

# Load pre-defined arguments
include(scriptsdir("predefined_args.jl"))
args = concat_predefined_args(args)

# Add keyword arguments
kwargs = args.model == "gramnet" ? (isclip_ratio=false, lambda=1f0, ismonitor_sqd=false) : NamedTuple()
length(kwargs) > 0 && (args = union(args, kwargs))

# Overwrite arguments from master
args = overwrite(args, argdict_master)

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
    if name in ["mnist", "cifar10"]
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
    logger = WBLogger(; project=args.wandb_name, notes=args.notes)
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

args.is_continue && loadparams!(model, args.modelpath == "" ? joinpath(modeldir, "model.bson") : args.modelpath)

with(logger) do
    train!(
        opt, model, dataset.X, args.n_epochs, args.batch_size; kwargs...,
        evalevery=args.evalevery, cbeval=cbeval, 
        savedir=(args.nosave ? nothing : modeldir)
    )
end
