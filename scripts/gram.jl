using DrWatson
@quickactivate "GRAM"

###

if length(ARGS) == 1
    dict = load(projectdir("_research", "tmp", first(ARGS)))
else
    dict = Dict(
        :dataset => "3dring",
        :model   => "gramnet",
    )
end

args_ignored = (
    n_epochs    = 10,
    evalevery   = 100,
    is_continue = false,
)

args = (args_ignored...,
    seed        = 1,
    dataset     = dict[:dataset],
    model       = dict[:model],
    batch_size  = 200,
)

@assert args.dataset in ["gaussian", "2dring", "3dring", "mnist", "cifar10"]
@assert args.model in ["gramnet", "mmdgan", "mmdnet", "gan"]

include(scriptsdir("predefined_args.jl"))
args = concat_predefined_args(args)

###

using WeightsAndBiasLogger

logger = WBLogger(; project=projectname())
config!(logger, args; ignores=keys(args_ignored))

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
    return Dataset(name, args...; kwargs...)
end

dataset = get_dataset(args.dataset; seed=args.seed)

###

using Flux, MLToolkit.Neural

# Flux.trainable(c::Conv) = (c.weight,)
# Flux.trainable(ct::ConvTranspose) = (ct.weight,)

include(srcdir("models.jl"))
using .Models

include(scriptsdir("model_by_args.jl"))

function get_cbeval(model, dataset)
    function cbeval()
        return evaluate(model, dataset)
    end
    return cbeval
end

###

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
        opt, model, dataset.X, args.n_epochs, args.batch_size;
        evalevery=args.evalevery, cbeval=cbeval, savedir=modeldir
    )
end
