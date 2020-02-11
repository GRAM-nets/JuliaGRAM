using DrWatson
@quickactivate "GRAM"

###

if length(ARGS) == 1
    dict = load(projectdir("_research", "tmp", first(ARGS)))
else
    dict = Dict(
        :dataset => "3dring",
        :model   => "gan",
    )
end

args_ignored = (
    n_epochs  = 10,
    evalevery = 10,
    is_continue = false,
)

args = (args_ignored...,
    seed        = 1,
    dataset     = dict[:dataset],
    model       = dict[:model],
    batch_size  = 200,
    
)

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

include(srcdir("models.jl"))
using .Models

include(scriptsdir("model_by_args.jl"))

###

model = get_model(args, dataset) |> gpu
modeldir = datadir("results", args.dataset, savename(args; ignores=[keys(args_ignored)..., :dataset]))

if args.opt == "adam"
    opt = ADAM(args.lr, (args.beta1, 999f-4))
elseif args.opt == "rmsprop"
    opt = RMSProp(args.lr)
end

args.is_continue && loadparams!(model, joinpath(modeldir, "model.bson"))

with(logger) do
    train!(
        opt, model, dataset.X, args.n_epochs, args.batch_size;
        evalevery=args.evalevery, cbeval=(() -> evaluate(model, dataset))
    )
end
