using DrWatson
@quickactivate "GRAM"

###

if length(ARGS) == 1
    dict = load(projectdir("_research", "tmp", ARGS[1]))
else
    dict = Dict(
        :dataset    =>  "3dring",
        :model      =>  "gramnet",
    )
end

args = (
    seed       = 1,
    dataset    = dict[:dataset],
    model      = dict[:model],
    batch_size = 1,
)

@assert args.dataset in ["gaussian", "2dring", "3dring", "mnist", "cifar10"]
@assert args.model in ["gramnet", "mmdnet", "gan"]

###

using Logging, WeightsAndBiasLogger

logger = WBLogger(; project="ds-gem")
config!(logger, args)

###

