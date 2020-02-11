module GRAM

using Statistics, LinearAlgebra, StatsFuns, Distributions, Humanize, Dates

parse_csv(T, l) = map(x -> parse(T, x), split(l, ","))
parse_act(op::String) = eval(Symbol(op))
parse_act(op) = op

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
        act = parse_act(args.act)
        actlast = parse_act(args.actlast)
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

export get_data, get_model

end # module
