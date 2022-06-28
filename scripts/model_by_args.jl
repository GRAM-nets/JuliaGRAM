using Flux, MLToolkit.Neural, MLToolkit.DistributionsX

parse_csv(T, l) = tuple(map(x -> parse(T, x), split(l, ","))...)
parse_act(op::String) = eval(Symbol(op))
parse_act(op::Expr) = eval(op)

function WHC(D)
    (D == (28, 28, 1) || D == 28 * 28 * 1) && return (28, 28, 1)
    (D == (32, 32, 3) || D == 32 * 32 * 3) && return (32, 32, 3)
    error("Only MNIST-like and CIFAR-like data are supported.")
end

function get_model(args, dataset)
    Dx = dataset.dim
    @unpack Dz, act, actlast, isnorm = args
    act, actlast = parse_act.((act, actlast))
    if args.base == "uniform"
        base = UniformNoise(Dz)
    elseif args.base == "gaussian"
        base = GaussianNoise(Dz)
    end
    if args.Dhs_g == "conv"
        g = NeuralSampler(base, ConvNet(Dz, WHC(Dx), act, actlast; isnorm=isnorm))
    else
        g = NeuralSampler(base, DenseNet(Dz, parse_csv(Int, args.Dhs_g), prod(Dx), act, actlast; isnorm=isnorm))
    end
    if args.model == "gan"
        if args.Dhs_d == "conv"
            d = ConvNet(WHC(Dx), 1, act, sigmoid; isnorm=isnorm)
        else
            d = DenseNet(prod(Dx), parse_csv(Int, args.Dhs_d), 1, act, sigmoid; isnorm=isnorm).f
        end
        m = GAN(g, d)
    else
        sigma = args.sigma == "median" ? [] : parse_csv(Float32, args.sigma)
        if args.model == "mmdnet"
            m = MMDNet(sigma, g)
        end
        if args.model == "mmdgan"
            if args.Dhs_f == "conv"
                error("Conv net is not supported for the projector of MMD-GAN.")
            else
                f_enc = DenseNet(prod(Dx), parse_csv(Int, args.Dhs_f), args.Df,  act, identity; isnorm=isnorm)
                f_dec = DenseNet(args.Df,  parse_csv(Int, args.Dhs_f), prod(Dx), act, identity; isnorm=isnorm)
            end
            m = MMDGAN(sigma, g, f_enc, f_dec)
        end
        if args.model == "gramnet"
            if args.Dhs_f == "conv"
                f = ConvNet(WHC(Dx), args.Df, act, identity; isnorm=isnorm)
            else
                f = DenseNet(prod(Dx), parse_csv(Int, args.Dhs_f), args.Df, act, identity; isnorm=isnorm)
            end
            m = GRAMNet(sigma, g, f)
        end
    end
    @info "Init $(args.model) with $(nparams(m)) parameters"
    return m
end
