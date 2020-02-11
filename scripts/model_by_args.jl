using Humanize, Flux, MLToolkit.Neural, MLToolkit.DistributionsX

parse_csv(T, l) = map(x -> parse(T, x), split(l, ","))
parse_act(op::String) = eval(Symbol(op))
parse_act(op) = op

function WHC(D)
    if D == (28, 28, 1) || D == 28 * 28
        return (28, 28, 1)
    end
    if D == (32, 32, 3) || D == 32 * 32 * 3
        return (32, 32, 3)
    end
    error("Only MNIST-like and CIFAR-like data are supported.")
end

function get_model(args, dataset)
    Dx = dataset.dim
    Dz = args.Dz
    act = parse_act(args.act)
    actlast = parse_act(args.actlast)
    name = args.model
    isnorm = args.isnorm
    if args.base == "uniform"
        base = UniformNoise(Dz)
    end
    if args.base == "gaussian"
        base = GaussianNoise(Dz)
    end
    if args.Dhs_g == "conv"
        if Dx == (28, 28, 1) || prod(Dx) == 28 * 28
            convnet = ConvNet(Dz, (28, 28, 1), act, actlast; isnorm=isnorm)
        end
        if Dx == (32, 32, 3) || prod(Dx) == 32 * 32 * 3
            convnet = build_convnet_outcifar(Dz, act, actlast; isnorm=isnorm)
        end
        g = NeuralSampler(base, convnet)
    else
        Dhs_g = parse_csv(Int, args.Dhs_g)
        g = NeuralSampler(base, DenseNet(Dz, Dhs_g, Dx, act, actlast; isnorm=isnorm))
    end
    if name == "gan"
        if args.Dhs_d == "conv"
            d = Projector(ConvNet(WHC(Dx), 1, act, sigmoid; isnorm=isnorm))
        else
            Dhs_d = parse_csv(Int, args.Dhs_d)
            d = Projector(DenseNet(Dx, Dhs_d, 1, act, sigmoid; isnorm=isnorm).f)
        end
        m = GAN(g, d)
    else
        sigma = args.sigma == "median" ? [] : parse_csv(Float32, args.sigma)
        if name == "mmdnet"
            m = MMDNet(sigma, g)
        end
        if name == "mmdgan"
            if args.Dhs_f == "conv"
                error("Conv net is not supported for the projector of MMD-GAN.")
            else
                Dhs_f = parse_csv(Int, args.Dhs_f)
                f_enc = Projector(DenseNet(Dx, Dhs_f, args.Df, act, identity; isnorm=isnorm))
                f_dec = Projector(DenseNet(args.Df, Dhs_f, Dx, act, identity; isnorm=isnorm))
            end
            m = MMDGAN(sigma, g, f_enc, f_dec)
        end
        if name == "gramnet"
            if args.Dhs_f == "conv"
                f = Projector(ConvNet(WHC(Dx), args.Df, act, identity; isnorm=isnorm))
            else
                Dhs_f = parse_csv(Int, args.Dhs_f)
                f = Projector(DenseNet(Dx, Dhs_f, args.Df, act, identity; isnorm=isnorm))
            end
            m = GRAMNet(sigma, g, f)
        end
    end
    @info "Init $name with $(nparams(m) |> Humanize.digitsep) parameters"
    return m
end
