function concat_predefined_args(args)

let dataset = args.dataset, model = args.model

throw_undef_error(dataset, model) = error("(dataset=$dataset, model=$model) is not defined")

if dataset == "gaussian"
    args = (args..., 
        base    = "uniform",
        Dz      = 10,
        Dhs_g   = "50,50",
        act     = "tanh",
        actlast = "identity",
        isnorm  = false,
    )
    if model == "gramnet"
        args = (args...,
            opt   = "adam",
            lr    = 1f-3,
            beta1 = 5f-1,
            sigma = "1,2",
            Dhs_f = "50,25",
            Df    = 5,
        )
    end
    if model == "mmdgan"
        throw_undef_error(dataset, model)
    end
    if model == "mmdnet"
        args = (args...,
            opt   = "rmsprop",
            lr    = 1f-3,
            sigma = "1,2",
        )
    end
    if model == "gan"
        args = (args...,
            opt   = "adam",
            lr    = 1f-4,
            beta1 = 5f-1,
            Dhs_d = "50,25",
        )
    end
end # if

if dataset in ["2dring", "3dring"]
    args = (args..., 
        base    = "gaussian",
        Dz      = 20,
        Dhs_g   = "100,100",
        act     = "relu",
        actlast = "identity",
        isnorm  = false,
    )
    if model == "gramnet"
        args = (args...,
            opt   = "adam",
            lr    = 1f-3,
            beta1 = 5f-1,
            sigma = "1",
            Dhs_f = "100,100",
            Df    = 10,
        )
    end
    if model == "mmdgan"
        args = (args...,
            opt   = "adam",
            lr    = 5f-5,
            beta1 = 5f-1,
            sigma = "1",
            Dhs_f = "100,100",
            Df    = 10,
        )
    end
    if model == "mmdnet"
        args = (args...,
            opt   = "rmsprop",
            lr    = 1f-3,
            sigma = "1",
        )
    end
    if model == "gan"
        args = (args...,
            opt   = "adam",
            lr    = 1f-4,
            beta1 = 5f-1,
            Dhs_d = "100,100",
        )
    end
end # if

if dataset == "mnist"
    args = (args..., 
        base    = "uniform",
        Dz      = 100,
        Dhs_g   = "600,600,800",
        act     = "relu",
        actlast = "sigmoid",
        isnorm  = true,
    )
    if model == "gramnet"
        args = (args...,
            opt   = "adam",
            lr    = 1f-3,
            beta1 = 5f-1,
            sigma = "0.1,1,10,100",
            Dhs_f = "conv",
            Df    = 100,
        )
    end
    if model == "mmdgan"
        throw_undef_error(dataset, model)
    end
    if model == "mmdnet"
        args = (args...,
            opt   = "rmsprop",
            lr    = 1f-3,
            sigma = "1,5,10",
        )
    end
    if model == "gan"
        args = (args...,
            opt   = "adam",
            lr    = 1f-4,
            beta1 = 5f-1,
            Dhs_d = "400,200",
        )
    end
end # if

if dataset == "cifar10"
    args = (args..., 
        base    = "uniform",
        Dz      = 150,
        Dhs_g   = "conv",
        act     = "x -> leakyrelu(x, 2f-1)",
        actlast = "identity",
        isnorm  = true,
        opt     = "adam",
        beta1   = 5f-1,
    )
    if model == "gramnet"
        args = (args...,
            lr    = 2f-4,
            sigma = "1,2,4,8,16",
            Dhs_f = "conv",
            Df    = 150,
        )
    end
    if model == "mmdgan"
        throw_undef_error(dataset, model)
    end
    if model == "mmdnet"
        args = (args...,
            lr    = 1f-3,
            sigma = "1,2,4,8,16",
        )
    end
    if model == "gan"
        args = (args...,
            lr    = 1f-4,
            Dhs_d = "conv",
        )
    end
end # if

end # let

return args

end # function
