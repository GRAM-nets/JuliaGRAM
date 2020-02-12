struct GRAMNet{T} <: Trainable
    σs::T
    g::NeuralSampler
    f::Projector
end

@functor GRAMNet

function Neural.update!(opt, m::GRAMNet, x_data)
    n_data = last(size(x_data))
    
    # Update f and g
    ps_f, ps_g = params(m.f), params(m.g)
    local x_gen, ratio, pearson_divergence, raito_mean, mmd, loss_f, loss_g
    gs_f, gs_g = gradient(ps_f, ps_g) do
        x_gen = rand(m.g, n_data)
        fx_gen, fx_data = m.f(x_gen), m.f(x_data)
        ratio, mmd = estimate_ratio_compute_mmd(fx_gen, fx_data, m.σs)
        loss_g = mmd
        pearson_divergence = mean((ratio .- 1).^2)
        loss_f = -pearson_divergence
        # Regularizer version
        #raito_mean = mean(ratio)
        #loss_f = -(pearson_divergence + mean(ratio))
        (loss_f, loss_g)
    end

    Optimise.update!(opt, ps_f, gs_f)
    Optimise.update!(opt, ps_g, gs_g)

    return (
        pearson_divergence=pearson_divergence,
        #raito_mean=raito_mean,
        mmd=mmd,
        loss_f=loss_f,
        loss_g=loss_g,
        #squared_distance=mean((estimate_ratio(flatten(x_gen), flatten(x_data), m.σs) - ratio).^2),
    )
end

evaluate(m::GRAMNet, ds) = evaluate(m.g, m.f, ds)

###

using TimerOutputs 
using Flux: Zygote

function Neural.update!(opt, m::GRAMNet, x_data, to)
    n_data = last(size(x_data))
    
    # Update f and g
    ps_f, ps_g = params(m.f), params(m.g)
    function forward()
        x_gen = rand(m.g, n_data)
        fx_gen, fx_data = m.f(x_gen), m.f(x_data)
        ratio, mmd = estimate_ratio_compute_mmd(fx_gen, fx_data, m.σs)
        loss_g = mmd
        pearson_divergence = mean((ratio .- 1).^2)
        loss_f = -pearson_divergence
        # Regularizer version
        #raito_mean = mean(ratio)
        #loss_f = -(pearson_divergence + mean(ratio))
        (loss_f, loss_g)
    end
    
    @timeit to "forward" ys, backs = Zygote.pullback(forward, ps_f, ps_g)
    
    n = length(ys)
    makesens(i) = tuple(map(j -> i == j ? Zygote.sensitivity(ys[i]) : zero(ys[i]), 1:n)...)
    @timeit to "gradient" gs_f, gs_g = tuple(map(i -> backs[i](makesens(i)), 1:n)...)

    @timeit to "updage f" Optimise.update!(opt, ps_f, gs_f)
    @timeit to "update g" Optimise.update!(opt, ps_g, gs_g)
    
    return to
end
