struct GRAMNet <: Trainable
    σs
    g::NeuralSampler
    f::Projector
end

@functor GRAMNet

function Neural.update!(opt, m::GRAMNet, x_data)
    # Update f and g
    ps_f, ps_g = params(m.f), params(m.g)
    
    local x_gen, ratio, pearson_divergence, raito_mean, mmd, loss_f, loss_g
    gs_f, gs_g = gradient(ps_f, ps_g) do
        x_gen = rand(m.g, last(size(x_data)))
        fx_gen, fx_data = m.f(x_gen), m.f(x_data)
        ratio, mmd = estimate_ratio_compute_mmd(fx_gen, fx_data; σs=m.σs)
        loss_g = mmd
        # Clip version; thanks to our reviewer
        ratio = clamp.(ratio, 0f0, 1f9)
        pearson_divergence = mean((ratio .- 1) .^ 2)
        loss_f = -pearson_divergence
        # Regularizer version
#         raito_mean = mean(ratio)
#         pearson_divergence = mean((ratio .- 1) .^ 2)
#         loss_f = -(pearson_divergence + raito_mean)
        (loss_f, loss_g)
    end

    Optimise.update!(opt, ps_f, gs_f)
    Optimise.update!(opt, ps_g, gs_g)

    return (
        pearson_divergence=pearson_divergence,
#         raito_mean=raito_mean,
        mmd=mmd,
        loss_f=loss_f,
        loss_g=loss_g,
#         squared_distance=mean((estimate_ratio(flatten(x_gen), flatten(x_data); σs=m.σs) - ratio).^2),
    )
end

evaluate(m::GRAMNet, ds) = evaluate(m.g, m.fenc, ds)
