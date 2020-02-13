struct GRAMNet{T} <: Trainable
    σs::T
    g::NeuralSampler
    f
end

@functor GRAMNet

function Neural.update!(opt, m::GRAMNet, x_data; isclip_ratio=false, monitor_sqd=false)
    # Update f and g
    ps_f, ps_g = params(m.f), params(m.g)
    local x_gen, ratio, mmd, pearson_divergence, loss_f, loss_g, raito_mean
    gs_f, gs_g = gradient(ps_f, ps_g) do
        x_gen = rand(m.g, last(size(x_data)))
        fx_gen, fx_data = m.f(x_gen), m.f(x_data)
        ratio, mmd = estimate_ratio_compute_mmd(
            fx_gen, fx_data, m.σs; 
            isclip_ratio=isclip_ratio
        )
        loss_g = mmd
        pearson_divergence = mean((ratio .- 1).^2)
        loss_f = -pearson_divergence
        # We add a positivity regularizer if the ratio is not clipped
        if !isclip_ratio
            raito_mean = mean(ratio)
            loss_f -= raito_mean
        end
        (loss_f, loss_g)
    end
    Optimise.update!(opt, ps_f, gs_f)
    Optimise.update!(opt, ps_g, gs_g)

    info = (mmd=mmd, pearson_divergence=pearson_divergence, loss_f=loss_f, loss_g=loss_g)
    if !isclip_ratio
        info = (info..., raito_mean=raito_mean)
    end
    if monitor_sqd
        sqd = mean((estimate_ratio(flatten(x_gen), flatten(x_data), m.σs) - ratio).^2)
        info = (info..., squared_distance=sqd)
    end
    return info
end

evaluate(m::GRAMNet, ds) = evaluate(m.g, m.f, ds)
