struct MMDGAN{T} <: Trainable
    σs::T
    g::NeuralSampler
    f_enc::Projector
    f_dec::Projector
end

@functor MMDGAN

function forward_and_loss(m::MMDGAN, x_data)
    x_gen = rand(m.g, last(size(x_data)))
    fx_gen, fx_data = m.f_enc(x_gen), m.f_enc(x_data)
    mmd = compute_mmd(fx_gen, fx_data, m.σs)
    one_side = -mean(relu.(-((mean(fx_gen; dims=2) - mean(fx_data; dims=2)))))
    return mmd, one_side, x_gen, fx_gen, fx_data
end

function Neural.update!(opt, m::MMDGAN, x_data)
    ps_fenc, ps_fdec, ps_g = params(m.f_enc), params(m.f_dec), params(m.g)
    ps_f = params([ps_fenc..., ps_fdec...])
    
    # NOTE: 3D ring fails with λ_ae_data=λ_ae_gen=8f0
    λ_rg, λ_ae_data, λ_ae_gen = 16f0, 8f0, 8f0
    
    # Update f
    local mmd_f, one_side_f, l2, l2_gen, loss_f
    for _ in 1:5
        
    # Clamp parameters of f encoder to a cube
    for p in ps_fenc
        p .= clamp.(p, -1f-1, 1f-1)
    end

    gs_f = gradient(ps_f) do
        mmd_f, one_side_f, x_gen, fx_gen, fx_data = forward_and_loss(m, x_data)
        xtilde_gen, xtilde = m.f_dec(fx_gen), m.f_dec(fx_data)
        l2 = mean((x_data - xtilde) .^ 2)
        l2_gen  = mean((x_gen  - xtilde_gen)  .^ 2)
        loss_f = -(mmd_f + λ_rg * one_side_f - λ_ae_data * l2 - λ_ae_gen * l2_gen)
        loss_f
    end
    Optimise.update!(opt, ps_f, gs_f)
    
    end # for
    
    # Update g
    local mmd_g, one_side_g, loss_g
    gs_g = gradient(ps_g) do
        mmd_g, one_side_g, _, _, _ = forward_and_loss(m, x_data)
        loss_g = mmd_g + λ_rg * one_side_g
    end
    Flux.Optimise.update!(opt, ps_g, gs_g)

    return (
        mmd_f=mmd_f,
        one_side_f=one_side_f, 
        l2=l2, 
        l2_gen=l2_gen,
        loss_f=loss_f,
        mmd_g=mmd_g,
        one_side_g=one_side_g, 
        loss_g=loss_g,
    )
end

evaluate(m::MMDGAN, ds) = evaluate(m.g, m.f_enc, ds)
