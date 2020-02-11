using Flux: binarycrossentropy
const BCE = binarycrossentropy

struct GAN <: Trainable
    g::NeuralSampler
    d::Projector
end

@functor GAN

function Neural.update!(opt, m::GAN, x_data)
    n_data = last(size(x_data))
    n_gen = n_data
    n_both = n_data + n_gen
    y_real, y_fake = 1, 0
    
    # Update d
    ps_d = params(m.d)
    local ŷ_d_real, ŷ_d_fake, loss_d
    gs_d = gradient(ps_d) do
        x_gen = rand(m.g, n_gen)
        ŷ_d_real, ŷ_d_fake = m.d(x_data), m.d(x_gen)
        loss_d = (sum(BCE.(ŷ_d_real, y_real)) + sum(BCE.(ŷ_d_fake, y_fake))) / n_both
    end
    Optimise.update!(opt, ps_d, gs_d)
    
    # Compute accuracy for d
    n_true_real, n_true_fake = sum(ŷ_d_real .> 5f-1), sum(ŷ_d_fake .< 5f-1)
    accuracy_d = (n_true_real + n_true_fake) / n_both

    # Update g
    ps_g = params(m.g)
    local ŷ_g_fake, loss_g
    gs_g = gradient(ps_g) do
        x_gen = rand(m.g, n_gen)
        ŷ_g_fake = m.d(x_gen)
        loss_g = mean(BCE.(ŷ_g_fake, y_real))
    end
    Optimise.update!(opt, ps_g, gs_g)
    
    # Compute accuracy for g
    accuracy_g = mean(ŷ_g_fake .< 5f-1)
    
    return (
        loss_d=loss_d,
        loss_g=loss_g, 
        accuracy_d=accuracy_d,
        accuracy_g=accuracy_g,
    )
end

evaluate(m::GAN, ds) = evaluate(m.g, ds)
