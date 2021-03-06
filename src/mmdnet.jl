struct MMDNet{T} <: Trainable
    σs::T
    g::NeuralSampler
end

@functor MMDNet

function (m::MMDNet)(x_data)
    x_gen = rand(m.g, last(size(x_data)))
    mmd = compute_mmd(flatten(x_gen), flatten(x_data), m.σs)
    return (loss_g=mmd, mmd=mmd,)
end

evaluate(m::MMDNet, ds) = evaluate(m.g, ds)
