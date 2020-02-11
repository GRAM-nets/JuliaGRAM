struct MMDNet <: Trainable
    σs
    g::NeuralSampler
end

@functor MMDNet

function Neural.loss(m::MMDNet, x_data)
    x_gen = rand(m.g, last(size(x_data)))
    mmd = compute_mmd(x_gen |> flatten, x_data |> flatten; σs=m.σs)
    return (loss=mmd, mmd=mmd,)
end

evaluate(m::MMDNet, ds) = evaluate(m.g, ds)
