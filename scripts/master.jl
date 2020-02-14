using DrWatson
@quickactivate "GRAM"
using ArgParse

s = ArgParseSettings()
@add_arg_table! s begin
    "--exp"
        arg_type = Int
        required = true
    "--n_gpus"
        arg_type = Int
        default  = 2
    "--nowandb"
        action = :store_true
end
args = dict2ntuple(parse_args(s))

@info "Master arguments" args...

if args.exp == 1    # Figure 1, 6 & 9; 16 runs
    general_args = Dict(
        :notes        => "Varying Df",
        :dataset      => ["2dring", "3dring"],
        :model        => ["gramnet", "mmdgan"],
        :Df           => [2, 4, 8, 16],
        :nowandb      => args.nowandb,
        :isclip_ratio => false,
    )
end

if args.exp == 2    # Figure 2 & 7; 96 runs
    general_args = Dict(
        :notes        => "Varying Dz and Dhs_g",
        :dataset      => ["2dring", "3dring"],
        :model        => ["gramnet", "mmdgan", "mmdnet", "gan"],
        :Dz           => [2, 4, 8, 16],
        :Dhs_g        => ["20,20", "100,100", "200,200"],
        :nowandb      => args.nowandb,
        :isclip_ratio => false,
    )
end

if args.exp == 3    # Figure 8; 1 run
    general_args = Dict(
        :notes        => "MNIST",
        :dataset      => "mnist",
        :model        => "gramnet",
        :lr           => 1f-3,
        :Dhs_g        => "600,600,800",
        :Dhs_f        => "conv",
        :sigma        => "0.1,1,10,100",
        :nowandb      => args.nowandb,
        :isclip_ratio => false,
    )
end

dicts = dict_list(general_args)
paths = tmpsave(dicts)
Threads.@threads for (p, d) in collect(zip(paths, dicts))
    withenv("CUDA_VISIBLE_DEVICES" => (Threads.threadid() % args.n_gpus)) do
        run(`julia $(scriptsdir("gram.jl")) $p`)
    end
end
