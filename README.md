# JuliaGRAM: Julia implementation of GRAM-nets

This is the source code for the paper [Generative Ratio Matching Networks](https://openreview.net/forum?id=SJg7spEYDS).

## Check out the results interactively right now

Our experiments are logged by [WeightsAndBiasLogger.jl](https://github.com/xukai92/WeightsAndBiasLogger.jl), 
a Julia interface for [Weights & Biases](https://www.wandb.ai/), 
please check them [here](https://app.wandb.ai/xukai92/gram-public).

## How to run the code?

1. Install [Julia](https://julialang.org/downloads/) and make `julia` available in your executable path.
2. Download the code in a location which we will refer as `GRAM_DIR`.
3. Start a Julia REPL by entering `julia` in your terminal.
    - Press `]` button to enter the package manager.
    - Install [DrWatson](https://github.com/JuliaDynamics/DrWatson.jl) by `add DrWatson`.
    - Activate the project environment by `activate $GRAM_DIR`.
    - Install all dependencies by `instantiate`.
    - Press `delete` or `backspace` to exit the package manager.
    - Input `using PyCall`.
        - Input `PyCall.Conda.add("matplotlib")` to install matplotlib.
        - Input `PyCall.Conda.add("wandb")` to install wandb (optional).
    - Exit the REPL.
4. Do `julia $GRAM_DIR/scripts/master.jl --exp 1`
    - This will produce Figure 1 (and Figure 6 & 9 in the appendix).
    - To produce other plots, change the argument to
        - `--exp 2` for Figure 2 (and Figure 7 in the appendix), or
        - `--exp 3` for Figure 8 in the appendix.
    - You can also adjust the parameter sweep in `master.jl` by yourself.
    - If you want to run multiple trainings in parallel, please append `JULIA_NUM_THREADS=$N_PARALLEL ` to the beginning of the command (with a space), where `N_PARALLEL` is the number of parallel trainings you want to run.
      - By default the script uses 2 GPUs for parallelism. You can add `--n_gpus $N_GPUS` to the command to control the number.

You can also modify the arguments in `scripts/gram.jl` and run the file on its own by `julia $GRAM_DIR/scripts/gram.jl`

Our code by default logs all the training details using [Weights & Biases](https://www.wandb.ai/), 
please install W&B and set it up following [here](https://docs.wandb.com/quickstart). 
Or if you don't want to log things, add `--nowandb`.

---

| Maintainer |
| :-: |
| [Kai Xu](http://xuk.ai/) |