# The Official Julia Implementation of Generative Ratio Matching Networks

## How to run the code

1. Install [Julia](https://julialang.org/downloads/) and make `julia` available in your executable path.
2. Download the code in a location which we will refer as `GRAM_DIR`.
3. Start a Julia REPL by entering `julia` in your terminal.
    - Press `]` button to enter the package manager.
    - Activate the project environment by `activate $GRAM_DIR`.
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
    - This script by default using 9 cores to run experiments in parallel.
      - If you want to use another number of cores, please add `--n_cores $N`.

Our code by default logs all the training details using [Weights & Biases](https://www.wandb.ai/), please install W&B and set it up following [here](https://docs.wandb.com/quickstart). Or if you don't want to log things, add `--nowandb`.
