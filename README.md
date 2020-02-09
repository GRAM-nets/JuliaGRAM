# Source code for *Generative Ratio Matching Networks*

## How to run the code

1. Install [Julia](https://julialang.org/downloads/) and make `julia` available in your executable path.
2. Download the code in a location which we will refer as `GRAM_DIR`.
3. Start a Julia REPL by entering `julia` in your terminal.
    - Press `]` button to enter the package manager.
    - Input `dev $GRAM_DIR`.
        - This will install all the Julia dependencies for you and might take a while.
    - Activate the project environment by `activate $GRAM_DIR`.
    - Press `delete` or `backspace` to exit the package manager.
    - Input `using PyCall`.
        - Input `PyCall.Conda.add("matplotlib")` to install matplotlib.
    - Exit the REPL.
4. Do `julia --project=$GRAM_DIR $GRAM_DIR/examples/parallel_exps.jl`
    - This by default produce Figure 1 (and Figure 6 in the appendix).
    - To produce other plots, you need to edit `parallel_exps.jl` as below and run the same command.
        - For Figure 2, uncomment L60 and L61.
        - For Figure 7 in the appendix, uncomment L64.
    - This script by default using 9 cores to run experiments in parallel. If you want to use another number of cores, please change L2. 

Our code by default logs all the training details in `$GRAM_DIR/logs`, for which you can view using TensorBoard. 