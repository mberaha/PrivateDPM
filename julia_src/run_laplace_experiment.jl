using Distributions
using CSV
using Random
using ProgressBars
using Tables
using DataFrames
using Printf

include("MCMC/private_neal5.jl")

CURR_DIR = @__DIR__
OUT_DIR = joinpath(CURR_DIR, "../privacy_experiments/unidimensional_laplace/out/")
println(OUT_DIR)


M = 5
NREP = 1

 
Random.seed!(2024)

function simulate_private_data(ndata) 
    means = [-5, 0, 5]
    probas = ones(3) ./ 3
    clus_allocs = rand(Categorical(probas), ndata)
    data = rand.(Normal.(means[clus_allocs]))
    return data, clus_allocs
end


function run_experiment(ndata, repnum)
    println("RUNNING NDATA $(ndata), REP $(repnum)")
    private_data, clus = simulate_private_data(ndata);

    priv_levels = [1.0, 2.0, 5.0, 10.0, 50.0]
    hyperparams = NIGHyperParams(0.0, 0.1, 3.0, 3.0, 5.0)

    for alpha in priv_levels
        base_fname = OUT_DIR * "neal2m$(M)_ndata_$(ndata)_alpha_$(@sprintf("%.6f", alpha))_rep_$(repnum)_"
        eps = 20.0 / alpha
        
        # read from csv
        input_fname = OUT_DIR * "_ndata_$(ndata)_alpha_$(@sprintf("%.6f", alpha))_rep_$(repnum)_sanitized_data.csv"
        sanitized_data = CSV.read(input_fname, DataFrame, header=false).Column1

        # sanitized_data = private_data .+ rand(Laplace(0, eps), ndata)
        chains, arate = run_neal5(
            sanitized_data, hyperparams, eps, 150000, 160000, M);
        if repnum == 0
            xgrid = LinRange(-10, 10, 1000)
            dens = [eval_dens(s, hyperparams, xgrid) for s in chains];
            dens = mapreduce(permutedims, vcat, dens)
            dens = log.(dens)
            fname = base_fname * "eval_dens.csv"
            CSV.write(fname, Tables.table(dens), writeheader=false)
        end
        
        nclus_chain = zeros(length(chains))
        for (i, state) in enumerate(chains)
            nclus_chain[i] = length(state.means)
        end

        fname = base_fname * "nclus_chain.csv"
        CSV.write(fname, Tables.table(nclus_chain), writeheader=false)

        fname = base_fname * "acceptance_rate.csv"
        CSV.write(fname, Tables.table([arate]), writeheader=false)
    end
end

function main()
    i = 0
    @Threads.threads for i in ProgressBar(16:47)
    for ndata in [50, 100, 200, 500, 1000]
            run_experiment(ndata, i)
        end
    end
    
end


main()