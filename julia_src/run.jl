using ArgParse 
using CSV
using DataFrames
using Tables

include("MCMC/private_neal5.jl")
include("MCMC/neal2.jl")
include("MCMC/utils.jl")


function run_mcmc(data, eps, hyperparams)
    if eps == 0
        chains = run_neal2(data, hyperparams, 5000, 10000);
        arate = 1
    else 
        chains, arate = run_neal5(data, hyperparams, eps, 5000, 10000, M);
    end

    return chains
end

function eval_chain_dens(chains, xgrid, hyperparams)
    dens = [eval_dens(s, hyperparams, xgrid) for s in chains];
    dens = mapreduce(permutedims, vcat, dens)
    return dens
end


function main()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--data"
            help = "csv file containing data"
        "--xgrid"
            help = "csv file containing grid on which the density is evaluated"
        "--eps"
            help = "Laplace noise"
            arg_type = Float64
        "--output_dens"
            help = "file where to save the density"
    end

    args = parse_args(ARGS, s)

    data = Matrix(CSV.read(args["data"], DataFrame, header=false))
    xgrid = Matrix(CSV.read(args["xgrid"], DataFrame, header=false))
    xgrid = vec(xgrid)
    
    hyperparams = NIGHyperParams(mean(data), 0.1, 2, 2, 1.0)

    chains = run_mcmc(data, args["eps"], hyperparams)
    dens = eval_chain_dens(chains, xgrid, hyperparams)

    CSV.write(args["output_dens"], Tables.table(dens), writeheader=false)
end

main()
