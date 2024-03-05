using StatsBase
using Distributions
using ProgressBars
using NNlib

include("utils.jl")

function eval_marg(y, params::NIGHyperParams, state=nothing)
    sig_n = sqrt(params.b * (params.lam + 1) ) / (params.a * params.lam)
    marg = LocationScale(params.mu, sig_n, TDist(2 * params.a))
    return logpdf(marg, y)
end


function eval_marg(y, params::NormHyperParams, state=nothing)
    marg = Normal(params.mu, sqrt(state.var + params.s0))
    return logpdf(marg, y)
end


function one_iter!(state, data, prior)
    ndata = length(data)
    for i in 1:ndata
        state = remove_datum(i, state)
        # println("*********** 1\n", state)
        
        nclus = length(state.n_by_clus)
        probas = vcat(state.n_by_clus, [prior.alpha])
        probas ./= sum(probas)    
        logprobas = log.(probas)

        for h in 1:nclus
            m, v = get_mean_and_var(state, h)
            logprobas[h] += logpdf(Normal(m, sqrt(v)), data[i])
        end

        logprobas[nclus + 1] += eval_marg(data[i], prior, state)
        new_c = rand(Categorical(softmax(logprobas)))

        if new_c == nclus + 1
            # println("new atom")
            new_atom = sample_full_cond([data[i]], prior, state)
            state = add_atom_to_state!(state, new_atom)
            # println("*********** 2\n", state)
        end
        state.n_by_clus[new_c] += 1
        state.clus_allocs[i] = new_c

    end
    n_clus = length(state.n_by_clus)
    for h in 1:n_clus
        curr_data = vec(data[state.clus_allocs .== h])
        new_atom = sample_full_cond(curr_data, prior, state)
        state = update_atom!(state, new_atom, h)
        # println("*********** 3\n", state)
    end

    state = update_hypers(data, state, prior)
    # println("*********** 4\n", state)
    return state 
end


function run_neal2(data, hyperparams, n_burn=5000, n_iter=10000)
    state = initialize(hyperparams, length(data))

    chains = Array{Union{LSMixtureState,LocMixtureState}, 1}(undef, n_iter - n_burn)
    for i in 1:n_iter
        state = one_iter!(state, data, hyperparams)
        if i > n_burn
            chains[i - n_burn] = deepcopy(state)
        end
    end
    return chains
end
