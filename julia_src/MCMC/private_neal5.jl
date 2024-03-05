using StatsBase
using Distributions
using ProgressBars

include("utils.jl")

function propose_c(n_by_clus, dp_alpha)
    probas = vcat(n_by_clus, [dp_alpha])
    probas ./= sum(probas)
    return rand(Categorical(probas))
end


function one_iter!(state, private_data, public_data, prior, lap_scale)
    n_prop = 0
    n_acc = 0
    ndata = size(private_data)[1]
    n_aux = size(private_data)[2]
    for i in 1:ndata
        tmp_state = remove_datum(i, state)
        new_c = propose_c(tmp_state.n_by_clus, prior.alpha)
        n_prop += 1
        if new_c == length(tmp_state.n_by_clus) + 1
            tmp_atom = sample_atom(prior)
            tmp_state = add_atom_to_state!(tmp_state, tmp_atom)
            # tmp_state.means = push!(tmp_state.means, m)
            # tmp_state.vars = push!(tmp_state.vars, s)
            # tmp_state.n_by_clus = push!(tmp_state.n_by_clus, 0)
        end

        m, s = get_mean_and_var(tmp_state, new_c)
        y_prime = rand(Normal(m, sqrt(s)), n_aux)
        a_rate = sum(pdf.(Laplace.(y_prime, lap_scale), public_data[i])) / sum(
            pdf.(Laplace.(private_data[i, :], lap_scale), public_data[i]))

        if rand(Uniform()) < a_rate
            n_acc += 1
            state = tmp_state 
            state.clus_allocs[i] = new_c
            state.n_by_clus[new_c] += 1
            private_data[i, :] .= y_prime
        else
        end 

        @assert(all(state.n_by_clus .>= 1))
        @assert(sum(state.n_by_clus) == ndata)
    end
    n_clus = length(state.n_by_clus)

    for h in 1:n_clus
        curr_data = vec(private_data[state.clus_allocs .== h, :])
        atom = sample_full_cond(curr_data, prior, state)
        state = update_atom!(state, atom, h)
    end

    state = update_hypers(private_data, state, prior)
    return state, private_data, n_prop, n_acc
end

function initialize(hyperparams, ndata)
    init_n_clus = 3
    init_var = hyperparams.lam * hyperparams.b / hyperparams.a
    p = ones(init_n_clus)
    p ./= sum(p)
    init_clus = rand(Categorical(p), ndata)
    n_by_clus = zeros(init_n_clus)
    for k in 1:init_n_clus
        n_by_clus[k] = sum(init_clus .== k)
    end

    state = MixtureState(
        rand(Normal(hyperparams.mu, sqrt(init_var)), init_n_clus),
        rand(InverseGamma(hyperparams.a, 1.0 / hyperparams.b), init_n_clus),
        init_clus,
        n_by_clus
    )
    return state
end

function run_neal5(public_data, hyperparams, laplace_scale, n_burn=5000, n_iter=10000, n_aux=1)
    n_prop = 0
    n_acc = 0
    state = initialize(hyperparams, length(public_data))
    private_data = zeros(length(public_data), n_aux)
    for i in 1:length(public_data)
        c = state.clus_allocs[i]
        m, s = get_mean_and_var(state, c)
        private_data[i, :] = rand(Normal(m, s), n_aux)
    end

    chains = Array{Union{LSMixtureState,LocMixtureState}, 1}(undef, n_iter - n_burn)
    for i in 1:n_iter
        state, private_data, prop, acc = one_iter!(
            state, private_data, public_data, hyperparams, laplace_scale)
        n_prop += prop
        n_acc += acc
        if i > n_burn
            chains[i - n_burn] = deepcopy(state)
        end
    end
    return chains, n_acc / n_prop
end

