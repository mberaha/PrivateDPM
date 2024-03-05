
mutable struct NIGHyperParams
    mu::Float64
    lam::Float64
    a::Float64
    b::Float64
    alpha::Float64
end

mutable struct NormHyperParams
    mu::Float64
    s0::Float64
    var_a::Float64
    var_b::Float64
    alpha::Float64
end

mutable struct LSMixtureState 
    means
    vars
    clus_allocs
    n_by_clus
end

mutable struct LocMixtureState
    means
    var
    clus_allocs
    n_by_clus
end

function sample_atom(params::NIGHyperParams)
    var = rand(InverseGamma(params.a, params.b))
    mean = rand(Normal(params.mu,  sqrt( var / params.lam)))
    return mean, var
end

function sample_atom(params::NormHyperParams)
    mean = rand(Normal(params.mu,  params.s0))
    return mean
end

function remove_datum(i, state::LSMixtureState)
    state = deepcopy(state)
    curr_c = state.clus_allocs[i]
    state.n_by_clus[curr_c] -= 1
    if state.n_by_clus[curr_c] == 0
        # println("REMOVING SINGLETON")
        deleteat!(state.means, curr_c)
        deleteat!(state.vars, curr_c)
        deleteat!(state.n_by_clus, curr_c)
        state.clus_allocs[state.clus_allocs .> curr_c] .-= 1
    end
    return state
end

function remove_datum(i, state::LocMixtureState)
    state = deepcopy(state)
    curr_c = state.clus_allocs[i]
    state.n_by_clus[curr_c] -= 1
    if state.n_by_clus[curr_c] == 0
        # println("REMOVING SINGLETON")
        deleteat!(state.means, curr_c)
        deleteat!(state.n_by_clus, curr_c)
        state.clus_allocs[state.clus_allocs .> curr_c] .-= 1
    end
    return state
end

function sample_full_cond(data, params::NormHyperParams, state::LocMixtureState) 
    n = length(data)
    xbar = mean(data)
    post_var = 1.0 / (1.0 / params.s0 + n / state.var)
    post_mean = (params.mu / params.s0 + n * xbar / state.var) * post_var
    post_params = NormHyperParams(
        post_mean, post_var, -1, -1, -1)
    return sample_atom(post_params)
end 

function sample_full_cond(data, prior::NIGHyperParams, state::LSMixtureState=nothing) 
    n = length(data)
    xbar = mean(data)
    post_lam = prior.lam + n
    post_mean = (prior.lam * prior.mu + n * xbar) / post_lam
    post_a = prior.a + n / 2
    post_b = prior.b + 0.5 * sum( (data .- xbar).^2 ) + 
        0.5 * n * prior.lam / post_lam * (xbar - prior.mu)^2
    post_params = NIGHyperParams(post_mean, post_lam, post_a, post_b, -1)
    return sample_atom(post_params)
end 


function eval_dens(state::LSMixtureState, prior::NIGHyperParams, xgrid) 
    n_mc_marg = 100
    mc_vars = rand(InverseGamma(prior.a, prior.b), n_mc_marg)
    mc_means = rand.(Normal.(prior.mu, sqrt.(mc_vars ./ prior.lam)))
    vars = vcat(state.vars, mc_vars)
    means = vcat(state.means, mc_means)

    weights = vcat(state.n_by_clus, ones(n_mc_marg) .* prior.alpha ./  n_mc_marg)
    weights ./= sum(weights)

    out = zeros(size(xgrid))
    for (w, m, v) in zip(weights, means, vars)
        out .+= w .* pdf.(Normal(m, sqrt(v)), xgrid)
    end

    return out
end

function eval_dens(state::LocMixtureState, prior::NormHyperParams, xgrid) 
    n_mc_marg = 100
    mc_means = rand(Normal(prior.mu, sqrt(prior.s0)), n_mc_marg)
    means = vcat(state.means, mc_means)

    weights = vcat(state.n_by_clus, ones(n_mc_marg) .* prior.alpha ./  n_mc_marg)
    weights ./= sum(weights)

    out = zeros(length(xgrid))
    for (w, m) in zip(weights, means)
        out .+= w .* pdf.(Normal(m, sqrt(state.var)), xgrid)
    end

    return out
end

function initialize(hyperparams::NIGHyperParams, ndata)
    init_n_clus = 3
    init_var = hyperparams.lam * hyperparams.b / hyperparams.a
    p = ones(init_n_clus)
    p ./= sum(p)
    init_clus = rand(Categorical(p), ndata)
    n_by_clus = zeros(init_n_clus)
    for k in 1:init_n_clus
        n_by_clus[k] = sum(init_clus .== k)
    end

    state = LSMixtureState(
        rand(Normal(hyperparams.mu, sqrt(init_var)), init_n_clus),
        rand(InverseGamma(hyperparams.a, 1.0 / hyperparams.b), init_n_clus),
        init_clus,
        n_by_clus
    )
    return state
end

function initialize(hyperparams::NormHyperParams, ndata)
    init_n_clus = 3
    init_var = hyperparams.var_b / (hyperparams.var_a + 1)
    p = ones(init_n_clus)
    p ./= sum(p)
    init_clus = rand(Categorical(p), ndata)
    n_by_clus = zeros(init_n_clus)
    for k in 1:init_n_clus
        n_by_clus[k] = sum(init_clus .== k)
    end

    state = LocMixtureState(
        rand(Normal(hyperparams.mu, sqrt(hyperparams.s0)), init_n_clus),
        init_var,
        init_clus,
        n_by_clus
    )
    return state
end

function update_atom!(state::LSMixtureState, atom, atom_index)
    m, s = atom
    state.means[atom_index] = m
    state.vars[atom_index] = s
    return state
end

function update_atom!(state::LocMixtureState, atom, atom_index)
    m = atom
    state.means[atom_index] = m
    return state
end

function add_atom_to_state!(tmp_state::LSMixtureState, atom)
    m, s = atom
    tmp_state.means = push!(tmp_state.means, m)
    tmp_state.vars = push!(tmp_state.vars, s)
    tmp_state.n_by_clus = push!(tmp_state.n_by_clus, 0)
    return tmp_state
end

function add_atom_to_state!(tmp_state::LocMixtureState, atom)
    m = atom
    tmp_state.means = push!(tmp_state.means, m)
    tmp_state.n_by_clus = push!(tmp_state.n_by_clus, 0)
    return tmp_state
end

function get_mean_and_var(state::LSMixtureState, comp_index)
    return state.means[comp_index], state.vars[comp_index]
end

function get_mean_and_var(state::LocMixtureState, comp_index)
    return state.means[comp_index], state.var
end

function update_hypers(y, state::LSMixtureState, params::NIGHyperParams)
    return state 
end

function update_hypers(y, state::LocMixtureState, params::NormHyperParams)
    n = length(y)
    post_a = params.var_a + n * 0.5
    post_b = params.var_b + 0.5 * sum((y - state.means[state.clus_allocs]).^2)
    state.var = rand(InverseGamma(post_a, post_b))
    return state
end


function py_sb_weights(theta, alpha, L)
    nus = rand.(Beta.(1 - alpha, theta .+ alpha .* collect(1:L)))
    ws = zeros(L)
    ws[1] = nus[1]
    ws[2:(end-1)] = nus[2:(end-1)] .* cumprod(1 .- nus[1:end-2])
    ws[end] = 1.0 - sum(ws[1:end-1])
    return ws
end

function eval_dens_conditional(state::LocMixtureState, prior::NormHyperParams, xgrid) 
    n_mc = 100
    
    new_weights = py_sb_weights(prior.alpha, 0, n_mc)
    new_means = rand.(Normal.(prior.mu, sqrt.(prior.s0)))
    
    dir_params = vcat(state.n_by_clus, [prior.alpha])
    dir_weigths = rand(Dirichlet(dir_params))

    mix_weigths = vcat(dir_weigths[1:end-1], dir_weigths[end] * new_weights)
    mix_means = vcat(state.means, new_means)

    out = zeros(length(xgrid))
    s = sqrt(state.var)
    for (w, m) in zip(mix_weigths, mix_means)
        out += w .* pdf.(Normal(m, s), xgrid)
    end

    return out
end


function eval_dens_conditional(state::LSMixtureState, prior::NIGHyperParams, xgrid) 
    n_mc = 100

    new_weights = py_sb_weights(prior.alpha, 0, n_mc)
    new_vars = rand(InverseGamma(prior.a, prior.b), n_mc)
    new_means = rand.(Normal.(prior.mu, sqrt.(new_vars ./ prior.lam)))
    
    dir_params = vcat(state.n_by_clus, [prior.alpha])
    dir_weigths = rand(Dirichlet(dir_params))

    mix_weigths = vcat(dir_weigths[1:end-1], dir_weigths[end] * new_weights)
    mix_means = vcat(state.means, new_means)
    mix_vars = vcat(state.vars, new_vars)

    out = zeros(size(xgrid))
    for (w, m, s2) in zip(mix_weigths, mix_means, mix_vars)
        out .+= w .* pdf.(Normal(m, sqrt(s2)), xgrid)
    end

    return out
end
