#!/usr/bin/env julia

##
## Author: Jeremy Van Cleve <jvancleve@uky.edu>
##
## Simulation of selection in an island-model

module metapop
export Population, Individual, Landscape
export wf_selection!, mutation!, migration!
export group_competition!, payoff!, addpayoff,  addpayoff_rand
export evolve_neutral!

using Distributions

struct Landscape
    b::Float64
    c::Float64
    δ::Float64
    μ::Array{Float64,1}
    m::Float64
    payfunc::Function

    Landscape(b, c, δ, μ, m, pf) = new(b, c, δ, μ, m, pf)
end

mutable struct Individual
    genotype::Array{Int8,1}
end

function copy(ind::Individual)
    return Individual(ind.genotype)
end

function copy!(i::Individual, j::Individual)
    i.genotype = j.genotype
end

mutable struct Population
    N::Array{Int64,1}
    landscape::Landscape
    payoff::Array{Float64,1}
    dispersed::Array{Float64,1}
    fertility::Array{Float64,2}
    fitness::Array{Float64,2}
    members::Array{Individual,1}
    offspring::Array{Individual,1}
    mean_allele::Array{Float64,2}
    i2d::Dict{Int64, Array{Int64,1}}
    d2i::Dict{Array{Int64,1}, Int64}
    generation::Int64
end

### Population constructor
function Population(N, landscape, init_geno::Array{Int64,2})
    nd = length(N)
    NT = sum(N)
    nloci = size(init_geno, 2)
    i2d, d2i = ind2deme(N)

    if NT != size(init_geno, 1)
        throw(ArgumentError(string("NT = ", NT, " != ",
                                   size(init_geno, 1), " = size(init_geno, 1)")))
    end
    if nloci != length(landscape.μ)
        throw(ArgumentError(string("nloci = ", nloci, " != ",
                                   landscape.μ, " = landscape.μ")))
    end

    parents = Array{Individual,1}()
    offspring = Array{Individual,1}()
    mean_allele = zeros((nd,nloci))
    for i in 1:NT
        d = i2d[i][1]
        push!(parents, Individual(init_geno[i,:]))
        push!(offspring, Individual(init_geno[i,:]))
        mean_allele[d,:] += init_geno[i,:] / N[d]
    end

    Population(N, landscape, zeros(NT), zeros(NT), zeros((nd,NT)), zeros((nd,NT)),
               parents, offspring, mean_allele, i2d, d2i, 0)
end

# creat dict mapping linear index of individuals to indices for
# deme and individual in deme
function ind2deme(N::Array{Int64,1})
    nd = length(N)
    i2d = Dict{Int64, Array{Int64,1}}()
    d2i = Dict{Array{Int64,1}, Int64}()
    for d in 1:nd
        for i in 1:N[d]
            d2i[ [d, i] ] = i + sum(N[1:d-1])
            i2d[ d2i[ [d, i] ] ] = [d, i]
        end
    end

    return i2d, d2i
end

# update deme allele frequencies
function update_mean_allele!(pop::Population)
    nd = length(pop.N)
    pop.mean_allele .= 0.0
    for d in 1:nd
        for i in 1:pop.N[d]
            @. @views pop.mean_allele[d,:] += pop.members[pop.d2i[[d,i]]].genotype / pop.N[d]
        end
    end
end

function payoff!(pop::Population)
    NT = sum(pop.N)
    pf = pop.landscape.payfunc

    update_mean_allele!(pop)
    for i in 1:NT
        pop.payoff[i] = pf(i, pop)
    end
end

# additive payoff function
function addpayoff(i::Int64, pop::Population)
    l = pop.landscape
    d = pop.i2d[i][1]
    N = pop.N[d]

    p_not_i = (N/(N-1) * pop.mean_allele[d] - pop.members[i].genotype[1]/(N-1))
    if p_not_i < 0
        throw(ErrorException(string(p_not_i, ". N: ", N, " i: ", i,
                                    " (d,i): (", d, ",", pop.i2d[i][2], ")",
                                    " mean ", pop.mean_allele[d],
                                    " focal: ", pop.members[i].genotype[1])))
    end
    return 1 + l.δ * (l.b * p_not_i - l.c * pop.members[i].genotype[1])
end

# additive payoff function
function addpayoff_rand(i::Int64, pop::Population)
    l = pop.landscape
    d = pop.members[i].deme

    # pick a random partner j in deme d
    drange = pop.d2i[[d,1]]:pop.d2i[[d,pop.N[d]]]
    j = rand(drange)
    while i == j
        j = rand(drange)
    end

    return 1 + l.δ * (l.b * pop.members[j].genotype[1] - l.c * pop.members[i].genotype[1])
end

function wf_selection!(pop::Population)
    nd = length(pop.N)
    NT = sum(pop.N)

    # resize offspring pop if necessary
    if NT != length(pop.offspring)
        resize!(pop.offspring, NT)
    end

    # normalize fitness
    pop.fitness .= pop.fitness ./ dropdims(sum(pop.fitness, dims=2), dims=2)

    # create offspring generation
    for d in 1:nd
        wfsampler = sampler(Categorical(pop.fitness[d,:]))
        for i in 1:pop.N[d]
            i_par = rand(wfsampler)
            copy!(pop.offspring[pop.d2i[[d,i]]], pop.members[i_par])
        end
    end

    # swap offspring and parents for next generation
    pop.members, pop.offspring = pop.offspring, pop.members
end

function group_competition!(pop::Population)
    nd = length(pop.N)
    NT = sum(pop.N)
    irange = [ pop.d2i[[d,1]]:pop.d2i[[d,pop.N[d]]] for d in 1:nd ]

    total_payoff = [sum(pop.payoff[irange[d]]) for d in 1:nd]
    total_total_payoff = sum(total_payoff)
    if total_total_payoff == 0
        throw(ErrorException("total population payoff is zero"))
    end

    groupcomp = sampler(Categorical(total_payoff / total_total_payoff))
    par = rand(groupcomp, nd)

    for d in 1:nd
        @. @views pop.fertility[d, irange[par[d]]] = pop.payoff[irange[par[d]]] / total_payoff[par[d]] * pop.N[par[d]]
    end
end

# island-type migration
function migration!(pop::Population)
    nd = length(pop.N)
    NT = sum(pop.N)
    m = pop.landscape.m

    # total dispersed
    pop.dispersed .= dropdims(sum(m / (nd-1) .* pop.fertility, dims=1), dims=1)

    for d in 1:nd
        # add philopatric component (minus m/(nd-1)
        @. @views pop.fitness[d,:] = (1-m*nd/(nd-1)) * pop.fertility[d,:] + pop.dispersed

    end
end

function mutation!(pop::Population)
    NT = sum(pop.N)
    nloci = length(pop.landscape.μ)

    # mutation for biallelic loci. generate all mutations first
    smplrs = [sampler(Bernoulli(pop.landscape.μ[i])) for i in 1:nloci]
    muts = hcat([rand(s, NT) for s in smplrs]...)
    # then assign to individuals in demes
    for i in 1:NT
        @. pop.members[i].genotype = (pop.members[i].genotype + @view muts[i,:]) % 2
    end
end

function lifecycle_neutral!(pop::Population)
    wf_selection!(pop)
    mutation!(pop)

    pop.generation += 1
end

function evolve_neutral!(pop::Population, nrep::Int64)

    # constant demography, so fertility/migration only needs to be run once
    for d in 1:length(pop.N)
        @. @views pop.fertility[d, pop.d2i[[d,1]]:pop.d2i[[d,pop.N[d]]]] = 1.0
    end
    migration!(pop)

    for n in 1:nrep
        lifecycle_neutral!(pop)
    end

end


# final end statement to close the module
end
