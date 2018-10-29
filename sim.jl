using Random
using Statistics
using Distributions
using Revise
using Profile
using DelimitedFiles
using Plots
pyplot()

push!(LOAD_PATH, ".")
using metapop
include("stats.jl")

function lifecycle!(pop::Population)
    pop.fertility .= 0.0
    pop.fitness .= 0.0
    payoff!(pop)
    group_competition!(pop)
    migration!(pop)
    wf_selection!(pop)
    pop.generation += 1
end

function mean_genotype(pop::Population, locus::Int64)
    mean(hcat([pop.members[i].genotype[locus] for i in 1:sum(p.N)]...))
end

function evolve_fix(pop::Population, ngen::Int64, nrep::Int64)
    result = []
    extinct = []
    for n in 1:nrep
        p = deepcopy(pop)
        freq = zeros(ngen+1)
        freq[1] = mean_genotype(p,1)

        g = 0
        not_extinct = true
        while g < ngen && not_extinct
            g += 1
            lifecycle!(p)
            freq[g+1] = mean_genotype(p, 1)
            if freq[g+1] == 0.0 || freq[g+1] == 1.0
                push!(extinct, g)
                not_extinct = false
            end
        end

        if freq[g+1] > 0.0
            freq[g+2:end] .= freq[g+1]
            push!(result, freq)
        end
    end

    return result, extinct
end

##
Random.seed!(21)
p = Population(5*ones(Int64, 500), Landscape(300., 1.0, 0.1, [0.1], 0.1, addpayoff),
               setindex!(zeros(Int64, 2500, 1), ones(Int64, 1), 1:1))
show([addpayoff(i, p) for i in 1:10])
[lifecycle!(p) for i in 1:100]
println(p.mean_allele)
println([p.members[i].genotype[1] for i in p.d2i[[1,1]]:p.d2i[[1,p.N[1]]]])
minimum(p.payoff)

###
Random.seed!(21)
p = Population(5*ones(Int64, 200), Landscape(3.0, 1.0, 0.1, [0.1], 0.1, addpayoff),
               setindex!(zeros(Int64, 1000, 1), ones(Int64, 1), 1:1))

@time r, e = evolve_fix(p, 500, 1000)
println("num non-extinctions: ", length(r), ". ", [el[end] for el in r])
println([minimum(e), mean(e), maximum(e)])
plot(r)


open("genos.csv", "w") do io
    writedlm(io, hcat(r...), ',')
end


plot(hcat(((mat)->[[quantile(mat[i,:], 0.25), median(mat[i,:]), mean(mat[i,:]), quantile(mat[i,:], 0.75)]
        for i in 1:size(mat)[1]])(hcat(r...))...)')



function test_evolve!(p::Population, ngen::Int64)
    for i in 1:ngen
        lifecycle!(p)
    end
end
Profile.init(delay=0.1)

@time test_evolve!(p, 100)

@profile test_evolve!(p, 20)
Profile.print()
ProfileView.view()


p = Population(10*ones(Int64, 100), Landscape(0., 0.0, 0.0, [0.1], 0.1, (i,p)->1.0),
               setindex!(zeros(Int64, 1000, 1), ones(Int64, 1), 1:1))

### simulatie neutral population and caluclate FST
fs = zeros(9,50)
ms = 0.1:0.1:0.9
for m in 1:length(ms)
    p = Population(10*ones(Int64, 100), Landscape(0., 0.0, 0.0, [0.1], ms[m], (i,p)->1.0),
                   setindex!(zeros(Int64, 1000, 1), ones(Int64, 10), 1:10))
    for i in 1:50
        evolve_neutral!(p, 10)
        fs[m,i] = FST(p,1)
    end
end

println("mean FST: ", [mean(fs[i,(!isnan).(fs[i,:])]) for i in 1:9])

function group_comp_neutral!(pop::Population, ngen::Int64)
    function cycle()
        pop.payoff .= 1.0
        group_competition!(pop)
        migration!(pop)
        wf_selection!(pop)
        pop.fertility .= 0.0
        pop.fitness .= 0.0
        pop.generation += 1
    end

    for g in 1:ngen
        cycle()
    end
end

p = Population(5*ones(Int64, 200), Landscape(3., 1.0, 0.01, [0.1], 0.1, addpayoff),
               setindex!(zeros(Int64, 1000, 1), ones(Int64, 500), 1:500))

group_comp_neutral!(p, 10)
FST(p, 1)
println("--")
[println([p.members[p.d2i[[d,i]]].genotype[1] for i in 1:p.N[d]]) for d in 1:length(p.N)]


### Output genotype data to CSV to analyze in R
using DelimitedFiles
open("genos.csv", "w") do io
    writedlm(io, [["deme" "genotype"]; [[p.i2d[i][1] for i in 1:sum(p.N)] [p.members[i].genotype[1] for i in 1:sum(p.N)]]], ',')
end

[["deme" "genotype"]; [[p.i2d[i][1] for i in 1:sum(p.N)] [p.members[i].genotype[1] for i in 1:sum(p.N)]]]
