# Calculate FST for biallelic locus
function FST(pop::Population, locus)
    N = pop.N
    nd = length(N)
    NT = sum(N)
    irange = [ pop.d2i[[d,1]]:pop.d2i[[d,N[d]]] for d in 1:nd ]

    pm = zeros(nd)
    for d in 1:nd
        pm[d] = mean([pop.members[i].genotype[locus] for i in irange[d]])
    end
    p = mean(pm)

    function Gwf(pm::Float64, n::Int64)
        return (pm*(n*pm - 1) + (1-pm)*(n*(1-pm) - 1)) / (n-1)
    end

    # This is FST calculated by homozygosity within (w) and between (b) populations
    # using sampling w/o replacement
    Gw = sum([N[d] / NT * Gwf(pm[d], N[d]) for d in 1:nd])
    Gb = sum([N[d]/(NT-N[d]) * ( pm[d]*(p - N[d]/NT*pm[d]) + (1-pm[d])*(1-p - N[d]/NT*(1-pm[d]))) for d in 1:nd])

    return ( Gw - Gb ) / ( 1 - Gb)
end
