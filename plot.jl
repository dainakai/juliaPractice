using Plots
using Distributions
using StatsPlots

# d = Normal(50,15)
# plot(d)
# g = Gamma(2,50/2)
# plot(g)

using DataFrames
using CSV

data = CSV.read("Results.csv",DataFrame)

ferets = data[:,2]
histogram(ferets)
# Plots.gr()
g = Gamma(2,48/2)
plot!(twinx(),g)