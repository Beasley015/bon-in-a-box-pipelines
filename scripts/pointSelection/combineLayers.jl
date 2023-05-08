using BiodiversityObservationNetworks
using SpeciesDistributionToolkit
using JSON
using Downloads
using GeoArrays: read 
using DataFrames
using CSV

# Read in input arguments and json
outputFolder = ARGS[1]
print(outputFolder)
filepath = joinpath(outputFolder,"input.json")
outputFilepath = joinpath(outputFolder,"data/")
isdir(outputFilepath) || mkdir(outputFilepath)

input = JSON.parsefile(filepath)

layermat = CSV.read(IOBuffer(input["layerdata"][1]), DataFrame)
println(layermat)

targetbalance = convert(Vector{Float64}, input["targetbalance"]) # this is the same as α
layer_col = layermat[:, :layer]
keep_rows = startswith.(layermat[:, :layer], "https:")

temppath = Downloads.download.(layermat[keep_rows, 1])

l = SimpleSDMPredictor.(temppath)

layers = BiodiversityObservationNetworks.stack(l)

# get weights with weights for a layer in columns 
W = Matrix(layermat[keep_rows, 2:end])

#priority = SimpleSDMPredictor(squish(layers, convert(Array{Float64}, transpose(W)), targetbalance))
priority = SimpleSDMPredictor(squish(layers, W, targetbalance))

priority_path = joinpath(outputFilepath, "priority_map.tiff")
###################
println(priority_path)
# write out the priority map
SpeciesDistributionToolkit.save(priority_path, priority)

println("pre_json save")

# write out json
outputDict = Dict("priority_map" => priority_path)
open(joinpath(outputFolder, "output.json"),"w") do f
    JSON.print(f, outputDict) 
end 
