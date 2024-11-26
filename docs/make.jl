using LowRankLayers
using Documenter

DocMeta.setdocmeta!(LowRankLayers, :DocTestSetup, :(using LowRankLayers); recursive=true)

makedocs(;
    modules=[LowRankLayers],
    authors="murrellb <murrellb@gmail.com> and contributors",
    sitename="LowRankLayers.jl",
    format=Documenter.HTML(;
        canonical="https://MurrellGroup.github.io/LowRankLayers.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MurrellGroup/LowRankLayers.jl",
    devbranch="main",
)
