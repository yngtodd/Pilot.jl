using Documenter, Pilot

makedocs(;
    modules=[Pilot],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/yngtodd/Pilot.jl/blob/{commit}{path}#L{line}",
    sitename="Pilot.jl",
    authors="yngtodd",
#    assets=String[],
)

deploydocs(;
    repo="github.com/yngtodd/Pilot.jl",
)
