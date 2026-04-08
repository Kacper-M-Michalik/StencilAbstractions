module StencilAbstractions

using DataStructures
using KernelAbstractions
using Polyester
using StaticArrays
using RuntimeGeneratedFunctions

export
    @stencil, @Input, @Output, @compile_stencil, 
    CompiledStencil, StencilSpecification, StencilEquation, alloc, compile_stencil,
    CPUBackend,
    CartesianGrid, TreeGrid, 
    GridBoundary, DefaultValue, WrapAround, Mirror, MirrorWithBounce, Halo,
    BasicSettings, SpatialSettings, DiamondSettings, CandySettings, GatherSettings, OverlappedSettings,
    Copy, Inplace_UNSAFE
    
include("Defs.jl")
include("Interface.jl")
include("Boundaries.jl")
include("Shared/Parser.jl")
include("Shared/KernelCompiler.jl")
include("CPU/BasicCompiler.jl")
include("CPU/SpatialCompiler.jl")
# include("CPU/SpatialInplaceCompiler.jl")
include("GPU/BasicCompiler.jl")
include("Macros.jl")

end