module StencilAbstractions

using DataStructures
using KernelAbstractions
using StaticArrays

export
    @stencil, @compile_stencil, compile_stencil, alloc, zeros, generate_mask, calc_halo_upper_offset, calc_halo_lower_offset,
    @iter,
    @box, @cross, @cursor,
    CompiledStencil, StencilSpecification, StencilEquation,     
    ArrayBacking, 
    Halo, DefaultValue, WrapAround, Mirror,
    BasicSettings, SpatialSettings, OverlappedSettings,
    Parallel, RedBlack, MultiColour, ExecutionMask, GREEDY,
    SingleThread, JuliaThreads, PolyesterThreads,
    CPUBackend

include("Defs.jl")

include("Shared/Boundaries.jl")
include("Mask/Generation.jl")
include("Graphs/CPU.jl")
include("Interface.jl")

include("Shared/Analyser.jl")
include("Shared/KernelCompiler.jl")
include("CPU/BasicCompiler.jl")
include("CPU/SpatialCompiler.jl")
include("GPU/BasicCompiler.jl")
include("GPU/SpatialCompiler.jl")
include("GPU/OverlappedCompiler.jl")

include("EDSLMacros.jl")

end