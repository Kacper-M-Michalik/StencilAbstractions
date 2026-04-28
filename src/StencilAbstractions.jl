module StencilAbstractions

using DataStructures
using KernelAbstractions
using StaticArrays

export
    @stencil, @compile_stencil, compile_stencil, alloc, generate_mask, calc_halo_upper_offset, calc_halo_lower_offset,
    @iter,
    @box, @cross,
    CompiledStencil, StencilSpecification, StencilEquation,     
    ArrayBacking, PatchTree, 
    GridBoundary, Halo, NonHalo, DefaultValue, WrapAround, Mirror,
    BasicSettings, SpatialSettings, OverlappedSettings,
    Parallel, RedBlack, MultiColour, ExecutionMask, GREEDY, ROSC,
    SingleThread, JuliaThreads, PolyesterThreads,
    CPUBackend

include("Defs.jl")

include("Boundaries.jl")
#include("SettingDefaults.jl")
include("Mask/Generation.jl")
include("Graphs/CPU.jl")
include("Interface.jl")

include("Shared/Analyser.jl")
include("Shared/KernelCompiler.jl")
include("CPU/BasicCompiler.jl")
include("CPU/SpatialCompiler.jl")
include("GPU/BasicCompiler.jl")
include("GPU/SpatialCompiler.jl")
#include("GPU/OverlappedCompiler.jl")

include("EDSLMacros.jl")

end