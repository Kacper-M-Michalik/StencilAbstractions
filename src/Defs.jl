const CPUBackend = KernelAbstractions.CPU
struct Grid end


abstract type Structure end
struct CartesianGrid <: Structure end


abstract type BoundaryType end
struct Halo <: BoundaryType end
abstract type NonHalo <: BoundaryType end

struct DefaultValue{T} <: NonHalo 
    value::T
end
struct WrapAround <: NonHalo end
struct Mirror <: NonHalo end
struct MirrorWithBounce <: NonHalo end


abstract type ProcessMode end
struct Parallel <: ProcessMode end
struct RedBlack <: ProcessMode end
struct MultiColour <: ProcessMode end


abstract type ThreadingMode end
struct SingleThread <: ThreadingMode end
struct JuliaThreads <: ThreadingMode end
struct PolyesterThreads <: ThreadingMode end


abstract type ColouringAlgorithm end
struct GREEDY <: ColouringAlgorithm end
struct ROSC <: ColouringAlgorithm end

struct ExecutionMask
    colours::Int
    mask
end


abstract type ComputationSettings end
Base.@kwdef struct BasicSettings{W <: ProcessMode, T <: ThreadingMode} <: ComputationSettings
    enable_fastmath::Bool = false
    enable_bounds_checks::Bool = false    
    enable_simd::Bool = false    
    processing::W = Parallel()
    
    threading::T = SingleThread()
    default_block_size = (32,16)
end

Base.@kwdef struct SpatialSettings{W <: ProcessMode, T <: ThreadingMode} <: ComputationSettings
    enable_fastmath::Bool = false
    enable_bounds_checks::Bool = false
    enable_simd::Bool = false    
    processing::W = Parallel()    

    threading::T = SingleThread()  
    enable_shared_mem = false
    stream_last_dim = false
    default_tile_size = (32, 32)
end

Base.@kwdef struct OverlappedSettings <: ComputationSettings
    enable_fastmath::Bool = false
    enable_bounds_checks::Bool = false    

    elements_per_thread = (2)
    block_size = (32, 8)
    timesteps::Int = 2 
end


Base.@kwdef mutable struct StencilDimension
    min::Int = 0
    max::Int = 0
end

struct StencilSource
    symbol::Symbol    
    radii::Vector{StencilDimension}
    offsets::Vector{Vector{Int64}}
end

struct StencilEquation
    name::Symbol
    args::Array{Tuple{Symbol, Union{Symbol, Nothing}}}
    ordered_grids::Array{Symbol}
    input_symbols::Array{Symbol} 
    output_symbols::Array{Symbol}   
    body::Expr
    sources::Dict{Symbol, StencilSource}   
    dimensions::Int64
    inplace::Bool
    unsafe::Bool
    contains_annotations::Bool
end

struct StencilSpecification{G <: Structure, B <: Tuple{Vararg{BoundaryType}}, D <: KernelAbstractions.Backend, S <: ComputationSettings}
    stencil::StencilEquation
    structure::G
    boundary::B
    device::D
    settings::S
end
# StencilSpecification(s::StencilEquation, g, b, d) = StencilSpecification(s, g, b, d, BasicSettings())
# StencilSpecification(s::StencilEquation, g, b, d, nothing) = StencilSpecification(s, g, b, d, BasicSettings())
StencilSpecification(s::Symbol, g, b, d) = StencilSpecification(get_stencil(Val(s)), g, b, d, BasicSettings())
StencilSpecification(s::Symbol, g, b, d, nothing) = StencilSpecification(get_stencil(Val(s)), g, b, d, BasicSettings())
StencilSpecification(s::Symbol, g, b, d, set) = StencilSpecification(get_stencil(Val(s)), g, b, d, set)
StencilSpecification(s::Symbol, g, b::BoundaryType, d) = StencilSpecification(get_stencil(Val(s)), g, (b,), d, BasicSettings())
StencilSpecification(s::Symbol, g, b::BoundaryType, d, nothing) = StencilSpecification(get_stencil(Val(s)), g, (b,), d, BasicSettings())
StencilSpecification(s::Symbol, g, b::BoundaryType, d, set) = StencilSpecification(get_stencil(Val(s)), g, (b,), d, set)

struct CompilationResult
    caller::Expr
    kernel::Expr    
    caller_name::Symbol
    kernel_name::Symbol
    settings::StencilSpecification
end

struct CompiledStencil
    kernel
    settings::StencilSpecification
end

struct StencilCollection
    specifications::Array{StencilSpecification}
end
StencilCollection(args...) = StencilCollection(Array{StencilSpecification}(args...))

# TODO CLEAN UP
const _KERNEL_END = Expr(:macrocall, Symbol("@label"), nothing, Symbol("kernel_end"))
const _KERNEL_GOTO = Expr(:macrocall, Symbol("@goto"), nothing, Symbol("kernel_end"))

const _SIZE_SYMBOL = Symbol("_RANGE")
const _UPPER_BOUND = Symbol("_UPPER")
const _LOWER_BOUND = Symbol("_LOWER")
const _GRID_TYPE = Symbol("Grid")
const _ITER_SYMBOL = Symbol("_ITER")

const _RANGE_SYMBOL = Symbol("_RANGE")
const _BLOCK_SYMBOL = Symbol("_BLOCK")
const _MASK_SYMBOL = Symbol("_MASK")

const inplace_operators = Set{Symbol}([:(+=), :(-=), :(*=), :(/=), :(\=), :(÷=), :(//=), :(^=), :(%=), :(&=), :(|=), :(⊻=), :(<<=), :(>>>=), :(>>=)])
const blocked_keywords = Set{Symbol}([:baremodule, :begin, :catch, :const, :do, :export, :finally, :function, :global, :import, :let, :macro, :module, :quote, :return, :struct, :try, :using])