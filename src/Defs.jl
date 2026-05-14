const CPUBackend = KernelAbstractions.CPU


abstract type Structure end
struct ArrayBacking <: Structure end
struct PatchTree <: Structure end


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

struct ExecutionMask{N}
    colours::Int
    mask::Array{UInt8, N}
end


abstract type ComputationSettings end
Base.@kwdef struct BasicSettings{N, W <: ProcessMode, T <: ThreadingMode} <: ComputationSettings
    enable_fastmath::Bool = false
    enable_bounds_checks::Bool = false    
    enable_simd::Bool = false    
    processing::W = Parallel()
    
    threading::T = SingleThread()
    default_block_size::NTuple{N, Int} = (32, 8)
end

Base.@kwdef struct SpatialSettings{N, M, W <: ProcessMode, T <: ThreadingMode} <: ComputationSettings
    enable_fastmath::Bool = false
    enable_bounds_checks::Bool = false
    enable_simd::Bool = false    
    processing::W = Parallel()    

    threading::T = SingleThread()  
    default_tile_size::NTuple{N, Int} = (32, 32)        
    
    enable_shared_mem::Bool = false
    default_block_size::NTuple{M, Int} = (256,)
end

Base.@kwdef struct OverlappedSettings{N} <: ComputationSettings
    swap::Vector{Tuple{Symbol, Symbol}}

    enable_fastmath::Bool = false
    enable_bounds_checks::Bool = false    

    elements_per_thread::Int = 1
    block_size::NTuple{N, Int} = ()
    timesteps::Int = 2
end


Base.@kwdef mutable struct StencilDimension
    min::Int = 0
    max::Int = 0
end

struct StencilSource
    symbol::Symbol    
    radii::Vector{StencilDimension}
    offsets::Vector{Vector{Int}}
end

struct StencilEquation
    name::Symbol
    args::Vector{Tuple{Symbol, Union{Expr, Symbol, Nothing}}}
    ordered_grids::Vector{Symbol}
    input_symbols::Vector{Symbol} 
    output_symbols::Vector{Symbol}   
    body::Expr
    sources::Dict{Symbol, StencilSource}   
    dimensions::Int
    inplace::Bool
    unsafe::Bool
    contains_annotations::Bool
end

struct StencilSpecification{B, G <: Structure, D <: KernelAbstractions.Backend, S <: ComputationSettings}
    name::Symbol
    stencil::StencilEquation
    structure::G
    boundary::NTuple{B, BoundaryType}
    device::D
    settings::S
end
StencilSpecification(s::StencilEquation, g, b, d) = StencilSpecification(gensym(), s, g, b, d, BasicSettings())

struct CompilationResult
    caller::Expr
    kernel::Expr    
    caller_name::Symbol
    kernel_name::Symbol
    settings::StencilSpecification
end

struct CompiledStencil{K, N, Specs <: StencilSpecification}
    kernel::K
    settings::Specs
end

@inline function (stencil::CompiledStencil{K, N})(@specialize(args...);
    range::Union{NTuple{N, UnitRange{Int}}, Nothing} = nothing, 
    tile_size::Union{NTuple{N, Int}, Nothing} = nothing,
    block_size = nothing, 
    mask::Union{ExecutionMask, Nothing} = nothing
    ) where {K, N}
    stencil.kernel(args..., range, tile_size, block_size, mask)
end


struct StencilCollection
    specifications::Array{StencilSpecification}
end
StencilCollection(args...) = StencilCollection(Array{StencilSpecification}(args...))

const _GRID_TYPE = Symbol("Grid")

const _CURSOR = gensym(Symbol("_CURSOR"))
const _UPPER_BOUND = gensym(Symbol("_UPPER"))
const _LOWER_BOUND = gensym(Symbol("_LOWER"))
const _LAST_DIM = gensym(Symbol("_LAST_DIM"))
const _ITER_SYMBOL = gensym(Symbol("_ITER"))

const _RANGE_SYMBOL = gensym(Symbol("_RANGE"))
const _TILE_SYMBOL = gensym(Symbol("_TILE"))
const _BLOCK_SYMBOL = gensym(Symbol("_BLOCK"))
const _MASK_SYMBOL = gensym(Symbol("_MASK"))

const inplace_operators = Set{Symbol}([:(+=), :(-=), :(*=), :(/=), :(\=), :(÷=), :(//=), :(^=), :(%=), :(&=), :(|=), :(⊻=), :(<<=), :(>>>=), :(>>=)])
const blocked_keywords = Set{Symbol}([:baremodule, :begin, :catch, :const, :do, :export, :finally, :function, :global, :import, :let, :macro, :module, :quote, :return, :struct, :try, :using])