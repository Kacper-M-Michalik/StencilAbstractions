const CPUBackend = KernelAbstractions.CPU

abstract type BoundaryType end
struct DefaultValue{T} <: BoundaryType 
    value::T
end
struct WrapAround <: BoundaryType end
struct Mirror <: BoundaryType end
struct MirrorWithBounce <: BoundaryType end
struct Halo <: BoundaryType end

abstract type Structure end
Base.@kwdef struct CartesianGrid <: Structure 
    halo::Bool = true
end
struct TreeGrid <: Structure end

struct Grid end

abstract type WriteMode end
struct Copy <: WriteMode end
struct Inplace_UNSAFE <: WriteMode end

abstract type ComputationSettings end
Base.@kwdef struct BasicSettings{W <: WriteMode} <: ComputationSettings
    enable_fastmath::Bool = false
    enable_bounds_checks::Bool = false    
    write_mode::W = Copy()
    default_block_size = (32,16)
end

Base.@kwdef struct SpatialSettings{W <: WriteMode} <: ComputationSettings
    enable_fastmath::Bool = false
    enable_bounds_checks::Bool = false
    write_mode::W = Copy()    
    tile_size = (32, 32)
    thread_count = 4    
end

Base.@kwdef struct DiamondSettings <: ComputationSettings
end

Base.@kwdef struct GatherSettings <: ComputationSettings
    enable_fastmath::Bool = false
    enable_bounds_checks::Bool = false    
    tile_size = (32, 32)
    block_size = (256)
end

Base.@kwdef struct OverlappedSettings <: ComputationSettings
    enable_fastmath::Bool = false
    enable_bounds_checks::Bool = false    
    elements_per_thread = (2)
    block_size = (32, 8)
    timesteps::Int64 = 2 
end

Base.@kwdef struct PatternSettings{W <: WriteMode} <: ComputationSettings
    enable_fastmath::Bool = false
    enable_bounds_checks::Bool = false    
    # GPU Only
    block_size = (32, 16)
    # CPU Only
    thread_count = 4    
end

Base.@kwdef mutable struct StencilDimension
    min::Int64 = 0
    max::Int64 = 0
end

struct StencilSource
    symbol::Symbol    
    radii::Vector{StencilDimension}
    offsets::Vector{Vector{Int64}}
end

struct StencilEquation
    name::Symbol
    args::Array{Tuple{Symbol, Union{Symbol, Nothing}}}
    body::Expr
    input_symbols::Array{Symbol} 
    output_symbols::Array{Symbol}   
    first_grid_index::Int64
    sources::Dict{Symbol, StencilSource}   
    dimensions::Int64
    inplace::Bool
    unsafe::Bool
end

struct StencilSpecification{G <: Structure, B <: BoundaryType, D <: KernelAbstractions.Backend, S <: ComputationSettings}
    stencil::StencilEquation
    structure::G
    boundary::B
    device::D
    settings::S
end
StencilSpecification(s, g, b, d) = StencilSpecification(s, g, b, d, BasicSettings())
StencilSpecification(s, g, b, d, nothing) = StencilSpecification(s, g, b, d, BasicSettings())

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

struct TranslationLine
    source::Expr
    dest::Expr
end

struct StencilCollection
    specifications::Array{StencilSpecification}
end
StencilCollection(args...) = StencilCollection(Array{StencilSpecification}(args...))

const _KERNEL_END = Expr(:macrocall, Symbol("@label"), nothing, Symbol("kernel_end"))
const _KERNEL_GOTO = Expr(:macrocall, Symbol("@goto"), nothing, Symbol("kernel_end"))
const _SIZE_SYMBOL = Symbol("_RANGE")
const _UPPER_BOUND = Symbol("_UPPER")
const _LOWER_BOUND = Symbol("_LOWER")
const _THREADS_SYMBOL = Symbol("_THREAD_COUNT")
const _GRID_TYPE = Symbol("Grid")

const inplace_operators = Set{Symbol}([:(+=), :(-=), :(*=), :(/=), :(\=), :(÷=), :(//=), :(^=), :(%=), :(&=), :(|=), :(⊻=), :(<<=), :(>>>=), :(>>=)])