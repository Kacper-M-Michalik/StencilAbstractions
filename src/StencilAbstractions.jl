module StencilAbstractions

using DataStructures
using KernelAbstractions

export
    @stencil, @Input, @Output, @returns,
    StencilSpecification, StencilEquation, alloc, compile_stencil,
    CPUBackend,
    CartesianGrid, TreeGrid, 
    GridBoundary, DefaultValue, WrapAround, Mirror, MirrorWithBounce, None,
    BasicSettings, SpatialSettings, DiamondSettings, CandySettings, GatherSettings, OverlappedSettings
    
include("Defs.jl")
include("Helpers.jl")
include("Boundaries.jl")
include("Shared/Parser.jl")
include("Shared/KernelCompiler.jl")
include("CPU/BasicCompiler.jl")
include("GPU/BasicCompiler.jl")

macro stencil(equation)      
    if !(equation isa Expr && equation.head === :function)
        error("Invalid stencil declaration!")
    end
    
    stencil_name = equation.args[1].args[1]
    stencil_object = process_stencil_equation(equation)

    return esc(quote    
        $stencil_name = $stencil_object
    end)
end

macro stencil(vars...)
    error("Invalid stencil declaration!")
end

macro Input(expr)
    return esc(expr)
end

macro Output(expr)
    return esc(expr)
end

macro returns(expr)
    return esc(expr)
end

function alloc(spec::StencilSpecification{<:CartesianGrid, <:Any, <:Any}, type, arr_size) 
    @assert length(arr_size) == spec.stencil.dimensions    

    if spec.structure.halo
        halos = GetMaximalHaloSizes(spec)
        temp = Vector{Int}(undef, length(arr_size))
        offset = Vector{Int}(undef, length(arr_size))
        for (i, radius) in enumerate(halos)
            temp[i] = arr_size[i] - radius.min + radius.max
            offset[i] = 1 - radius.min
        end
        true_size = Tuple(temp)
        lower_corner = Tuple(offset)      
        upper_corner = lower_corner .+ arr_size .- 1     
        ranges = UnitRange.(Tuple(lower_corner), upper_corner) 
      
        device_array = KernelAbstractions.allocate(spec.device, type, true_size) 
        data = view(device_array, ranges...)
        return device_array, data
    else
        device_array = KernelAbstractions.allocate(spec.device, type, arr_size) 
        return device_array, view(device_array, axes(device_array)...)
    end
end

function alloc(spec::StencilSpecification{<:CartesianGrid, <:Any, <:Any}, array::AbstractArray)    
    arr_size = size(array)
    @assert length(arr_size) == spec.stencil.dimensions 

    if spec.structure.halo
        halos = GetMaximalHaloSizes(spec)
        temp = Vector{Int}(undef, length(arr_size))
        offset = Vector{Int}(undef, length(arr_size))
        for (i, radius) in enumerate(halos)
            temp[i] = arr_size[i] - radius.min + radius.max
            offset[i] = 1 - radius.min
        end
        true_size = Tuple(temp)
        lower_corner = Tuple(offset)      
        upper_corner = lower_corner .+ arr_size .- 1     
        ranges = UnitRange.(Tuple(lower_corner), upper_corner) 
      
        device_array = KernelAbstractions.zeros(spec.device, eltype(array), true_size)
        dest = view(device_array, ranges...)
        copyto!(dest, array)       
        return device_array, dest 
    else
        device_array = KernelAbstractions.allocate(spec.device, eltype(array), arr_size)
        copyto!(device_array, array)
        return device_array, view(device_array, axes(device_array)...)
    end

    return device_array
end

function Base.zeros(spec::StencilSpecification{<:CartesianGrid, <:Any, <:Any}, type, arr_size) 
    @assert length(arr_size) == spec.stencil.dimensions    
   
    if spec.structure.halo
        halos = GetMaximalHaloSizes(spec)
        temp = Vector{Int}(undef, length(arr_size))
        offset = Vector{Int}(undef, length(arr_size))
        for (i, radius) in enumerate(halos)
            temp[i] = arr_size[i] - radius.min + radius.max
            offset[i] = 1 - radius.min
        end
        true_size = Tuple(temp)
        lower_corner = Tuple(offset)      
        upper_corner = lower_corner .+ arr_size .- 1     
        ranges = UnitRange.(Tuple(lower_corner), upper_corner) 
      
        device_array = KernelAbstractions.zeros(spec.device, type, true_size) 
        data = view(device_array, ranges...)
        return device_array, data
    else
        device_array = KernelAbstractions.zeros(spec.device, type, arr_size) 
        return device_array, view(device_array, axes(device_array)...)
    end
end

function (stencil::CompiledStencil)(args...; lower=(), upper=())
    stencil.kernel(args...; lower = lower, upper = upper)
end

end