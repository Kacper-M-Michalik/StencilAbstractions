function GetMaximalHaloSizes(collection::StencilCollection)
    if length(collection.stencils) == 0
        return Array{StencilDimension}()
    end

    dims = collection.specifications[1].stencil.dimensions
    halos = [StencilDimension() for _ in 1:dims]
    for spec in collection.specifications
        @assert spec.stencil.dimensions == dims
        halo = GetMaximalHaloSizes(spec)
        for (i, radius) in halo
            if radius.min < halos[i].min && radius.min < 0
                halos[i].min = radius.min
            end
            if radius.max > halos[i].max && radius.max > 0
                halos[i].max = radius.max
            end
        end
    end

    return halos
end

function GetMaximalHaloSizes(spec::StencilSpecification)
    halos = [StencilDimension() for _ in 1:spec.stencil.dimensions]

    for source in values(spec.stencil.sources)
        for (i, radius) in enumerate(source.radii)
            if radius.min < halos[i].min && radius.min < 0
                halos[i].min = radius.min
            end
            if radius.max > halos[i].max && radius.max > 0
                halos[i].max = radius.max
            end
        end
    end

    return halos
end

function GetHaloLowerOffset(spec::StencilSpecification)
    offset = zeros(Int, spec.stencil.dimensions)

    for source in values(spec.stencil.sources)
        for (i, radius) in enumerate(source.radii)
            if -radius.min > offset[i]
                offset[i] = -radius.min
            end
        end
    end

    return Tuple(offset)
end

function GetHaloUpperOffset(spec::StencilSpecification)
    offset = zeros(Int, spec.stencil.dimensions)

    for source in values(spec.stencil.sources)
        for (i, radius) in enumerate(source.radii)
            if radius.max > offset[i]
                offset[i] = radius.max
            end
        end
    end

    return Tuple(offset)
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

function (stencil::CompiledStencil)(args...; lower=(), upper=(), threads=0, block_size=(), order=nothing)
    stencil.kernel(args..., lower, upper, threads, block_size, order)
end