function GetLargestStencilIndices(spec::StencilSpecification)
    offsets = Set{CartesianIndex}()    
    for source in values(spec.stencil.sources)
        for offset in source.offsets
            push!(offsets, CartesianIndex(offset...))
        end
    end
    return collect(offsets)
end

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

function calc_halo_lower_offset(spec::StencilSpecification)
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

function calc_halo_upper_offset(spec::StencilSpecification)
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

function alloc(spec::StencilSpecification{<:Any, <:ArrayBacking, <:Any, <:Any}, type, arr_size)
    @assert length(arr_size) == spec.stencil.dimensions    
    @assert length(spec.boundary) > 0     

    if spec.boundary[1] isa Halo
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
        return device_array, ranges
    else
        device_array = KernelAbstractions.allocate(spec.device, type, arr_size) 
        return device_array, axes(device_array)
    end
end

function alloc(spec::StencilSpecification{<:Any, <:ArrayBacking, <:Any, <:Any}, array::AbstractArray)    
    arr_size = size(array)
    @assert length(arr_size) == spec.stencil.dimensions 
    @assert length(spec.boundary) > 0     

    if spec.boundary[1] isa Halo
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
        ranges = UnitRange.(lower_corner, upper_corner) 

        device_array = KernelAbstractions.zeros(spec.device, eltype(array), true_size)
        dest = view(device_array, ranges...)
        copyto!(dest, array)       
        return device_array, ranges 
    else
        device_array = KernelAbstractions.allocate(spec.device, eltype(array), arr_size)
        copyto!(device_array, array)
        return device_array, axes(device_array)
    end
end

function alloc(spec::StencilSpecification{<:Any, <:ArrayBacking, <:Any, <:OverlappedSettings}, array::AbstractArray)    
    arr_size = size(array)
    @assert length(arr_size) == spec.stencil.dimensions    
    @assert spec.boundary[1] isa Halo    

    shared_size, logical_size, lower_valid, upper_valid, _ = get_tile_properties(spec)

    halos = GetMaximalHaloSizes(spec)
    temp = Vector{Int}(undef, length(arr_size))
    offset = Vector{Int}(undef, length(arr_size))
    for (i, radius) in enumerate(halos)
        temp[i] = cld(arr_size[i], logical_size[i]) * logical_size[i] + shared_size[i] - upper_valid[i] + lower_valid[i] - 1
        offset[i] = -radius.min + lower_valid[i]
    end
    true_size = Tuple(temp)
    lower_corner = Tuple(offset)      
    upper_corner = lower_corner .+ arr_size .- 1     
    ranges = UnitRange.(lower_corner, upper_corner) 

    device_array = KernelAbstractions.zeros(spec.device, eltype(array), true_size)
    dest = view(device_array, ranges...)
    copyto!(dest, array)       
    return device_array, ranges 
end

function Base.zeros(spec::StencilSpecification{<:Any, <:ArrayBacking, <:Any, <:Any}, type, arr_size) 
    @assert length(arr_size) == spec.stencil.dimensions    
    @assert length(spec.boundary) > 0     

    if spec.boundary[1] isa Halo
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
        return device_array, ranges
    else
        device_array = KernelAbstractions.zeros(spec.device, type, arr_size) 
        return device_array, axes(device_array)
    end
end

function Base.zeros(spec::StencilSpecification{<:Any, <:ArrayBacking, <:Any, <:OverlappedSettings}, type, arr_size) 
    @assert length(arr_size) == spec.stencil.dimensions    
    @assert spec.boundary[1] isa Halo  

    shared_size, logical_size, lower_valid, upper_valid, _ = get_tile_properties(spec)

    halos = GetMaximalHaloSizes(spec)
    temp = Vector{Int}(undef, length(arr_size))
    offset = Vector{Int}(undef, length(arr_size))
    for (i, radius) in enumerate(halos)
        temp[i] = cld(arr_size[i], logical_size[i]) * logical_size[i] + shared_size[i] - upper_valid[i] + lower_valid[i] - 1
        offset[i] = -radius.min + lower_valid[i]
    end
    true_size = Tuple(temp)
    lower_corner = Tuple(offset)      
    upper_corner = lower_corner .+ arr_size .- 1     

    ranges = UnitRange.(Tuple(lower_corner), upper_corner)       
    device_array = KernelAbstractions.zeros(spec.device, type, true_size) 
    return device_array, ranges    
end

macro compile_stencil(vars...)
    if (length(vars) == 5 || length(vars) == 6)      
        local equation
        try
            equation = Core.eval(__module__, vars[2])               
        catch
            error("Could not find/retreive stencil definition on $(__source__.line) in $(__source__.file)")
        end          

        if !(equation isa StencilEquation)
            error("Stencil argument ($(vars[2])) does not point to @stencil function")
        end

        name = vars[1]
        structure = Core.eval(__module__, vars[3])
        boundary = Core.eval(__module__, vars[4])
        if !(boundary isa Tuple) boundary = (boundary,) end
        backend = Core.eval(__module__, vars[5])
        settings = nothing            
        if (length(vars) == 6)
            settings = Core.eval(__module__, vars[6])
        end            

        spec = StencilSpecification(name, equation, structure, boundary, backend, settings)
        result = internal_compile_stencil(spec)

        return esc(quote
            $(result.kernel)

            $(result.caller)

            const $name = StencilAbstractions.CompiledStencil{typeof($(result.caller_name)), $(spec.stencil.dimensions), typeof($(spec))}($(result.caller_name), $(spec))
        end)       
    else
        error("Invalid args in @compile_stencil call!")
    end
end

function compile_stencil(mod::Module, spec::StencilSpecification)
    result = internal_compile_stencil(spec)

    callable = Core.eval(mod, quote
        $(result.kernel)

        $(result.caller)

        StencilAbstractions.CompiledStencil{typeof($(result.caller_name)), $(spec.stencil.dimensions), typeof($(spec))}($(result.caller_name), $(spec))
    end)

    return callable
end

macro iter(vars...)
    @assert length(vars) == 3 || length(vars) == 4 "@iter requires a count, backend and do block!"

    func = vars[1]
    count = vars[2]
    backend = Core.eval(__module__, vars[3])
    unroll_count = length(vars) == 4 ? vars[4] : 1

    @assert func.head === :(->)
    @assert length(func.args) == 2    
    for arg in func.args[2].args
        if (arg isa Expr)
            @assert arg.head === :call
        elseif !(arg isa LineNumberNode)
            error("Only stencil calls are allowed in an @iter block!")
        end
    end    
    @assert count isa Expr || count isa Symbol || count isa Int "@iter requries count be an Expr, Variable or Int!"
    @assert unroll_count isa Int

    backend_obj = Core.eval(__module__, backend)
    body = generate_graph_loop(func.args[2], count, backend_obj, unroll_count)

    return body
end