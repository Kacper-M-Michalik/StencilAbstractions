function max_time(source, block_size)
    max_timesteps = typemax(Int) 
    for (i, radius) in enumerate(source.radii)
        bounds = -radius.min + radius.max
        
        possible = 0
        remaining = block_size[i]
        while (remaining > 0)   
            possible += 1
            remaining -= bounds
        end
        max_timesteps = min(max_timesteps, possible)
    end
    
    return max_timesteps
end

function get_tile_properties(spec)
    
    block_size = spec.settings.block_size
    eff_block_size = block_size .* spec.settings.elements_per_thread
    timesteps = spec.settings.timesteps
    #println(spec.stencil.dimensions)
    bounds = [StencilDimension(0,0) for _ in 1:spec.stencil.dimensions]
    reductions =  [StencilDimension(0,0) for _ in 1:spec.stencil.dimensions]
    for (name, source) in spec.stencil.sources
        for (i, radius) in enumerate(source.radii)
            bounds[i].min = max(bounds[i].min, -radius.min)
            bounds[i].max = max(bounds[i].max, radius.max)

            left_reduction = -radius.min * (timesteps - 1)
            right_reduction = radius.max * (timesteps - 1)
            reductions[i].min = max(reductions[i].min, left_reduction)
            reductions[i].max = max(reductions[i].max, right_reduction)
        end
    end

    bounding_size = [0 for _ in 1:spec.stencil.dimensions]
    logical_size = [0 for _ in 1:spec.stencil.dimensions]
    lower_bound = [0 for _ in 1:spec.stencil.dimensions]
    upper_bound = [0 for _ in 1:spec.stencil.dimensions]
    cursor_offset = [0 for _ in 1:spec.stencil.dimensions]
    for i in 1:spec.stencil.dimensions
        b = bounds[i]
        r = reductions[i]
        bounding_size[i] = eff_block_size[i] + b.min + b.max
        logical_size[i] = eff_block_size[i] - r.min - r.max
        lower_bound[i] = r.min + 1
        upper_bound[i] = eff_block_size[i] - r.max
        cursor_offset[i] = b.min
    end
    
    return Tuple(bounding_size), Tuple(logical_size), Tuple(lower_bound), Tuple(upper_bound), Tuple(cursor_offset)
end

function internal_compile_stencil(spec::StencilSpecification{B, ArrayBacking, D, S}) where {B, D <: KernelAbstractions.GPU, S <: OverlappedSettings}
    @assert length(spec.settings.swap) > 0 "Must specify at least one array swap in an overlapped stencil!"
    @assert !(spec.stencil.inplace) "Inplace stencils not supported by the Overlapped Compiler, use only Jacobi-like stencils!"
    @assert spec.boundary[1] isa Halo "Only Halo boundaries are supported by Overlapped Compiler!"
    @assert length(spec.stencil.input_symbols) > 0 "Stencils must have at least one input field for the Overlapped Compiler!"
    @assert spec.settings.elements_per_thread == 1 "Compiler does not current support elements_per_thread values other than 1!"

    # Generate hygenic symbols for kernel function name, etc.
    kernel_name = gensym(Symbol(spec.stencil.name))
    caller_name = Symbol(kernel_name, "_caller")    

    # Process args   
    caller_args = Any[]
    stencil_args = Any[]
    kernel_args = Any[]
    for arg in spec.stencil.args       
        if (arg[1] in spec.stencil.ordered_grids) 
            type = arg[2]
            if (type === nothing)                
                type = :(<:Any)
            end
            push!(caller_args, Expr(:(::), arg[1], :(AbstractArray{$type, $(spec.stencil.dimensions)})))      

            if (in(arg[1], spec.stencil.input_symbols) && !in(arg[1], spec.stencil.output_symbols))      
                push!(kernel_args, Expr(:macrocall, Symbol("@Const"), nothing, arg[1]))
            else               
                push!(kernel_args, arg[1])
            end        
        else
            if (arg[2] !== nothing)
                push!(caller_args, Expr(:(::), arg[1], arg[2]))
                push!(kernel_args, Expr(:(::), arg[1], arg[2]))
            else
                push!(caller_args, arg[1])            
                push!(kernel_args, arg[1])            
            end
        end
        push!(stencil_args, arg[1])
    end

    setup_caller_args!(spec, caller_args)    
    push!(kernel_args, :($(_LOWER_BOUND)::NTuple{$(spec.stencil.dimensions), Int}))
    push!(kernel_args, :($(_UPPER_BOUND)::NTuple{$(spec.stencil.dimensions), Int}))

    block_size = spec.settings.block_size
    eff_block_size = spec.settings.block_size .* spec.settings.elements_per_thread
    timesteps = spec.settings.timesteps

    lin_idx = gensym("lin_idx")
    block_idx = gensym("block_idx")
    stride = prod(block_size)
    time_var = gensym("t")

    shmem_idx = gensym("lin_load_idx")
    shmem_idx_cart = gensym("cart_load_idx")

    max_timesteps = typemax(Int)   
    field_to_shmem = Dict{Symbol, Symbol}()
    shmems = Expr(:block)
    loads = Expr(:block)

    for (name, source) in spec.stencil.sources
        possible_timesteps = max_time(source, eff_block_size)
        max_timesteps = min(max_timesteps, possible_timesteps)
        #push!(shared_sizes, shared_size)
    end
    if (timesteps > max_timesteps)
        error("Overlapped settings block size does not allow for this many skewed time steps")
    end

    shared_size, logical_size, lower_valid, upper_valid, cursor_offset = get_tile_properties(spec)    #collect(values(spec.stencil.sources)), shared_sizes, timesteps)
    cart_size = CartesianIndices(shared_size)
    total_elems = prod(shared_size)

    # Generate shared memory tiles, write registers, and coalesced load loops
    for (name, source) in spec.stencil.sources
        # TODO elem count        
        shmem_name = gensym(name)
        field_to_shmem[name] = shmem_name
        
        push!(shmems.args, :($shmem_name = KernelAbstractions.@localmem eltype($name) $shared_size))
        push!(loads.args, 
            :(for $shmem_idx = ($lin_idx):($stride):($total_elems)
                $shmem_idx_cart = ($cart_size)[$shmem_idx]
                ($shmem_name)[$shmem_idx_cart] = ($name)[$block_idx + $shmem_idx_cart]
            end)
        )
    end

    swap_mapping = Dict{Symbol, Symbol}()
    for (lhs, rhs) in spec.settings.swap
        swap_mapping[lhs] = rhs
        swap_mapping[rhs] = lhs
    end

    body = spec.stencil.body
    for (i, grid) in enumerate(spec.stencil.input_symbols)        
        body = transform_read(body, grid, false, nothing, field_to_shmem[grid], _CURSOR, nothing)             
    end

    shmem_writes = Expr(:block)
    register_writes = Expr(:block)
    body_commits = Expr(:block)
    out_to_var = Dict{Symbol, Symbol}()
    var_inits = Expr(:block)
    
    global_loc = gensym("global_cursor")

    for name in spec.stencil.output_symbols 
        if haskey(swap_mapping, name)            
            in_name = swap_mapping[name]
            shmem_name = field_to_shmem[in_name]

            push!(shmem_writes.args, :(for $shmem_idx = ($lin_idx):($stride):($total_elems)
                $shmem_idx_cart = ($cart_size)[$shmem_idx]
                $global_loc = $block_idx + $shmem_idx_cart
                if (all($lower_valid .<= ($shmem_idx_cart).I) && all(($shmem_idx_cart).I .<= $upper_valid) && all($_LOWER_BOUND .<= ($global_loc).I) && all(($global_loc).I .<= $_UPPER_BOUND))
                    ($name)[$global_loc] = ($shmem_name)[$shmem_idx_cart]
                end
            end))
            
            out_var = gensym(name)
            #println("shawr:", name, "    ($out_var)")
            push!(body_commits.args, :(($shmem_name)[$_CURSOR] = $out_var))
            push!(var_inits.args, :($out_var = ($shmem_name)[$_CURSOR]))

            body = transform_write(body, name, out_var, nothing, nothing)
        else
            out_var = gensym(name)
            out_to_var[name] = out_var 
            #println("regwr:", name, "    ($out_var)")
            push!(register_writes.args, :(($name)[$global_loc] = $out_var))

            body = transform_write(body, name, out_var, nothing, nothing)
        end
    end
    @assert body isa Expr
    @assert body.head === :block

    in_range = gensym("in_range")

    reg_write = []
    if (length(register_writes.args) > 0)
        push!(reg_write, 
            :(if (all($lower_valid .<= ($_CURSOR).I) && all(($_CURSOR).I .<= $upper_valid) && $in_range)
                $(register_writes.args...)
            end)
        )
    end

    final_kernel_body = quote
        $block_idx = CartesianIndex((@index(Group, NTuple) .- 1) .* $logical_size)
        $_CURSOR = @index(Local, Cartesian) + CartesianIndex($cursor_offset)
        $lin_idx = @index(Local, Linear)

        $(shmems.args...)

        $(loads.args...)
        
        KernelAbstractions.@synchronize
        $(var_inits.args...)

        $global_loc = $block_idx + $_CURSOR
        $in_range = (all($_LOWER_BOUND .<= ($global_loc).I) && all(($global_loc).I .<= $_UPPER_BOUND))

        for $time_var in 1:$timesteps
            if ($in_range)
                $(body.args...)
            end            
            KernelAbstractions.@synchronize
            if ($in_range)
                $(body_commits.args...)
            end
            KernelAbstractions.@synchronize
        end

        $(reg_write...)

        $(shmem_writes.args...)
    end
    
    if (!spec.settings.enable_bounds_checks)
        final_kernel_body = Expr(:macrocall, Symbol("@inbounds"), nothing, final_kernel_body)
    end    
    if (spec.settings.enable_fastmath)
        final_kernel_body = Expr(:macrocall, Symbol("@fastmath"), nothing, final_kernel_body)
    end    

    kernel = quote
        @kernel unsafe_indices=true cpu=false function $(kernel_name)($(kernel_args...))   
            $(final_kernel_body.args...)
        end
    end
    kernel = kernel.args[2]

    kernel_caller = quote
        function $(caller_name)($(caller_args...))     
            $_LOWER_BOUND = first.($_RANGE_SYMBOL)
            $_UPPER_BOUND = last.($_RANGE_SYMBOL)          
            $_RANGE_SYMBOL = cld.($_UPPER_BOUND .- $_LOWER_BOUND .+ 1, $logical_size) .* $block_size

            $(kernel_name)($(spec.device), $block_size)($(stencil_args...), $(_LOWER_BOUND), $(_UPPER_BOUND); ndrange=$_RANGE_SYMBOL)
        end
    end

    #println("Overlapped Calc")
    #println(shared_sizes)
    #println(max_timesteps)
    #println(kernel)   

    return CompilationResult(kernel_caller, kernel, caller_name, kernel_name, spec)
end