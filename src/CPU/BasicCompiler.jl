function internal_compile_stencil(spec::StencilSpecification{ArrayBacking, B, KernelAbstractions.CPU, BasicSettings{W, T}}) where {B, W, T}

    # Generate hygenic symbols for kernel function name, etc.
    kernel_name = gensym(Symbol(spec.stencil.name, "!"))
    caller_name = gensym(Symbol(kernel_name, "_caller"))   

    # Process args
    stencil_args = Any[]
    for arg in spec.stencil.args       
        if (arg[2] !== nothing)
            push!(stencil_args, Expr(:(::), arg[1], arg[2]))
        else
            push!(stencil_args, arg[1])            
        end
    end
    caller_args = copy(stencil_args)
    kernel_args = copy(stencil_args)

    setup_caller_args!(caller_args)
    push!(kernel_args, _RANGE_SYMBOL)

    # Check/get valid boundaries
    boundaries = get_boundaries(spec)

    # Compile kernel body    
    gather_blocks = Expr[]
    body = spec.stencil.body
    for (i, grid) in enumerate(spec.stencil.ordered_grids)
        if (grid in spec.stencil.input_symbols)
            if (boundaries[i] isa Halo)
                body = transform_read(body, grid, false, nothing, grid, _CURSOR, nothing)
            else
                for offset in spec.stencil.sources[grid].offsets
                    var_name = gensym(Symbol(grid))         
                    push!(gather_blocks, get_boundary_check(boundaries[i], grid, var_name, _CURSOR, offset))
                    body = transform_read(body, grid, true, offset, var_name, nothing, nothing)
                end
            end     
        end  
    end
    for output in spec.stencil.output_symbols
        body = transform_write(body, output, output, _CURSOR, nothing)
    end 
    @assert body isa Expr
    @assert body.head === :block

    if (length(gather_blocks) != 0)
        new_body = Expr(:block)
        for block in gather_blocks
            for node in block.args
                push!(new_body.args, node)
            end
        end
        for node in body.args
            push!(new_body.args, node)
        end
        body = new_body
    end    
    
    # Execution setup
    call_block = Expr(:block)
    if (W === RedBlack)
        test = :((sum(($_CURSOR).I) % 2) == $(_ITER_SYMBOL))
        body = Expr(:if, test, body)      
        
        push!(call_block.args, :($(kernel_name)($(stencil_args...), $(_RANGE_SYMBOL), 0)))
        push!(call_block.args, :($(kernel_name)($(stencil_args...), $(_RANGE_SYMBOL), 1)))

        push!(kernel_args, _ITER_SYMBOL)
    elseif (W === MultiColour)
        test = :($(_MASK_SYMBOL)[$_CURSOR] == $(_ITER_SYMBOL))
        body = Expr(:if, test, body)     

        call_block = quote
            for iter in 1:mask.colours
                $(kernel_name)($(stencil_args...), $(_RANGE_SYMBOL), $(_ITER_SYMBOL), (($(_MASK_SYMBOL))).mask)
            end
        end        

        push!(kernel_args, _ITER_SYMBOL)
        push!(kernel_args, _MASK_SYMBOL)
    else
        push!(call_block.args, :($(kernel_name)($(stencil_args...), $(_RANGE_SYMBOL))))
    end

    #iterator = :($(cursor_symbol) = $(start_index):$(end_index))
    iterator = :($_CURSOR = CartesianIndices($(_RANGE_SYMBOL)))
    loop = Expr(:for, iterator, body)

    # Apply optimisation flags to loop
    if (spec.settings.enable_simd)
        # simd
        loop = Expr(:macrocall, Symbol("@simd"), nothing, loop)
    end    
    if (!spec.settings.enable_bounds_checks)
        # inbounds
        loop = Expr(:macrocall, Symbol("@inbounds"), nothing, loop)
    end    
    if (spec.settings.enable_fastmath)
        # fastmath
        loop = Expr(:macrocall, Symbol("@fastmath"), nothing, loop)
    end    

    generated_kernel = quote        
        function $kernel_name($(kernel_args...))           
            $loop
        end
    end
    # Kernel wrapped in block by default
    generated_kernel = generated_kernel.args[2]
    
    local range_block
    if (boundaries[1] isa Halo)
        lower_offset = GetHaloLowerOffset(spec)
        upper_offset = GetHaloUpperOffset(spec)   
        range_block = quote
            temp = axes($(spec.stencil.ordered_grids[1]))
            $_RANGE_SYMBOL = UnitRange.(first.(temp) .+ $lower_offset, last.(temp) .- $upper_offset)
        end
    else
        range_block = quote
            $_RANGE_SYMBOL = axes($(spec.stencil.ordered_grids[1]))
        end
    end

    kernel_caller = quote
        function $(caller_name)($(caller_args...))        
            if (length($(_RANGE_SYMBOL)) == 0)
                $(range_block.args...)
            end
            $(call_block.args...)
        end
    end

    return CompilationResult(kernel_caller, generated_kernel, caller_name, kernel_name, spec)
end

#     threads = Threads.nthreads()
#     partionperthread = dim/threads
#     @threads for id in 0:(threads-1)
#         lower = base + id * partionperthread
#         upper = min(lower + partionperthread - 1, dimupper)
#         range = range[same, lower:upper]
#         $(loop)
#   $(step) = cld($(_SIZE_SYMBOL)[1], $(_THREADS_SYMBOL))
#             Polyester.@batch for $(id_symbol) = 0:($(_THREADS_SYMBOL)-1)
#                 # Range calculations                    
#                 $(rng_low) =  $(_LOWER_BOUND)[1] + $(step)*$(id_symbol)
#                 $(rng_high) = min($(rng_low)+$(step)-1, $(_UPPER_BOUND)[1])
#                 $(start_index) = CartesianIndex($(rng_low), $(lower_access...))
#                 $(end_index) = CartesianIndex($(rng_high), $(upper_access...))
#                 # Main loop
#                 $(loop)
#             end   
    # Generate iteration domain calculations  
    # lower_access = Any[]
    # upper_access = Any[]
    # for i in 2:spec.stencil.dimensions
    #     push!(lower_access, :($(_LOWER_BOUND)[$(i)]))
    #     push!(upper_access, :($(_UPPER_BOUND)[$(i)]))
    # end  
    # start_index = gensym(Symbol("start"))
    # end_index = gensym(Symbol("end"))        
    # step = gensym(Symbol("step"))
    # rng_low = gensym(Symbol("lower"))
    # rng_high = gensym(Symbol("high"))     

    # if (spec.settings.threading isa JuliaThreads)
    #     handle = Expr(:(.), Symbol("Threads"), QuoteNode(Symbol("@threads")))
    #     loop = Expr(:macrocall, handle, nothing, loop)
    # elseif (spec.settings.threading isa PolyesterThreads)
    #     if (spec.stencil.contains_annotations) error("Stencil cannot contain type annotations when compiling with PolyesterThreads (Polyester throws errors)!") end
    #     handle = Expr(:(.), Symbol("Polyester"), QuoteNode(Symbol("@batch")))
    #     loop = Expr(:macrocall, handle, nothing, loop)
    # end