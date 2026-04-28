function internal_compile_stencil(spec::StencilSpecification{B, ArrayBacking, KernelAbstractions.CPU, S}) where {B, S <: BasicSettings}

    # Generate hygenic symbols for kernel function name, etc.
    kernel_name = gensym(Symbol(spec.stencil.name, "!"))
    caller_name = gensym(Symbol(kernel_name, "_caller"))   

    # Process args
    tuple_args = Any[]

    caller_args = Any[]
    stencil_args = Any[]
    kernel_args = Any[]
    for arg in spec.stencil.args       
        if (arg[1] in spec.stencil.ordered_grids)
            type = arg[2]
            if (type === nothing)                
                type = :(<:Any)
            end
            push!(caller_args, Expr(:(::), arg[1], :(Array{$type, $(spec.stencil.dimensions)})))    
            push!(kernel_args, Expr(:(::), arg[1], :(Array{$type, $(spec.stencil.dimensions)})))       
            push!(tuple_args, :(Array{$type, $(spec.stencil.dimensions)}))
        else
            if (arg[2] !== nothing)
                push!(caller_args, Expr(:(::), arg[1], arg[2]))
                push!(kernel_args, Expr(:(::), arg[1], arg[2]))
                push!(tuple_args, arg[2])
            else
                push!(caller_args, arg[1])            
                push!(kernel_args, arg[1])            
                push!(tuple_args, :(<:Any))
            end
        end
        push!(stencil_args, arg[1])
    end

    # push!(tuple_args, :(NTuple{$(spec.stencil.dimensions), UnitRange{Int}}))
    # push!(tuple_args, :(NTuple{$(spec.stencil.dimensions), Int}))
    # push!(tuple_args, :(Union{ExecutionMask, Nothing}))
    setup_caller_args!(spec, caller_args)
    push!(kernel_args, :($(_RANGE_SYMBOL)::NTuple{$(spec.stencil.dimensions), UnitRange{Int}}))

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
    if (spec.settings.processing isa RedBlack)
        test = :((sum(($_CURSOR).I) % 2) == $(_ITER_SYMBOL))
        body = Expr(:block, Expr(:if, test, body))
        
        push!(call_block.args, :($(kernel_name)($(stencil_args...), $(_RANGE_SYMBOL), 0)))
        push!(call_block.args, :($(kernel_name)($(stencil_args...), $(_RANGE_SYMBOL), 1)))

        push!(kernel_args, _ITER_SYMBOL)
    elseif (spec.settings.processing isa MultiColour)
        test = :($(_MASK_SYMBOL)[$_CURSOR] == $(_ITER_SYMBOL))
        body = Expr(:block, Expr(:if, test, body))     

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

    # Apply optimisation flags to loop
    # if (spec.settings.enable_simd)
    #     # simd
    #     body = Expr(:macrocall, Symbol("@simd"), nothing, body)
    # end    
    if (!spec.settings.enable_bounds_checks)
        # inbounds
        body = Expr(:macrocall, Symbol("@inbounds"), nothing, body)
    end    
    if (spec.settings.enable_fastmath)
        # fastmath
        body = Expr(:macrocall, Symbol("@fastmath"), nothing, body)
    end    
    
    iterator = :($_CURSOR = CartesianIndices($(_RANGE_SYMBOL)))    
    loop = Expr(:for, iterator, Expr(:block, body))

    if (spec.settings.threading isa JuliaThreads)
        handle = Expr(:(.), Symbol("Threads"), QuoteNode(Symbol("@threads")))
        loop = Expr(:macrocall, handle, nothing, QuoteNode(:(static)), loop)
    elseif (spec.settings.threading isa PolyesterThreads)
        if (spec.stencil.contains_annotations) error("Stencil cannot contain type annotations when compiling with PolyesterThreads (Polyester throws errors)!") end
        handle = Expr(:(.), Symbol("Polyester"), QuoteNode(Symbol("@batch")))
        loop = Expr(:macrocall, handle, nothing, loop)
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
        lower_offset = calc_halo_lower_offset(spec)
        upper_offset = calc_halo_upper_offset(spec)   
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
            if (($_RANGE_SYMBOL) === nothing)
                $(range_block.args...)
            end
            $(call_block.args...)
        end
    end

    return CompilationResult(kernel_caller, generated_kernel, caller_name, kernel_name, spec)
end