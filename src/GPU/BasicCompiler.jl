# Stencil transformations are performed here:
# 1. Replace returns
# 2. Replace input refs
# 3. Generate distribution/dispatch methods
function internal_compile_stencil(spec::StencilSpecification{CartesianGrid, B, D, BasicSettings{W}}) where {B, D <: KernelAbstractions.GPU, W}

    # Generate hygenic symbols for kernel function name, etc.
    kernel_name = gensym(Symbol(spec.stencil.name, "!"))
    caller_name = gensym(Symbol(kernel_name, "_caller"))   
    cursor_symbol = gensym(Symbol("cursor"))
    first_grid_index = spec.stencil.first_grid_index

    # Mark kernel args as constant for KernelAbstractions
    wrapped_args = Any[]    
    kernel_args = Any[]
    for arg in spec.stencil.args
        if (in(arg[1], spec.stencil.input_symbols) && !in(arg[1], spec.stencil.output_symbols))           
            push!(wrapped_args, Expr(:macrocall, Symbol("@Const"), nothing, arg[1]))
            push!(kernel_args, arg[1])
        elseif (!in(arg[1], spec.stencil.input_symbols) && !in(arg[1], spec.stencil.output_symbols) && arg[2] !== nothing)
            push!(wrapped_args, Expr(:(::), arg[1], arg[2]))
            push!(kernel_args, Expr(:(::), arg[1], arg[2]))
        else
            push!(wrapped_args, arg[1])            
            push!(kernel_args, arg[1])
        end
    end        

    # Compile kernel body    
    body_array = gen_kernel(spec.stencil.body, spec, cursor_symbol, !spec.structure.halo, Dict{Symbol, Symbol}())
    @assert length(body_array) == 1
    body::Expr = body_array[1]

    # Apply optimisation flags
    if (spec.settings.enable_fastmath)
        # fastmath
        body = Expr(:macrocall, Symbol("@fastmath"), nothing, body)
    end
    if (!spec.settings.enable_bounds_checks)
        # inbounds
        body = Expr(:macrocall, Symbol("@inbounds"), nothing, body)
    end

    # Generate kernel, eval into function table
    generated_kernel = quote
        @kernel unsafe_indices=true function $(kernel_name)($(wrapped_args...), $_LOWER_BOUND, $_UPPER_BOUND)        
            $cursor_symbol = @index(Global, Cartesian)      
            $cursor_symbol += CartesianIndex($_LOWER_BOUND)
            
            if (all(($cursor_symbol).I .<= $_UPPER_BOUND))               
                $(body.args...)               
            end 
        end        
    end
    
    lower_offset = GetHaloLowerOffset(spec)
    upper_offset = GetHaloUpperOffset(spec)

    # Generate the dispatcher function
    # Kernel abstractions automatically calculates the requried number of tiles
    local kernel_caller
    if (spec.structure.halo)
        kernel_caller = quote
            function $(caller_name)($(kernel_args...), lower, upper, threads, block, order)
                local offset
                if length(lower) != 0
                    @assert length(lower) == $(spec.stencil.dimensions)
                    offset = lower .+ $(lower_offset) .- 1
                else
                    offset = $(lower_offset)
                end 

                if length(upper) != 0
                    # TODO: Add actual size checks?
                    @assert length(upper) == $(spec.stencil.dimensions)
                    upper = upper .+ $(lower_offset)
                else
                    upper = size($(kernel_args[first_grid_index])) .- $(upper_offset)                    
                end          

                if length(block) == 0
                    block = $(spec.settings.default_block_size)
                end
                
                range = (upper .- offset)
                $(kernel_name)($(spec.device), block)($(kernel_args...), offset, upper; ndrange=range)  
            end
        end
    else
        kernel_caller = quote
            function $(caller_name)($(kernel_args...), lower, upper, threads, block, order)
                local offset
                if length(lower) != 0
                    @assert length(lower) == $(spec.stencil.dimensions)
                    offset = lower .- 1
                else
                    offset = ntuple(val -> 0, $(spec.stencil.dimensions))
                end 

                if length(upper) != 0
                    # TODO: Add actual size checks?
                    @assert length(upper) == $(spec.stencil.dimensions)
                else
                    upper = size($(kernel_args[first_grid_index]))
                end          
                
                if length(block) == 0
                    block = $(spec.settings.default_block_size)
                end

                range = (upper .- offset)
                $(kernel_name)($(spec.device), block)($(kernel_args...), offset, upper; ndrange=range)  
            end
        end
    end    
    
    return CompilationResult(kernel_caller, generated_kernel, caller_name, kernel_name, spec)
end

function internal_compile_stencil(spec::StencilSpecification{CartesianGrid, B, D, BasicSettings{W, T}}) where {B, D <: KernelAbstractions.GPU, W, T}

    # Generate hygenic symbols for kernel function name, etc.
    kernel_name = gensym(Symbol(spec.stencil.name, "!"))
    caller_name = gensym(Symbol(kernel_name, "_caller"))   
    cursor_symbol = gensym(Symbol("cursor"))

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
                body = transform_read(body, grid, false, nothing, grid, cursor_symbol, nothing)
            else
                for offset in spec.stencil.sources[grid].offsets
                    var_name = gensym(Symbol(grid))         
                    push!(gather_blocks, get_boundary_check(boundaries[i], grid, var_name, cursor_symbol, offset))
                    body = transform_read(body, grid, true, offset, var_name, nothing, nothing)
                end
            end     
        end  
    end
    for output in spec.stencil.output_symbols
        body = transform_write(body, output, output, cursor_symbol, nothing)
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
        test = :((sum(($(cursor_symbol)).I) % 2) == $(_ITER_SYMBOL))
        body = Expr(:if, test, body)      
        
        push!(call_block.args, :($(kernel_name)($(stencil_args...), $(_RANGE_SYMBOL), 0)))
        push!(call_block.args, :($(kernel_name)($(stencil_args...), $(_RANGE_SYMBOL), 1)))

        push!(kernel_args, _ITER_SYMBOL)
    elseif (W === MultiColour)
        test = :($(_MASK_SYMBOL)[$(cursor_symbol)] == $(_ITER_SYMBOL))
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
    iterator = :($(cursor_symbol) = CartesianIndices($(_RANGE_SYMBOL)))
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
    
    kernel_caller = quote
        function $(caller_name)($(caller_args...))        
            if (length($(_RANGE_SYMBOL)) == 0)
                $(_RANGE_SYMBOL) = axes($(spec.stencil.ordered_grids[1]))
            end
            $(call_block.args...)
        end
    end

    println(generated_kernel)

    return CompilationResult(kernel_caller, generated_kernel, caller_name, kernel_name, spec)
end