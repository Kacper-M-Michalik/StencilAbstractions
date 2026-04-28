# Stencil transformations are performed here:
# 1. Replace returns
# 2. Replace input refs
# 3. Generate distribution/dispatch methods
function internal_compile_stencil(spec::StencilSpecification{B, ArrayBacking, D, S}) where {B, D <: KernelAbstractions.GPU, S <: SpatialSettings}

    # Generate hygenic symbols for kernel function name, etc.
    name = gensym(spec.name)
    kernel_name = Symbol(name, "_kernel")
    caller_name = Symbol(name, "_caller")
    specialised_kernel = Symbol(name, "_spc")
    block_size = spec.settings.default_block_size

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
            push!(caller_args, Expr(:(::), arg[1], :(CuArray{$type, $(spec.stencil.dimensions)})))      
            
            if (in(arg[1], spec.stencil.input_symbols) && !in(arg[1], spec.stencil.output_symbols))      
                push!(kernel_args, Expr(:macrocall, Symbol("@Const"), nothing, arg[1]))
            else               
                push!(kernel_args, arg[1])# Expr(:(::), arg[1], :(AbstractArray{$type, $(spec.stencil.dimensions)})))     
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
    # push!(kernel_args, _LOWER_BOUND)
    # push!(kernel_args, _UPPER_BOUND)
    # push!(kernel_args, _LAST_DIM)
    push!(kernel_args, :($(_LOWER_BOUND)::NTuple{$(spec.stencil.dimensions - 1), Int}))
    push!(kernel_args, :($(_UPPER_BOUND)::NTuple{$(spec.stencil.dimensions - 1), Int}))
    push!(kernel_args, :(($_LAST_DIM)::UnitRange{Int}))

    # Check/get valid boundaries
    boundaries = get_boundaries(spec)

    # Compile kernel body TODO: ADD SHARED MEMORY
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
        body = Expr(:if, test, body)      
        
        push!(call_block.args, :($(kernel_name)($(spec.device), $_BLOCK_SYMBOL)($(stencil_args...), $_LOWER_BOUND, $_UPPER_BOUND, $_LAST_DIM, 0; ndrange=$_RANGE_SYMBOL)))
        push!(call_block.args, :($(kernel_name)($(spec.device), $_BLOCK_SYMBOL)($(stencil_args...), $_LOWER_BOUND, $_UPPER_BOUND, $_LAST_DIM, 1; ndrange=$_RANGE_SYMBOL)))

        push!(kernel_args, _ITER_SYMBOL)
    elseif (spec.settings.processing isa RedBlack)
        test = :($(_MASK_SYMBOL)[$_CURSOR] == $(_ITER_SYMBOL))
        body = Expr(:if, test, body)     

        call_block = quote
            for iter in 1:mask.colours
                $(kernel_name)($(spec.device), $_BLOCK_SYMBOL)($(stencil_args...), $_LOWER_BOUND, $_UPPER_BOUND, $_LAST_DIM, $_ITER_SYMBOL, (($(_MASK_SYMBOL))).mask; ndrange=$_RANGE_SYMBOL)
            end
        end        

        push!(kernel_args, _ITER_SYMBOL)
        push!(kernel_args, _MASK_SYMBOL)
    else
        push!(call_block.args, :($(specialised_kernel)($(stencil_args...), $_LOWER_BOUND, $_UPPER_BOUND, $_LAST_DIM; ndrange=$_RANGE_SYMBOL)))
    end

    # Apply optimisation flags to loop    
    if (!spec.settings.enable_bounds_checks)
        # inbounds
        body = Expr(:macrocall, Symbol("@inbounds"), nothing, body)
    end    
    if (spec.settings.enable_fastmath)
        # fastmath
        body = Expr(:macrocall, Symbol("@fastmath"), nothing, body)
    end    

    base_cursor = gensym(Symbol("_BASE_CURSOR"))
    dim_index = gensym(Symbol("_LAST_IDX"))

    generated_kernel = quote
        @kernel unsafe_indices=true cpu=false function $(kernel_name)($(kernel_args...))        
            $base_cursor = @index(Global, NTuple)      
            $base_cursor = $base_cursor .+ $_LOWER_BOUND
            
            if (all($base_cursor .<= $_UPPER_BOUND))         
                for $dim_index in $_LAST_DIM      
                    $_CURSOR = CartesianIndex(($base_cursor)..., $dim_index)
                    $body
                end
            end 
        end
    end
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
        $specialised_kernel = $(kernel_name)($(spec.device), $block_size)
        function $(caller_name)($(caller_args...))        
            if (($_RANGE_SYMBOL) === nothing)
                $(range_block.args...)
            end
            if (($_BLOCK_SYMBOL) === nothing)
                $_BLOCK_SYMBOL = $(spec.settings.default_block_size)
            end

            $_LOWER_BOUND = first.(Base.front($_RANGE_SYMBOL)) .- 1
            $_UPPER_BOUND = last.(Base.front($_RANGE_SYMBOL))   
            $_LAST_DIM = ($_RANGE_SYMBOL)[end]      
            $_RANGE_SYMBOL = $_UPPER_BOUND .- $_LOWER_BOUND .+ 2         
            
            $(call_block.args...)
        end
    end

    return CompilationResult(kernel_caller, generated_kernel, caller_name, kernel_name, spec)
end