# Stencil transformations are performed here:
# 1. Replace returns
# 2. Replace input refs
# 3. Wrap stencil body in some distribution loops
function compile_stencil(stencil::StencilEquation, settings::CartesianGrid{B, H, KernelAbstractions.CPU, S}, optimisations::OptimisationSettings = OptimisationSettings()) where {B <: GridBoundary, H <: HaloType, S <: CPUStrategy}
    
    @assert optimisations.thread_count > 0

    # Generate hygenic symbols for kernel function name, etc.
    kernel_name = gensym(Symbol(stencil.name, "!"))
    cursor_symbol = gensym(Symbol("cursor"))
    id_symbol = gensym(Symbol("id"))

    local transformed_body::Expr
    if (settings.strategy isa BasicCPU)
        body = generic_compile(stencil.body, stencil, settings, cursor_symbol)
        @assert length(body) == 1
        transformed_body = body[1]
    elseif (settings.strategy isa RegisterBlockingCPU)
        transformed_body = cpu_body_staged(stencil.body, stencil, settings, cursor_symbol)
    else
        error("Strategy not yet implemented!")
    end    

    iter_block = Expr(:block)
    rng_block = Expr(:block)
    index_symbols = Symbol[]
    for dim in stencil.dimensions:-1:1
        index = gensym(Symbol("dim", dim))
        range = gensym(Symbol("rng", dim))
        push!(index_symbols, index)

        # Add range to for loop
        iter = Expr(:(=), index, range)
        push!(iter_block.args, iter)

        # Index range calculation
        if (dim == stencil.dimensions)            
            # step = div(size[$dim], $(optimisations.thread_count), Base.RoundUp)
            # lower = step*$id_symbol + 1
            # upper = min(step*($id_symbol+1), size[$dim])
            # $range = lower:upper
            push!(rng_block.args, :($range = (div($_SIZE_SYMBOL[$dim], $(optimisations.thread_count), Base.RoundUp)*$id_symbol+1):min(div($_SIZE_SYMBOL[$dim], $(optimisations.thread_count), Base.RoundUp)*($id_symbol+1), $_SIZE_SYMBOL[$dim])))
        else
            push!(rng_block.args, :($range = 1:($_SIZE_SYMBOL[$dim])))
        end
    end

    # Insert cursor definition
    cursor_assignment = Expr(:(=), cursor_symbol, :(CartesianIndex($(reverse(index_symbols)...))))
    insert!(transformed_body.args, 1, cursor_assignment)

    # Create kernel end label
    push!(transformed_body.args, _KERNEL_END)

    main_loops = Expr(:for, iter_block, transformed_body)
    
    function_tags = Symbol[]

    # Doesnt work with dynamic boundary function
    # if (optimisations.enable_fastmath && !optimisations.enable_bounds_checks)
    #     # Turbo implies fastmath and inbounds
    #     main_loops = Expr(:macrocall, Symbol("@turbo"), nothing, main_loops)   
    # else 
    if (optimisations.enable_fastmath)
        # fastmath
        push!(function_tags, Symbol("@fastmath"))
    end
    if (!optimisations.enable_bounds_checks)
        # inbounds
        blocked_loops = Expr(:block, main_loops)
        main_loops = Expr(:macrocall, Symbol("@inbounds"), nothing, blocked_loops)
    end

    args = Any[]
    for arg in stencil.args       
        if (arg[2] !== nothing)
            push!(args, Expr(Symbol("::"), arg[1], arg[2]))
        else
            push!(args, arg[1])            
        end
    end

    # TODO: Base.Experimental.@aliasscope
    # Example:
    # Base.Experimental.@aliasscope begin
    #     AC = Base.Experimental.Const(A) 
    #     for i in eachindex(B, AC)
    #         B[i] = AC[i] 
    #     end
    # end

    generated_kernel = quote        
        function $kernel_name($(args...); $(_SIZE_SYMBOL))      
            Threads.@threads for $id_symbol = 0:$(optimisations.thread_count-1)
                # Range calculations
                $(rng_block.args...)

                # Main loops
                $main_loops
            end
        end
    end

    # Kernel wrapped in block by default
    generated_kernel = generated_kernel.args[2]
    for tag in function_tags
        generated_kernel = Expr(:macrocall, tag, nothing, generated_kernel)
    end

    print("Generated kernel:\n")
    print(generated_kernel)
    print("\n")

    # First call to function object is what actually triggers compilation of kernel, we can't do a dummy run here since we don't necessarily know all arg types
    specialised_kernel_ptr = eval(generated_kernel)
    return CompiledStencil(specialised_kernel_ptr, settings)
end

function generic_compile(expr, stencil::StencilEquation, settings::CartesianGrid, cursor_symbol::Symbol)
    if !(expr isa Expr)
        return Any[expr]
    end
    
    if (expr.head === :(=) && expr.args[1] isa Symbol && in(expr.args[1], stencil.output_args))
        @assert length(expr.args) == 2          
        out_arg = expr.args[1]

        assign_ref = Expr(:ref, out_arg, :i)
        @assert length(assign_ref.args) == 2
        
        assign_line = Expr(:(=), assign_ref) 
        @assert length(assign_line.args) == 1
        
        rhs_old = expr.args[2]
        if (rhs_old isa Expr)
            rhs = generic_compile(rhs_old, stencil, settings, cursor_symbol)
            @assert length(rhs) == 1
            push!(assign_line.args, rhs[1]) 
        else
            push!(assign_line.args, rhs_old)  
        end

        return Any[assign_line]     

    # Replace return with output array write and early return
    elseif (expr.head === :return)        
        @assert length(expr.args) == 1
        
        #TODO: Find correct output if return is multi arg
        assign_ref = Expr(:ref, stencil.output_grids[1], cursor_symbol)
        @assert length(assign_ref.args) == 2
        
        assign_line = Expr(:(=), assign_ref) 
        @assert length(assign_line.args) == 1
        
        rhs_old = expr.args[1]
        if (rhs_old !== nothing)
            if (rhs_old isa Expr)
                rhs = generic_compile(rhs_old, stencil, settings, cursor_symbol)
                @assert length(rhs) == 1
                push!(assign_line.args, rhs[1]) 
            else
                push!(assign_line.args, rhs_old)  
            end
            return Any[assign_line, _KERNEL_GOTO]
        end
        
        return Any[_KERNEL_GOTO]     

    # Modify array ref with appropriate bounds checked access
    elseif (expr.head === :ref)        
        symbol = expr.args[1]
        indices = expr.args[2:end]
        offset = :($cursor_symbol + CartesianIndex($(indices...)))                
        ref_new = _perform_gather(symbol, offset, settings.boundary)   

        return Any[ref_new]
    
    # Perform standard array access
    elseif (expr.head === :macrocall && expr.args[1] == Symbol("@abs"))
        symbol = expr.args[3].args[1]
        indices = expr.args[3].args[2:end]
        offset = Expr(:call, :CartesianIndex, indices...)                    
        ref_new = Expr(:ref, symbol, offset)

        return Any[ref_new]

    # Otherwise copy expr
    else  
        new_expr = Expr(expr.head)
        for line in expr.args
            new_lines = generic_compile(line, stencil, settings, cursor_symbol)
            for new_line in new_lines
                push!(new_expr.args, new_line)
            end            
        end

        return Any[new_expr]
    end
end

# The staged version first reads all indicies from memory into a StaticArray, for which LLVM will try to store all entries of the array in register/spillover, which may allow better performance on kernels where neighbour values are frequently reread (to be profiled)
# TODO: try to define the next static array using the previous one, as should prevent some reads from memory
function cpu_body_staged(expr, stencil::StencilEquation, settings::StencilSettings, cursor_symbol)
    #     

end

# TODO: Finish
function staged_compile(expr, stencil::StencilEquation, settings::StencilSettings, cursor_symbol)
    if !(expr isa Expr)
        return Any[expr]
    end
    
    if (expr.head === :(=) && expr.args[1] isa Symbol && in(expr.args[1], stencil.output_args))
        @assert length(expr.args) == 2          
        out_arg = expr.args[1]

        assign_ref = Expr(:ref, out_arg, :i)
        @assert length(assign_ref.args) == 2
        
        assign_line = Expr(:(=), assign_ref) 
        @assert length(assign_line.args) == 1
        
        rhs_old = expr.args[2]
        if (rhs_old isa Expr)
            rhs = staged_compile(rhs_old, stencil, settings, cursor_symbol)
            @assert length(rhs) == 1
            push!(assign_line.args, rhs[1]) 
        else
            push!(assign_line.args, rhs_old)  
        end

        return Any[assign_line]     

    # Replace return with output array write and early return
    elseif (expr.head === :return)        
        @assert length(expr.args) == 1
        
        #TODO: Find correct output if return is multi arg
        assign_ref = Expr(:ref, stencil.output_args[1], cursor_symbol)
        @assert length(assign_ref.args) == 2
        
        assign_line = Expr(:(=), assign_ref) 
        @assert length(assign_line.args) == 1
        
        rhs_old = expr.args[1]
        if (rhs_old !== nothing)
            if (rhs_old isa Expr)
                rhs = staged_compile(rhs_old, stencil, settings, cursor_symbol)
                @assert length(rhs) == 1
                push!(assign_line.args, rhs[1]) 
            else
                push!(assign_line.args, rhs_old)  
            end
            return Any[assign_line, _KERNEL_GOTO]
        end
        
        return Any[_KERNEL_GOTO]     

    # Modify array ref with appropriate bounds checked access
    elseif (expr.head === :ref)        
        symbol = expr.args[1]
        indices = expr.args[2:end]
        offset = :($cursor_symbol + CartesianIndex($(indices...)))                
        ref_new = _perform_gather(symbol, offset, settings)   

        return Any[ref_new]
    
    # Perform standard array access
    elseif (expr.head === :macrocall && expr.args[1] == Symbol("@abs"))
        symbol = expr.args[3].args[1]
        indices = expr.args[3].args[2:end]
        offset = Expr(:call, :CartesianIndex, indices...)                    
        ref_new = Expr(:ref, symbol, offset)

        return Any[ref_new]

    # Otherwise copy expr
    else  
        new_expr = Expr(expr.head)
        for line in expr.args
            new_lines = staged_compile(line, stencil, settings, cursor_symbol)
            for new_line in new_lines
                push!(new_expr.args, new_line)
            end            
        end

        return Any[new_expr]
    end
end