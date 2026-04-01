function gen_jacobi_kernel(expr, spec::StencilSpecification, cursor_symbol::Symbol)
    if !(expr isa Expr)
        return Any[expr]
    end
    
    # Replace return with output array write and early return
    if (expr.head === :return)        
        @assert length(expr.args) == 1
        
        #TODO: Find correct output if return is multi arg
        assign_ref = Expr(:ref, stencil.output_grids[1], cursor_symbol)
        @assert length(assign_ref.args) == 2
        
        assign_line = Expr(:(=), assign_ref) 
        @assert length(assign_line.args) == 1
        
        rhs_old = expr.args[1]
        if (rhs_old !== nothing)
            if (rhs_old isa Expr)
                rhs = gen_jacobi_kernel(rhs_old, spec, cursor_symbol)
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
        # TODO, only apply to refs in input arg symbols
        symbol = expr.args[1]
        indices = expr.args[2:end]
        offset = :($cursor_symbol + CartesianIndex($(indices...)))                
        ref_new = Expr(:ref, symbol, offset)

        return Any[ref_new]
    
    # Otherwise copy expr
    else  
        new_expr = Expr(expr.head)
        for line in expr.args
            new_lines = gen_jacobi_kernel(line, spec, cursor_symbol)
            for new_line in new_lines
                push!(new_expr.args, new_line)
            end            
        end

        return Any[new_expr]
    end
end