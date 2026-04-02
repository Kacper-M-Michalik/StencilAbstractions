function gen_kernel(expr, spec::StencilSpecification{CartesianGrid, B, D}, output_symbols::Array{Symbol}, cursor_symbol::Symbol, access_check::Bool, symbol_replacement::Dict{Symbol, Symbol}) where {B, D}
    if !(expr isa Expr)
        return Any[expr]
    end
    
    # TODO: @returns

    # Replace return with output array write and early return
    if (expr.head === :return)        
        @assert length(expr.args) == 1
     
        #TODO: Copy into multiple arrays if one return and multiple outputs
        assign_ref = Expr(:ref, output_symbols[1], cursor_symbol)
        @assert length(assign_ref.args) == 2
        
        assign_line = Expr(:(=), assign_ref) 
        @assert length(assign_line.args) == 1
        
        rhs_old = expr.args[1]
        if (rhs_old !== nothing)
            if (rhs_old isa Expr)
                rhs = gen_kernel(rhs_old, spec, output_symbols, cursor_symbol, access_check, symbol_replacement)
                @assert length(rhs) == 1
                push!(assign_line.args, rhs[1]) 
            else
                push!(assign_line.args, rhs_old)  
            end
            return Any[assign_line, _KERNEL_GOTO]
        end
        
        return Any[_KERNEL_GOTO]     

    # Modify array ref with appropriate access
    elseif (expr.head === :ref && expr.args[1] in spec.stencil.input_symbols)        
        symbol = expr.args[1]
        indices = expr.args[2:end]
        offset = :($cursor_symbol + CartesianIndex($(indices...)))                
        
        if (haskey(symbol_replacement, symbol)) 
           symbol = symbol_replacement[symbol]
        end

        local ref_new
        if (access_check)
            # Bounds checked
            ref_new = _perform_gather(symbol, offset, spec.boundary)   
        else
            ref_new = Expr(:ref, symbol, offset)
        end

        return Any[ref_new]
    
    # Otherwise copy expr
    else  
        new_expr = Expr(expr.head)
        for line in expr.args
            new_lines = gen_kernel(line, spec, output_symbols, cursor_symbol, access_check, symbol_replacement)
            for new_line in new_lines
                push!(new_expr.args, new_line)
            end            
        end

        return Any[new_expr]
    end
end