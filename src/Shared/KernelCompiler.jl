function setup_caller_args!(args)
    push!(args, _RANGE_SYMBOL)
    push!(args, _BLOCK_SYMBOL)
    push!(args, _MASK_SYMBOL)
end

function get_boundaries(spec::StencilSpecification)
    boundaries = []

    if (length(spec.boundary) == 0) error("Must specify at least one boundary condition!") end
    if (length(spec.boundary) == 1)
        for i in 1:length(spec.stencil.ordered_grids)
            push!(boundaries, spec.boundary[1])
        end
    else
        if (length(spec.boundary) != length(spec.stencil.ordered_grids)) error("If specifying different boundary types for arrays, number specified must match used input grid count!") end
        for boundary in spec.boundary
            if (boundary isa Halo) error("Cannot mix halo and nonhalo boundaries!") end
            push!(boundaries, boundary)
        end
    end

    return boundaries
end

function transform_read(node, old::Symbol, by_offset, search_offset, new::Symbol, cursor_symbol, cursor_offset)
    if !(node isa Expr)
        return node
    end    

    # Modify array ref with appropriate access
    if (node.head === :ref && node.args[1] == old && (!by_offset || node.args[2:end] == search_offset))        
        indices = node.args[2:end]   

        ref_new = new
        if (cursor_symbol !== nothing)
            offset = (cursor_offset === nothing) ? :($cursor_symbol + CartesianIndex($(indices...))) : :($cursor_symbol + CartesianIndex($(indices...)) + CartesianIndex($(cursor_offset...)))     
            ref_new = Expr(:ref, new, offset)
        end
        
        return ref_new

    # Otherwise copy expr
    else  
        new_expr = Expr(node.head)
        for line in node.args
            new_line = transform_read(line, old, by_offset, search_offset, new, cursor_symbol, cursor_offset)
            push!(new_expr.args, new_line)
        end

        return new_expr
    end
end

function transform_write(node, old::Symbol, new::Symbol, cursor_symbol, cursor_offset)
    if !(node isa Expr)
        return node
    end    

    # Process assignment to output
    if (node.head === :(=) && node.args[1] isa Symbol && node.args[1] == old)
        @assert length(node.args) == 2          

        assign_ref = new
        if (cursor_symbol !== nothing)
            offset = (cursor_offset === nothing) ? :($cursor_symbol) : :($cursor_symbol + CartesianIndex($(cursor_offset...)))     
            assign_ref = Expr(:ref, new, offset)
        end
        
        assign_line = Expr(:(=), assign_ref) 
        @assert length(assign_line.args) == 1
        
        rhs_old = node.args[2]
        if (rhs_old isa Expr)
            rhs = transform_write(rhs_old, old, new, cursor_symbol, cursor_offset)
            push!(assign_line.args, rhs) 
        else
            push!(assign_line.args, rhs_old)  
        end

        return assign_line

    # Otherwise copy expr
    else  
        new_expr = Expr(node.head)
        for line in node.args
            new_line = transform_write(line, old, new, cursor_symbol, cursor_offset)
            push!(new_expr.args, new_line)
        end

        return new_expr
    end
end


# function gen_kernel(expr, spec::StencilSpecification{ArrayBacking, B, D}, cursor_symbol::Symbol, access_check::Bool, symbol_replacement::Dict{Symbol, Symbol}) where {B, D}
#     if !(expr isa Expr)
#         return Any[expr]
#     end
    
#     # Process assignment to output buffers
#     if (expr.head === :(=) && expr.args[1] isa Symbol && in(expr.args[1], spec.stencil.output_symbols))
#         @assert length(expr.args) == 2          
#         out_arg = expr.args[1]

#         assign_ref = Expr(:ref, out_arg, cursor_symbol)
#         @assert length(assign_ref.args) == 2
        
#         assign_line = Expr(:(=), assign_ref) 
#         @assert length(assign_line.args) == 1
        
#         rhs_old = expr.args[2]
#         if (rhs_old isa Expr)
#             rhs = gen_kernel(rhs_old, spec, cursor_symbol, access_check, symbol_replacement)
#             @assert length(rhs) == 1
#             push!(assign_line.args, rhs[1]) 
#         else
#             push!(assign_line.args, rhs_old)  
#         end

#         return Any[assign_line]  
#     # Replace return with output array write and early return
#     # if (expr.head === :return)        
#     #     @assert length(expr.args) == 1
     
#     #     #TODO: Copy into multiple arrays if one return and multiple outputs
#     #     assign_ref = Expr(:ref, output_symbols[1], cursor_symbol)
#     #     @assert length(assign_ref.args) == 2
        
#     #     assign_line = Expr(:(=), assign_ref) 
#     #     @assert length(assign_line.args) == 1
        
#     #     rhs_old = expr.args[1]
#     #     if (rhs_old !== nothing)
#     #         if (rhs_old isa Expr)
#     #             rhs = gen_kernel(rhs_old, spec, output_symbols, cursor_symbol, access_check, symbol_replacement)
#     #             @assert length(rhs) == 1
#     #             push!(assign_line.args, rhs[1]) 
#     #         else
#     #             push!(assign_line.args, rhs_old)  
#     #         end
#     #         return Any[assign_line, _KERNEL_GOTO]
#     #     end
        
#     #     return Any[_KERNEL_GOTO]     

#     # Modify array ref with appropriate access
#     elseif (expr.head === :ref && in(expr.args[1], spec.stencil.input_symbols))        
#         symbol = expr.args[1]
#         indices = expr.args[2:end]
#         offset = :($cursor_symbol + CartesianIndex($(indices...)))                
        
#         if (haskey(symbol_replacement, symbol)) 
#            symbol = symbol_replacement[symbol]
#         end

#         local ref_new
#         if (access_check)
#             # Bounds checked
#             ref_new = _perform_gather(symbol, offset, spec.boundary)   
#         else
#             ref_new = Expr(:ref, symbol, offset)
#         end

#         return Any[ref_new]

#     # Otherwise copy expr
#     else  
#         new_expr = Expr(expr.head)
#         for line in expr.args
#             new_lines = gen_kernel(line, spec, cursor_symbol, access_check, symbol_replacement)
#             for new_line in new_lines
#                 push!(new_expr.args, new_line)
#             end            
#         end

#         return Any[new_expr]
#     end
# end