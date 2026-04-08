
function inline_body(mod::Module, node)
    if node isa Symbol
        if (isdefined(mod, node) && isconst(mod, node))
            return getfield(mod, node)
        end
        return node
    elseif node isa Expr
        args = Any[]
        for arg in node.args
            push!(args, inline_body(mod, arg))
        end
        return Expr(node.head, args...)
    end
    return node
end

function process_stencil_equation(equation::Expr, mod::Module)
    # Verify stencil definition args 
    stencil_name = equation.args[1].args[1]
    stencil_args = Tuple{Symbol, Union{Symbol, Nothing}}[]    
    input_grids = Set{Symbol}()
    output_grids = Set{Symbol}()    
    grids = Symbol[]    
    first_grid_index = 0
    sources = Dict{Symbol, StencilSource}()
    dimensions = 0
    unsafe_stencil = false
    inplace_stencil = false

    # Verify function args are all symbols
    for (i, arg) in enumerate(equation.args[1].args[2:end])
        # if (arg isa Expr && arg.head === :macrocall)
        #     @assert length(arg.args) == 3
        #     @assert arg.args[1] == Symbol("@Input") || arg.args[1] == Symbol("@Output")
        #     #@assert length(arg.args) == 3 && arg.args[1] == Symbol("@Output") "Incorrectly used @Output macro"
            
        #     if (first_grid_index == 0)
        #         first_grid_index = i
        #     end

        #     if (arg.args[1] == Symbol("@Output"))
        #         push!(output_grids, arg.args[3])
        #         explicit_output_count += 1
        #     else
        #         push!(input_grids, arg.args[3])                             
        #     end           
        #     push!(stencil_args, (arg.args[3], nothing))   
        
        # else
        if (arg isa Expr && arg.head === :(::))
            @assert length(arg.args) == 2
            if (arg.args[2] == _GRID_TYPE)
                if (first_grid_index == 0)
                    first_grid_index = i
                end

                push!(stencil_args, (arg.args[1], nothing))
                push!(grids, arg.args[1])
            else
                push!(stencil_args, (arg.args[1], arg.args[2]))
            end
        elseif (arg isa Symbol)
            push!(stencil_args, (arg, nothing))
        else
            error("Stencil argument at position $i is not a valid variable!")
        end
    end

    # Extract stencil program block
    body = equation.args[2]
    if !(body isa Expr && body.head === :block)
        error("Invalid stencil body declaration!")
    end

    # Expand macros and inline consts to allow for proper parsing of body
    body = macroexpand(mod, body)
    body = inline_body(mod, body)

    # Here we verify the stencil kernel syntax
    # 1. TODO: Check all variables used are declared locally (maintain declaration stack and stop global usage)
    # 2. Discover all mutations, to determine inputs/outputs and if this is an inplace stencil
    # 3. Discover all grid accesses, ensure they match in dimensionality, and use offsets to calculate stencil size
    grids_lookup = Set{Symbol}(grids)
    
    exprs = Stack{Expr}()
    push!(exprs, body)

    while (!isempty(exprs))
        current::Expr = pop!(exprs)
        
        # Check if we are writing to a grid
        if (current.head === :(=) || in(current.head, inplace_operators))
            @assert length(current.args) == 2
            lhs = current.args[1]
            rhs = current.args[2]

            if (lhs isa Symbol && in(lhs, grids_lookup))
                if (current.head === :(=))
                    push!(output_grids, lhs)
                else
                    push!(input_grids, lhs)
                    push!(output_grids, lhs)
                end
            elseif (lhs isa Expr && lhs.head === :ref)
                error("Cannot write to an array at an index!")
                #@assert (!(lhs.args[1] isa Symbol) || !(in(lhs.args[1], grids))) "Cannot index into a grid assignment, to write to a grid only use its variable name!"
            end

            if (rhs isa Expr) push!(exprs, rhs) end        

        # Found local access, save indexing information                    
        elseif (current.head === :ref && in(current.args[1], grids_lookup)) 
            var_name = current.args[1]
            indices = current.args[2:end]

            @assert length(indices) > 0 "Stencil array indices cannot be empty!"
            for index in indices
                if !(index isa Int) 
                    error("")
                    println("Error in stencil def: ")
                    print(stencil_name)
                    println(equation)
                    dump(index)
                    println() 
                end
                @assert index isa Int "Stencil array indices must be integer constants!"
            end            

            if (dimensions == 0)
                dimensions = length(indices)
            else
                @assert length(indices) == dimensions
            end

            push!(input_grids, var_name)
            if !haskey(sources, var_name)
                radii = Vector{StencilDimension}()
                for index in indices
                    push!(radii, StencilDimension(index, index))
                end

                offsets = Vector{Vector{Int64}}()
                push!(offsets, indices)
                
                sources[var_name] = StencilSource(var_name, radii, offsets)
            else
                source = sources[var_name]

                for (i, index) in enumerate(indices)
                    if (index > source.radii[i].max) source.radii[i].max = index end
                    if (index < source.radii[i].min) source.radii[i].min = index end
                end

                push!(source.offsets, indices)
            end       
                 
        else
            for arg in current.args
                if (arg isa Expr)   
                    push!(exprs, arg)
                end
            end
        end
    end

    for input in input_grids
        if (in(input, output_grids))
            inplace_stencil = true
            if (length(sources[input].offsets) > 1 || sources[input].offsets[1] != ntuple(val -> 0, dimensions))
                unsafe_stencil = true
            end
        end
    end

    return StencilEquation(stencil_name, stencil_args, body, collect(input_grids), collect(output_grids), first_grid_index, sources, dimensions, inplace_stencil, unsafe_stencil)
end