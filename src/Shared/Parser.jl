const mutating_operators = Set{Symbol}([:(=), :(+=), :(-=), :(*=), :(/=), :(\=), :(÷=), :(//=), :(^=), :(%=), :(&=), :(|=), :(⊻=), :(<<=), :(>>>=), :(>>=)])
# TODO: ADD BROADCASTING OPERATORS

function inline_consts(node, mod::Module)
    if node isa Symbol
        if (isdefined(mod, node) && isconst(mod, node))
            return getfield(mod, node)
        end
        return node
    elseif node isa Expr
        args = Any[]
        for arg in node.args
            push!(args, inline_consts(arg, mod))
        end
        return Expr(node.head, args...)
    end
    return node
end

function process_stencil_equation(equation::Expr, mod::Module)
    # Verify stencil definition args 
    stencil_name = equation.args[1].args[1]
    stencil_args = Tuple{Symbol, Union{Symbol, Nothing}}[]    
    input_symbols = Symbol[]
    output_symbols = Symbol[]    
    explicit_output_count = 0
    first_grid_index = 0

    # Verify function args are all symbols
    for (i, arg) in enumerate(equation.args[1].args[2:end])
        if (arg isa Expr && arg.head === :macrocall)
            @assert length(arg.args) == 3
            @assert arg.args[1] == Symbol("@Input") || arg.args[1] == Symbol("@Output")
            #@assert length(arg.args) == 3 && arg.args[1] == Symbol("@Output") "Incorrectly used @Output macro"
            
            if (first_grid_index == 0)
                first_grid_index = i
            end

            if (arg.args[1] == Symbol("@Output"))
                push!(output_symbols, arg.args[3])
                explicit_output_count += 1
            else
                push!(input_symbols, arg.args[3])                             
            end           
            push!(stencil_args, (arg.args[3], nothing))   
        
        elseif (arg isa Expr && arg.head === :(::))
            @assert length(arg.args) == 2
            push!(stencil_args, (arg.args[1], arg.args[2]))   

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
    body = inline_consts(body, mod)

    # Here we verify the stencil kernel syntax
    # 1. TODO: Check all variables used are declared locally (maintain declaration stack and stop global usage)
    # 2. Check mutations only occur to local variables and not any stencil arguments
    # 3. Discover all grid accesses, ensure they match in dimensionality, and use offsets to calculate stencil size
    inputs_lookup = Set{Symbol}(input_symbols)

    sources = Dict{Symbol, StencilSource}()
    dimensions = 0
    
    exprs = Stack{Expr}()
    push!(exprs, body)

    while (!isempty(exprs))
        current::Expr = pop!(exprs)

        # Check if a mutating operator has been used on an argument
        if (in(current.head, mutating_operators))
            @assert length(current.args) == 2
            lhs = current.args[1]
            rhs = current.args[2]
            
            if (lhs isa Symbol)
                @assert !in(lhs, inputs_lookup) "Cannot mutate stencil arguments!" 
            elseif (lhs isa Expr && lhs.head === :ref)
                error("Cannot write to an array at an index!")
            end
            if (rhs isa Expr) push!(exprs, rhs) end

        # Multi buffer return macro (@returns)
        elseif (current.head === :macrocall && current.args[1] == Symbol("@returns"))
            @assert length(current.args) == 3 && current.args[3] isa Expr && current.args[3].head === :tuple
            return_list = current.args[3]
            @assert length(return_list.args) > 0
            explicit_output_count = max(explicit_output_count, length(return_list.args))

        # Found local access, save indexing information                    
        elseif (current.head === :ref && in(current.args[1], inputs_lookup)) 
            var_name = current.args[1]
            indices = current.args[2:end]

            @assert length(indices) > 0 "Stencil array indices cannot be empty!"
            for index in indices
                if !(index isa Int64) 
                    println("Stencil def: ")
                    print(stencil_name)
                    println(equation)
                    dump(index)
                    println() 
                end
                @assert index isa Int64 "Stencil array indices must be integer constants!"
            end

            if (dimensions == 0)
                dimensions = length(indices)
            else
                @assert length(indices) == dimensions
            end
            
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

    # for input in input_symbols
    #     if !haskey(sources, input)
            
    #     end
    # end

    # # Create any implicit end-of-args output buffer arguments
    # while (length(stencil_outputs) < explicit_output_count)        
    #     output_symbol = gensym("output")
    #     push!(stencil_outputs, output_symbol)
    #     push!(stencil_args, (output_symbol, nothing))        
    # end

    return StencilEquation(stencil_name, stencil_args, body, input_symbols, output_symbols, explicit_output_count, first_grid_index, sources, dimensions)
end