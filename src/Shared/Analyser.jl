
function inline_body(mod::Module, node)
    if (node isa Symbol)
        #if (isdefined(mod, node) && isconst(mod, node) && isbits(getfield(mod, node))) return getfield(mod, node) end
        return node
    elseif (node isa Expr)
        args = Any[]
        for arg in node.args
            push!(args, inline_body(mod, arg))
        end
        return Expr(node.head, args...)
    end
    return node
end

function verify_keywords(mod::Module, node)
    if (node isa Expr)
        if (node.head in blocked_keywords)         
            return false 
        end     
        for arg in node.args
            if (verify_keywords(mod, arg) == false) return false end
        end
    end
    
    return true
end

function verify_annotations(mod::Module, node)
    if (node isa Expr)
        if (node.head === :(::))         
            return true 
        end     
        for arg in node.args
            if (verify_annotations(mod, arg) == true) return true end
        end
    end
    
    return false
end

function process_args(signature)
    stencil_args = Tuple{Symbol, Union{Symbol, Nothing}}[]    
    grids = Symbol[]    

    # Verify function args are all symbols
    for (i, arg) in enumerate(signature.args[2:end])
        if (arg isa Expr && arg.head === :(::))
            @assert length(arg.args) == 2
            # Type annotation with curly for type specialisation
            if (arg.args[2] isa Expr)
                base_type = arg.args[2].args[1]
                if (base_type == _GRID_TYPE)
                    @assert length(arg.args[2].args) == 2
                    push!(grids, arg.args[1])
                    # Get only field type symbol not whole expr
                    push!(stencil_args, (arg.args[1], arg.args[2].args[2]))
                else               
                    push!(stencil_args, (arg.args[1], arg.args[2]))
                end
            else
            # Just regular type annotation
                if (arg.args[2] == _GRID_TYPE)  
                    push!(stencil_args, (arg.args[1], nothing))
                    push!(grids, arg.args[1])
                else
                    push!(stencil_args, (arg.args[1], arg.args[2]))
                end
            end
        elseif (arg isa Symbol)
            push!(stencil_args, (arg, nothing))
        else
            error("Invalid stencil argument at position $i")
        end
    end

    return stencil_args, grids
end

function process_stencil_definition(equation::Expr, mod::Module, line, file)
    @assert (equation isa Expr && equation.head === :function) "Invalid stencil declaration, must be a function definition on line $line in $file"
    @assert length(equation.args) == 2 "Invalid stencil declaration, too many arguments on line $line in $file"

    # Extract stencil information
    stencil_name = equation.args[1].args[1]
    body = equation.args[2]
    if !(body isa Expr && body.head === :block)
        error("Invalid stencil declaration on line $line in $file")
    end

    # Process function signature
    stencil_args, grids = process_args(equation.args[1])

    # Expand macros and inline consts to allow for proper parsing of body    
    body = macroexpand(mod, body)
    body = inline_body(mod, body)     

    # Verify valid definition   
    if (!verify_keywords(mod, body))
        error("Invalid keyword used inside stencil body on line $line in $file")
    end
    contains_annotations = verify_annotations(mod, body)
    
    # Static analysis of stencil
    input_grids = Set{Symbol}()
    output_grids = Set{Symbol}()    
    sources = Dict{Symbol, StencilSource}()
    dimensions = 0
    unsafe_stencil = false
    inplace_stencil = false

    # Here we verify the stencil kernel syntax
    # 1. Discover all mutations, to determine inputs/outputs and if this is an inplace stencil
    # 2. Discover all grid accesses, ensure they match in dimensionality, and use offsets to calculate stencil size
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
                @assert length(indices) == dimensions "Index dimension mismatch inside stencil body on line $line in $file"
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

    # Check if stencil is Jacobi-like or Gauss-Siedel-like
    for input in input_grids
        if (in(input, output_grids))
            inplace_stencil = true
            if (length(sources[input].offsets) > 1 || sources[input].offsets[1] != ntuple(val -> 0, dimensions))
                unsafe_stencil = true
            end
        end
    end

    return StencilEquation(stencil_name, stencil_args, grids, collect(input_grids), collect(output_grids), body, sources, dimensions, inplace_stencil, unsafe_stencil, contains_annotations)
end