# const STENCIL_REGISTRY = Dict{Symbol, StencilEquation}()

macro stencil(equation)      
    if !(equation isa Expr && equation.head === :function)
        error("Invalid stencil declaration!")
    end
    
    stencil_name = equation.args[1].args[1]
    stencil_object = process_stencil_equation(equation, __module__)

    # @assert !haskey(STENCIL_REGISTRY, stencil_name) "Stencil with this name has already been declared!"
    #STENCIL_REGISTRY[stencil_name] = stencil_object

    return quote    
        StencilAbstractions.get_stencil(::Val{$(QuoteNode(stencil_name))}) = $stencil_object
        # $stencil_name = $stencil_object
    end
end

macro compile_stencil(vars...)
    if (length(vars) == 5 || length(vars) == 6)
        equation = StencilAbstractions.get_stencil(Val(vars[2]))
        
        if (equation !== nothing)
            structure = Core.eval(__module__, vars[3])
            boundary = Core.eval(__module__, vars[4])
            backend = Core.eval(__module__, vars[5])
            settings = nothing            
            if (length(vars) == 6)
                settings = Core.eval(__module__, vars[6])
            end            

            spec = StencilSpecification(equation, structure, boundary, backend, settings)
            compiled_stencil = internal_compile_stencil(spec)

            # Fix escaping
            return quote
                $(compiled_stencil.caller)

                $(compiled_stencil.kernel)

                $(esc(vars[1])) = StencilAbstractions.CompiledStencil($(compiled_stencil.caller_name), $(spec))
            end
        else
            error("Stencil with this name does not exist!")
        end
    else
        error("Invalid args in @compile_stencil call!")
    end
end

macro Box(grid, n, r)
    accesses = Expr[]
    for index in CartesianIndex(ntuple(i -> -r, n)):CartesianIndex(ntuple(i -> r, n))
        push!(accesses, Expr(:ref, grid, index.I...))
    end    
    return esc(Expr(:call, :SVector, accesses...))
end

macro Cross(expr)
    return esc(expr)
end

macro Star(expr)
    return esc(expr)
end

macro VonNeumann(expr)
    return esc(expr)
end

macro iter(vars...)
    # TODO, add default backend
    @assert length(vars) == 3 || length(vars) == 4 "@iter requires a count, backend and do block!"

    func = vars[1]
    count = vars[2]
    backend = vars[3]
    unroll_count = length(vars) == 4 ? vars[4] : 1

    @assert func.head === :(->)
    @assert length(func.args) == 2    
    for arg in func.args[2].args
        if (arg isa Expr)
            @assert arg.head === :call
        elseif !(arg isa LineNumberNode)
            error("Only stencil calls are allowed in an @iter block!")
        end
    end
    
    @assert count isa Expr || count isa Symbol || count isa Int "@iter requries count be an Expr, Variable or Int!"

    @assert backend isa Expr && backend.head === :call "@iter requries a backend instance!"

    @assert unroll_count isa Int

    backend_obj = Core.eval(__module__, backend)
    body = generate_graph_loop(func.args[2], count, backend_obj, unroll_count)

    return body
end