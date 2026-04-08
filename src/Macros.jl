const STENCIL_REGISTRY = Dict{Symbol, StencilEquation}()

macro stencil(equation)      
    if !(equation isa Expr && equation.head === :function)
        error("Invalid stencil declaration!")
    end
    
    stencil_name = equation.args[1].args[1]
    stencil_object = process_stencil_equation(equation, __module__)

    @assert !haskey(STENCIL_REGISTRY, stencil_name) "Stencil with this name has already been declared!"
    STENCIL_REGISTRY[stencil_name] = stencil_object

    return esc(quote    
        $stencil_name = $stencil_object
    end)
end

macro stencil(vars...)
    error("Invalid stencil declaration!")
end

macro compile_stencil(vars...)
    if (length(vars) == 5 || length(vars) == 6)
        if (haskey(STENCIL_REGISTRY, vars[2]))
            
            structure = Core.eval(__module__, vars[3])
            boundary = Core.eval(__module__, vars[4])
            backend = Core.eval(__module__, vars[5])
            settings = nothing            
            if (length(vars) == 6)
                settings = Core.eval(__module__, vars[6])
            end            

            spec = StencilSpecification(STENCIL_REGISTRY[vars[2]], structure, boundary, backend, settings)
            compiled_stencil = internal_compile_stencil(spec)

            return esc(quote
                $(compiled_stencil.caller)

                $(compiled_stencil.kernel)

                $(vars[1]) = CompiledStencil($(compiled_stencil.caller_name), $(spec))
            end)
        else
            error("Stencil with this name does not exist!")
        end
    else
        error("Invalid args in @compile_stencil call!")
    end
end

macro Input(expr)
    return esc(expr)
end

macro Output(expr)
    return esc(expr)
end

macro Box(expr)
    return esc(expr)
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

# macro returns(expr)
#     return esc(expr)
# end