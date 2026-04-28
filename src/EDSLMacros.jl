macro box(grid, n, r)
    mod == __module__
    if !(n isa Int)
        if (isdefined(mod, n) && isconst(mod, n)) 
            n = getfield(mod, n) 
        end
        if !(n isa Int)
            error("Macro box, value n must be an Integer or const Integer")
        end
    end
    if !(r isa Int)
        if (isdefined(mod, r) && isconst(mod, r)) 
            r = getfield(mod, r) 
        end
        if !(r isa Int)
            error("Macro box, value r must be an Integer or const Integer")
        end
    end    

    accesses = Expr[]
    for index in CartesianIndex(ntuple(i -> -r, n)):CartesianIndex(ntuple(i -> r, n))
        push!(accesses, Expr(:ref, grid, index.I...))
    end    
    return esc(Expr(:call, :SVector, accesses...))
end

macro cross(grid, n, r)
    mod == __module__
    if !(n isa Int)
        if (isdefined(mod, n) && isconst(mod, n)) 
            n = getfield(mod, n) 
        end
        if !(n isa Int)
            error("Macro cross, value n must be an Integer or const Integer")
        end
    end
    if !(r isa Int)
        if (isdefined(mod, r) && isconst(mod, r)) 
            r = getfield(mod, r) 
        end
        if !(r isa Int)
            error("Macro cross, value r must be an Integer or const Integer")
        end
    end    

    accesses = Set{Expr}()
    for dim in 1:n
        index = zeros(Int, 1, n)
        for offset in (-r):(r)
            index[dim] = offset
            push!(accesses, Expr(:ref, grid, Tuple(index)))
        end
    end    
    return esc(Expr(:call, :SVector, collect(accesses)...))
end

macro stencil(equation)    
    stencil = process_stencil_definition(equation, __module__, __source__.line, __source__.file)

    return esc(quote    
        const $(stencil.name) = $stencil
        # get_stencil(::Val{$(QuoteNode(stencil.name))}) = $stencil
    end)
end