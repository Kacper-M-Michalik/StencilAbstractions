macro box(grid, n, r)
    accesses = Expr[]
    for index in CartesianIndex(ntuple(i -> -r, n)):CartesianIndex(ntuple(i -> r, n))
        push!(accesses, Expr(:ref, grid, index.I...))
    end    
    return esc(Expr(:call, :SVector, accesses...))
end

macro cross(expr)
    return esc(expr)
end

macro star(expr)
    return esc(expr)
end

macro von_neumann(expr)
    return esc(expr)
end

macro stencil(equation)          
    stencil = process_stencil_definition(equation, __module__, __source__.line, __source__.file)

    return quote    
        StencilAbstractions.get_stencil(::Val{$(QuoteNode(stencil.name))}) = $stencil
    end
end