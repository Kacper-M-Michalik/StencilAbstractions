
function compose_stencils()
    # For resulting index dependency space of two stencils:
    # Take indices as list of vectors
    # Take product of two stencil index sets
    # Sum each tuple individually to produce new vector per tuple
    # Take all resulting vectors into set -> new index dependency space
end

# function compose(left::StencilEquation, right::StencilEquation)
#     return StencilEquation(Expr(:nothing), Expr(:nothing), 0, 0)
# end
