# todo, edge kernels
# on alloc we apply edge? or before iter -> before leave invalid boundary
#





@inline Base.@propagate_inbounds function _get_const(grid, index, value)
    if checkbounds(Bool, grid, index)
        return grid[index]
    end
    return value
end

# mod(Index - 1, dimension_size) + 1 -> +-1 required as arrays are 1 indexed
# mod1 implements this behaviour, we also use broadcasting to apply mod element wise
@inline Base.@propagate_inbounds function _get_wrap_around(grid, index)
    if checkbounds(Bool, grid, index)
        return grid[index]
    end    
    return grid[CartesianIndex(mod1.(index.I, size(grid)))]
end

@inline Base.@propagate_inbounds function _get_mirror(grid, index::CartesianIndex{N}) where {N}
    if checkbounds(Bool, grid, index)
        return grid[index]
    end
    
    bounds = size(grid)
    # This block is emitted/unrolled for each dimension d
    new_indices = ntuple(Val(N)) do d
        val = index[d]
        bound = bounds[d]
        
        if val < 1
            return 1 - val
        elseif val > bound
            return 2 * bound + 1 - val
        else
            return val
        end
    end
    new_index = CartesianIndex(new_indices)    

    # Checks if in bounds, may disable for optimisation?
    return grid[new_index]
end

# # If index still not in range after mirroring, repeat mirroring till in range
# @inline @propagate_inbounds function _get_mirror_bounce(grid, index)
#     if checkbounds(Bool, grid, index)
#         return grid[index]
#     end    
#     return 
# end

function _perform_gather(symbol, offset, boundary::DefaultValue{T}) where {T}
    return :($(_get_const)($(symbol), $offset, $(boundary.value)))
end
function _perform_gather(symbol, offset, boundary::WrapAround)
    return :($(_get_wrap_around)($(symbol), $offset))  
end
function _perform_gather(symbol, offset, boundary::Mirror)
    return :($(_get_mirror)($(symbol), $offset))  
end
function _perform_gather(symbol, offset, boundary::B) where {B}
    error("Unknown boundary type!")
end
