function get_boundary_check(boundary::DefaultValue, array, var, cursor, offset)
    return quote
        $var = get($array, ($cursor + CartesianIndex($(offset...))), $(boundary.value))
    end
end

function get_boundary_check(boundary::WrapAround, array, var, cursor, offset)
    return quote
        local $var
        let index = ($cursor + CartesianIndex($(offset...)))
            if checkbounds(Bool, $array, index)
                $var = $(array)[index]
            else
                $var = $(array)[CartesianIndex(mod1.(index.I, size($array)))]
            end
        end
    end
end

function get_boundary_check(boundary::Mirror, array, var, cursor, offset::NTuple{N}) where {N}
    return quote
        local $var
        let index = ($cursor + CartesianIndex($(offset...)))
            if checkbounds(Bool, $array, index)
                $var = $(array)[index]
            else
                bounds = size($array)
                new_indices = ntuple(Val($N)) do d
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
                $var = $(array)[CartesianIndex(new_indices...)]
            end
        end
    end
end




 # return quote
    #     local $var
    #     if checkbounds(Bool, $array, ($cursor + CartesianIndex($(offset...))))
    #         $var = grid[($cursor + CartesianIndex($(offset...)))]
    #     else
    #         $var = $(boundary.value)
    #     end
    # end

# @inline Base.@propagate_inbounds function _get_const(grid, index, value)
#     if checkbounds(Bool, grid, index)
#         return grid[index]
#     end
#     return value
# end

# # mod(Index - 1, dimension_size) + 1 -> +-1 required as arrays are 1 indexed
# # mod1 implements this behaviour, we also use broadcasting to apply mod element wise
# @inline Base.@propagate_inbounds function _get_wrap_around(grid, index)
#     if checkbounds(Bool, grid, index)
#         return grid[index]
#     end    
#     return grid[CartesianIndex(mod1.(index.I, size(grid)))]
# end

# @inline Base.@propagate_inbounds function _get_mirror(grid, index::CartesianIndex{N}) where {N}
#     if checkbounds(Bool, grid, index)
#         return grid[index]
#     end
    
#     bounds = size(grid)
#     # This block is emitted/unrolled for each dimension d
#     new_indices = ntuple(Val(N)) do d
#         val = index[d]
#         bound = bounds[d]
        
#         if val < 1
#             return 1 - val
#         elseif val > bound
#             return 2 * bound + 1 - val
#         else
#             return val
#         end
#     end
    
#     return grid[CartesianIndex(new_indices)]
# end

# function _perform_gather(symbol, offset, boundary::DefaultValue{T}) where {T}
#     return :($(_get_const)($(symbol), $offset, $(boundary.value)))
# end
# function _perform_gather(symbol, offset, boundary::WrapAround)
#     return :($(_get_wrap_around)($(symbol), $offset))  
# end
# function _perform_gather(symbol, offset, boundary::Mirror)
#     return :($(_get_mirror)($(symbol), $offset))  
# end
# function _perform_gather(symbol, offset, boundary::B) where {B}
#     error("Unknown boundary type!")
# end
