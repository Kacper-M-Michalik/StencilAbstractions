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