function rosc_colouring(spec, shape)
@inbounds begin 
    # mask = zeros(UInt8, shape)
    # max = 0
    # indicies = GetLargestStencilIndices(spec)

    # Threads.@threads for index in CartesianIndices(shape)
    #     colors = Set{UInt8}()
    #     for offset in indicies
    #         neighbour = index+offset
    #         if (checkbounds(Bool, mask, neighbour))
    #             push!(colors, mask[neighbour])
    #         end
    #     end
    #     for c in 1:255
    #         if !(c in colors)
    #             mask[index] = c
    #             if (c > max) max = c end
    #             break
    #         end
    #     end        
    # end

    # u = mask
    # while length(u) != 0
    #     defective = Array{CartesianIndex}()
    #     Threads.@threads for index in u
    #         colors = Set{UInt8}()
    #         for offset in indicies
    #             neighbour = index+offset
    #             if (checkbounds(Bool, mask, neighbour))
    #                 push!(colors, mask[neighbour])
    #             end
    #         end
    #         if (mask[index] in colors && )
    #             for c in 1:255
    #                 if !(c in colors)
    #                     mask[index] = c
    #                     if (c > max) max = c end
    #                     push!(defective, )
    #                     break
    #                 end
    #             end        
    #         end 
    #     end
    #     u = defective
    # end


    # return max, mask
end
end

function greedy_colouring(spec, shape)
    mask = zeros(UInt8, shape)
    max = 0
    indicies = GetLargestStencilIndices(spec)

    @inbounds for index in CartesianIndices(shape)
        colors = Set{UInt8}()
        for offset in indicies
            neighbour = index+offset
            if (checkbounds(Bool, mask, neighbour))
                push!(colors, mask[neighbour])
            end
        end
        for c in 1:255
            if !(c in colors)
                mask[index] = c
                if (c > max) max = c end
                break
            end
        end        
    end

    return max, mask
end

function generate_mask(spec, shape, algorithm::GREEDY)
    max, mask = greedy_colouring(spec, shape)
    return ExecutionMask(max, mask)
end

function generate_mask(spec, shape, algorithm::ROSC)
    max, mask = rosc_colouring(spec, shape)
    return ExecutionMask(max, mask)
end
