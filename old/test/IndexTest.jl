module IndexTest

    const block_size = (8, 8)
    const warp_size = 32
    stencil_dims = [(-1, 1), (-1, 1)]
    neighbour_size = (block_size[1] - stencil_dims[1][1] + stencil_dims[1][2], block_size[2] - stencil_dims[2][1] + stencil_dims[2][2])
    thread_jump = prod(block_size)
    last_index = prod(neighbour_size)
    index_region = (Int(ceil(neighbour_size[1]/warp_size))*warp_size, neighbour_size[2])

    out_grid_size = (40, 40)
    grid_dim = (Int(ceil(out_grid_size[1]/block_size[1])), Int(ceil(out_grid_size[2]/block_size[2])))

    arr = zeros(out_grid_size)

    threadID = (0,0)    
    blockID = (0,0)    
    id = threadID[1] + threadID[2] * block_size[1]
    block_offset = CartesianIndex(blockID[1] * block_size[1], blockID[2] * block_size[2])


    linear_global = id
    while linear_global < last_index
        linear2d = linear_global
        out_y = div(linear2d, index_region[1])
        
        linear1d = linear2d % index_region[1]
        out_x = linear1d
        
        shared_index = CartesianIndex(out_x + 1, out_y + 1)
        global_index = block_offset + CartesianIndex(stencil_dims[1][1], stencil_dims[2][1]) + shared_index

        print(global_index)
        arr[out_index] = 1

        global linear_global += thread_jump
    end

    print(arr)

    linear3d = linear_global
    out_3d = div(linear3d, index_region[1] * index_region[2])

    linear2d = linear3d % (index_region[1] * index_region[2])
    out_y = div(linear2d, index_region[1])
        
    linear1d = linear2d % index_region[1]
    out_x = linear1d

# print("\n\n")

# global_pos = CartesianIndex(1, 1)
# local_linear = 1
# block_size = CartesianIndex(8, 8)
# grid_size = Tuple(10, 10)

# gmin = -2
# gmax = 2

# linear_jump = 1
# for dim in grid_size
#     linear_jump *= dim
# end

# D2_index = local_linear / grid_size[2]
# D1_index = local_linear % grid_size[2]
# res = CartesianIndex(D1_index, D2_index)

# print(D)

end