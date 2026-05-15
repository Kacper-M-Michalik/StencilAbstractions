using KernelAbstractions
using Polyester
using CUDA
using StaticArrays
using StencilAbstractions

@stencil function conway(A::Grid, B::Grid) 
    cnt = (A[-1, -1] + A[-1, 0] + A[-1, 1] + A[0, -1] + A[0, 1] + A[1, -1] + A[1, 0] + A[1, 1])
    if (A[0, 0] == 1)
        if (cnt < 2 || cnt > 3) B = 0
        else B = 1
        end
    else
        if (cnt == 3) B = 1
        else B = 0
        end
    end
end

@compile_stencil(CONWAY, conway, ArrayBacking(), Halo(), CUDABackend(), OverlappedSettings(swap=[(:A, :B)], block_size=(32,8), timesteps=4))
@compile_stencil(CONWAY_ref, conway, ArrayBacking(), Halo(), CUDABackend(), BasicSettings(default_block_size=(32,8)))

test_size = (4,4)
init = Int32.(rand(Bool, test_size))
# Int32.([
#     1 0 0 1
#     1 1 0 0 
#     0 1 0 1 
#     0 0 1 0
# ])# 
in, range = alloc(CONWAY.settings, init) 
out, range = zeros(CONWAY.settings, eltype(init), test_size)

in2, r = alloc(CONWAY_ref.settings, init) 
out2, r = zeros(CONWAY_ref.settings, eltype(init), test_size)

println(range)
println("allocs complete")

# @kernel unsafe_indices = true cpu = false function congpu(@Const(A), B, LOWER::NTuple{2, Int}, UPPER::NTuple{2, Int})
#     @inbounds begin
#     #     lower_cart = CartesianIndex(LOWER)
#     #     upper_cart = CartesianIndex(UPPER)

#         block = CartesianIndex(((@index(Group, NTuple) .- 1) .* (30, 6)))
#         cursor = @index(Local, Cartesian) + CartesianIndex(1,1)
#         linid = @index(Local, Linear)

#         AT = @localmem eltype(A) (34, 10)
#         cart_grid = CartesianIndices((34, 10))

#         for linload = linid:256:340            
#             cart = cart_grid[linload]
#             AT[cart] = A[block + cart]           
#         end

#         @synchronize
#         Bv = AT[cursor]     

#         global_loc = block + cursor
#         eligible = (all(LOWER .<= global_loc.I) && all(global_loc.I .<= UPPER))

#         if (cursor == CartesianIndex(33,9)) 
#             for idx in CartesianIndex(1,1):CartesianIndex(34,10)                
#                 @print(AT[idx])
#                 if(idx.I[1] == 34) @print("\n") end
#             end
#         end
        
#         for t = 1:2 
#             if (eligible)
#                 cnt = AT[cursor + CartesianIndex(-1, -1)] + AT[cursor + CartesianIndex(-1, 0)] + AT[cursor + CartesianIndex(-1, 1)] + 
#                     AT[cursor + CartesianIndex(0, -1)] + AT[cursor + CartesianIndex(0, 1)] + 
#                     AT[cursor + CartesianIndex(1, -1)] + AT[cursor + CartesianIndex(1, 0)] + AT[cursor + CartesianIndex(1, 1)]
                
#                 if AT[cursor + CartesianIndex(0, 0)] == 1
#                     if cnt < 2 || cnt > 3
#                         Bv = 0
#                     else
#                         Bv = 1
#                     end
#                 else
#                     if cnt == 3
#                         Bv = 1
#                     else
#                         Bv = 0
#                     end
#                 end
#             end
#             @synchronize
#             if (eligible)
#                 AT[cursor] = Bv
#             end
#             @synchronize 
#             if (cursor == CartesianIndex(33,9)) 
#                 @print("\n")
#                 for idx in CartesianIndex(1,1):CartesianIndex(34,10)                
#                     @print(AT[idx])
#                     if(idx.I[1] == 34) @print("\n") end
#                 end
#             end
#             @synchronize
#         end
       
#         for linload = linid:256:340
#             cart = cart_grid[linload]
#             global_loc = (block + cart) 
#             if (all((2, 2) .<= cart.I) && all(cart.I .<= (31, 7)) && all(LOWER .<= global_loc.I) && all(global_loc.I .<= UPPER))
#                 B[global_loc] = AT[cart]
#             end
#         end
#     end
# end
    
# function caller(A::AbstractArray{<:Any, 2}, B::AbstractArray{<:Any, 2}, range::Union{NTuple{2, AbstractUnitRange{Int}}, Nothing})
#     lower = first.(range)
#     upper = last.(range)
#     range = cld.((upper .- lower) .+ 1, (30, 6)) .* (32, 8)
#     println(lower)
#     println(upper)
#     println(range)
#     (congpu)(CUDACore.CUDAKernels.CUDABackend(false, false), (32, 8))(A, B, lower, upper; ndrange = range)
# end

println("call start")
CONWAY_ref(in2, out2)
CONWAY_ref(out2, in2)
CONWAY_ref(in2, out2)
CONWAY_ref(out2, in2)
CONWAY(in, out; range=range)
# caller(in, out, range)

KernelAbstractions.synchronize(CUDABackend())
println("run done")

eql = out[range...] == in2[r...]
println("init")
println(init)
println("out")
println(Array(out)[range...])
println("ref")
println(Array(out2)[r...])
println(Array(in2)[r...])
println("equal $(eql)")