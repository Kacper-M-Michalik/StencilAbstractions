
# @parallel_indices (ix, iy) function box_blur_ps(A::Data.Array, B::Data.Array)
#     sum = (A[ix-1, iy-1] + A[ix-1, iy] + A[ix-1, iy+1] + A[ix, iy-1] + A[ix, iy] + A[ix, iy+1] + A[ix+1, iy-1] + A[ix+1, iy] + A[ix+1, iy+1])
#     B[ix, iy] = sum / 9.0
#     return
# end

# @stencil function gaussian_blur_2d(A::Grid, B::Grid) 
#     sum = (A[-1, -1] + A[-1, 1] + A[1, -1] + A[1, 1])
#     sum += 2.0 * (A[-1, 0] + A[0, -1] + A[0, 1] + A[1, 0])
#     sum += 4.0 * A[0, 0]
#     B = sum / 16.0
# end
@stencil function box_blur(A::Grid, B::Grid)
    sum = (A[-1, -1] + A[-1, 0] + A[-1, 1] + A[0, -1] + A[0,0] + A[0, 1] + A[1, -1] + A[1, 0] + A[1, 1])
    B = sum / 9.0
    end

@stencil function box_blur_alt(A::Grid, B::grid) 
    B = mean(@box(A, 2, 1))
end