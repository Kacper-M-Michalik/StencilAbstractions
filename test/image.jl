@stencil function box_blur(A::Grid, B::Grid)
    sum = (A[-1, -1] + A[-1, 0] + A[-1, 1] + A[0, -1] + A[0,0] + A[0, 1] + A[1, -1] + A[1, 0] + A[1, 1])
    B = sum / 9.0
end

@stencil function box_blur_alt(A::Grid, B::grid) 
    B = mean(@box(A, 2, 1))
end