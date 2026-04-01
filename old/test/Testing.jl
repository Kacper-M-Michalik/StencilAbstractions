module Testing 

using CUDA
using KernelAbstractions
using CUDA.CUDAKernels

CUDA.allowscalar(false)

expr = @macroexpand @kernel function testkernel!(A, @Const(B), @Const(c))
    i = @index(Global, Cartesian)
    print(i)
    @inbounds A[i] = B[i] + c 
end

@kernel function copy_kernel!(A, @Const(B), @Const(c))
    i = @index(Global, Cartesian)
    print(i)
    @inbounds A[i] = B[i] + c 
end

a = CuArray(zeros(1,1))
backend = get_backend(a)
kernel! = copy_kernel!(backend)

#dump(code)
#print(kernel!)
print(expr)

print("\n\n\n")

expr2 = quote
    a = []
    for j in 1:10
        for i in 2:2
            a[i,j] = 1
        end
    end
end

dump(expr2)

function mycopy!(A, B, c)
    backend = get_backend(A)
    @assert size(A) == size(B)

    #kernel! = copy_kernel!(backend)
    #kernel!(A, B, c, ndrange = size(A))
    
    return backend
end

A = zeros(10, 2)
B = ones(10, 2)

#test = mycopy!(A, B, 5.0)
#KernelAbstractions.synchronize(test)

print(A)
print(B)

# print("\n")
# print(mod1(4, 3))
# print("\n")
# print(mod1(1, 3))
# print("\n")
# print(mod1(-2, 3))
# print("\n")
# print(mod1(-5, 3))

# print("\n")
# print(mod1(5, 3))
# print("\n")
# print(mod1(2, 3))
# print("\n")
# print(mod1(-1, 3))
# print("\n")
# print(mod1(-4, 3))

# print("\n")
# print(mod1(6, 3))
# print("\n")
# print(mod1(3, 3))
# print("\n")
# print(mod1(0, 3))
# print("\n")
# print(mod1(-3, 3))

# val = CartesianIndex(0, 2)
# dump(val)
# dump(val.I)
# dump(size(A))

# modulo = mod1.(val.I, size(A))
# dump(modulo)

end