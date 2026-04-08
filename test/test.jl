using Polyester
using ThreadPinning
pinthreads(:cores)

dump(quote
@fastmath @inbounds begin
    p = t + s
end
end)


macro gen(expr)
    return esc(quote
        function mykernel(arr1, arr2, size, threads, k; LOWER_BOUND=(), UPPER_BOUND=())            
        @batch for id in 0:(threads-1)
            # Range calculations
            step = cld(size[1], threads)
            rng_low = LOWER_BOUND[1] + step*id
            rng_high = min(rng_low+step-1, UPPER_BOUND[1])
            start_index = CartesianIndex(rng_low, LOWER_BOUND[2])
            end_index = CartesianIndex(rng_high, UPPER_BOUND[2])

            # Main loopstart
            for ind in start_index:end_index                
                myvar = (arr1[ind] + arr1[ind + CartesianIndex(1,0)])
                arr2[ind] = myvar + k                
            end
        end 
    end
end)
end
@gen 1

mykernel(a, b, size(a), 6, 2; LOWER_BOUND=(1,1), UPPER_BOUND=(99, 100))
@time mykernel(a, b, size(a), 6, 2; LOWER_BOUND=(1,1), UPPER_BOUND=(99, 100))

function mykernel2(arr1, arr2, size, threads, k; LOWER_BOUND=(), UPPER_BOUND=())            
    @batch for id in 0:(threads-1)
        step = cld(size[1], threads)
        rng_low = LOWER_BOUND[1] + step*id
        rng_high = min(rng_low+step-1, UPPER_BOUND[1])
        start_index = CartesianIndex(rng_low, LOWER_BOUND[2])
        end_index = CartesianIndex(rng_high, UPPER_BOUND[2])
        range = start_index:end_index
        for ind in start_index:end_index                
            myvar = (arr1[ind] + arr1[ind + CartesianIndex(1,0)])::Int32
            arr2[ind] = myvar + k                
        end
    end  
end

# a = ones(Int32, 300,100)
# b = zeros(Int32, 300,100)

mykernel2(a, b, size(a), 6, 2; LOWER_BOUND=(1,1), UPPER_BOUND=(99, 100))
@time mykernel2(a, b, size(a), 6, 2; LOWER_BOUND=(1,1), UPPER_BOUND=(99, 100))
print("Success")
# println(b)