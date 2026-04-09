module StencilAbstractionsCUDAExt

using StencilAbstractions
using CUDA
using CUDA.CUDAKernels

function StencilAbstractions.generate_graph_loop(func, count, backend::CUDABackend, unroll_count)
    unrolled_func = Expr(:block)
    for i in 1:unroll_count
        for call in func.args
            push!(unrolled_func.args, call)
        end
    end

    base_iters = gensym(Symbol("base_iters"))
    iters = gensym(Symbol("iters"))
    graph = gensym(Symbol("graph"))
    exec_graph = gensym(Symbol("exec"))
    index = gensym(Symbol("i"))

    return esc(quote
        $base_iters = ($count)
        $iters = div($base_iters, $unroll_count)
        if ($base_iters > 0)
            if ($iters > 0)
                $graph = CUDA.capture() do
                    $unrolled_func
                end         
                $exec_graph = CUDA.instantiate($graph)
                for $index in 1:$iters
                    CUDA.launch($exec_graph)
                end
            end
            
            $graph = CUDA.capture() do
                $func
            end         
            $exec_graph = CUDA.instantiate($graph)
            for $index in 1:rem($base_iters, $unroll_count)
                CUDA.launch($exec_graph)           
            end        
            CUDA.synchronize()
        end        
    end)
end


    # return quote
    #     base_iters = $(esc(count))
    #     iters = div(base_iters, $(unroll_count))
    #     if (base_iters > 0)
    #         if (iters > 0)
    #             graph = CUDA.capture() do
    #                 $(esc(unrolled_func))
    #             end         
    #             exec_graph = CUDA.instantiate(graph)
    #             for t in 1:(iters)
    #                 CUDA.launch(exec_graph)
    #             end
    #         end
            
    #         graph = CUDA.capture() do
    #             $(esc(func))
    #         end         
    #         exec_graph = CUDA.instantiate(graph)
    #         for i in 1:rem(base_iters, $(unroll_count))
    #             CUDA.launch(exec_graph)           
    #         end        
    #         CUDA.synchronize()
    #     end        
    # end

    # base_iters = gensym(Symbol("base_iters"))
    # iters = gensym(Symbol("iters"))
    # graph = gensym(Symbol("graph"))
    # exec_graph = gensym(Symbol("exec"))
    # index = gensym(Symbol("i"))

    # return esc(quote
    #     $base_iters = $(esc(count))
    #     $iters = div($base_iters, $unroll_count)
    #     if ($base_iters > 0)
    #         if ($iters > 0)
    #             $graph = CUDA.capture() do
    #                 $(unrolled_func)
    #             end         
    #             $exec_graph = CUDA.instantiate($graph)
    #             for $index in 1:$iters
    #                 CUDA.launch($exec_graph)
    #             end
    #         end
            
    #         $graph = CUDA.capture() do
    #             $(func)
    #         end         
    #         $exec_graph = CUDA.instantiate($graph)
    #         for $index in 1:rem($base_iters, $unroll_count)
    #             CUDA.launch($exec_graph)           
    #         end        
    #         CUDA.synchronize()
    #     end        
    # end)

end