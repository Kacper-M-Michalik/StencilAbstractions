module StencilAbstractionsAMDGPUExt

using StencilAbstractions
using AMDGPU

function StencilAbstractions.generate_graph_loop(func, count, backend::ROCBackend, unroll_count)
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
                $graph = AMDGPU.capture() do
                    $unrolled_func
                end         
                $exec_graph = AMDGPU.instantiate($graph)
                for $index in 1:$iters
                    AMDGPU.launch($exec_graph)
                end
            end
            
            $graph = AMDGPU.capture() do
                $func
            end         
            $exec_graph = AMDGPU.instantiate($graph)
            for $index in 1:rem($base_iters, $unroll_count)
                AMDGPU.launch($exec_graph)           
            end        
            AMDGPU.synchronize()
        end        
    end)
end

end