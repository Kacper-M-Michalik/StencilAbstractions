function generate_graph_loop(func, count, backend, unroll_count)
    unrolled_func = Expr(:block)
    for i in 1:unroll_count
        for call in func.args
            push!(unrolled_func.args, call)
        end
    end

    base_iters = gensym(Symbol("base_iters"))
    iters = gensym(Symbol("iters"))
    index = gensym(Symbol("i"))

    return esc(quote
        $base_iters = ($count)
        $iters = div($base_iters, $unroll_count)    
        for $index in 1:$iters
            $unrolled_func
        end            
        for $index in 1:rem($base_iters, $unroll_count)
            $func   
        end              
    end)
end