# Basic method does dispatch in rectangles
function internal_compile_stencil(spec::StencilSpecification{CartesianGrid, B, KernelAbstractions.CPU, BasicSettings{W}}) where {B, W}

    # Generate hygenic symbols for kernel function name, etc.
    kernel_name = gensym(Symbol(spec.stencil.name, "!"))
    caller_name = gensym(Symbol(kernel_name, "_caller"))   
    cursor_symbol = gensym(Symbol("cursor"))
    id_symbol = gensym(Symbol("id"))
    first_grid_index = spec.stencil.first_grid_index

    lower_offset = GetHaloLowerOffset(spec)
    upper_offset = GetHaloUpperOffset(spec)

    # Process args
    kernel_args = Any[]
    for arg in spec.stencil.args       
        if (arg[2] !== nothing)
            push!(kernel_args, Expr(:(::), arg[1], arg[2]))
        else
            push!(kernel_args, arg[1])            
        end
    end

    # Compile kernel body    
    body_array = gen_kernel(spec.stencil.body, spec, cursor_symbol, !spec.structure.halo, Dict{Symbol, Symbol}())
    @assert length(body_array) == 1
    body::Expr = body_array[1]

    # Generate iteration domain calculations  
    lower_access = Any[]
    upper_access = Any[]
    for i in 2:spec.stencil.dimensions
        push!(lower_access, :($(_LOWER_BOUND)[$(i)]))
        push!(upper_access, :($(_UPPER_BOUND)[$(i)]))
    end  

    start_index = gensym(Symbol("start"))
    end_index = gensym(Symbol("end"))        
    step = gensym(Symbol("step"))
    rng_low = gensym(Symbol("lower"))
    rng_high = gensym(Symbol("high"))     
    
    # Apply optimisation flags
    if (spec.settings.enable_fastmath)
        # fastmath
        body = Expr(:macrocall, Symbol("@fastmath"), nothing, body)
    end
    if (!spec.settings.enable_bounds_checks)
        # inbounds
        body = Expr(:macrocall, Symbol("@inbounds"), nothing, body)
    end

    # TODO: simd? or tturbo?
    generated_kernel = quote        
        function $kernel_name($(kernel_args...), $(_LOWER_BOUND), $(_UPPER_BOUND), $(_SIZE_SYMBOL), $(_THREADS_SYMBOL))            
            $(step) = cld($(_SIZE_SYMBOL)[1], $(_THREADS_SYMBOL))

            @batch for $(id_symbol) = 0:($(_THREADS_SYMBOL)-1)
                # Range calculations                    
                $(rng_low) =  $(_LOWER_BOUND)[1] + $(step)*$(id_symbol)
                $(rng_high) = min($(rng_low)+$(step)-1, $(_UPPER_BOUND)[1])
                $(start_index) = CartesianIndex($(rng_low), $(lower_access...))
                $(end_index) = CartesianIndex($(rng_high), $(upper_access...))

                # Main loops
                for $(cursor_symbol) in $(start_index):$(end_index)
                    $(body)
                end
            end   
        end
    end

    # Kernel wrapped in block by default
    generated_kernel = generated_kernel.args[2]
    
    # TODO: drop adding offset, do in global frame?
    local kernel_caller
    if (spec.structure.halo)
        kernel_caller = quote
            function $(caller_name)($(kernel_args...), lower, upper, threads, block, order)                
                if (length(lower) != 0)
                    @assert length(lower) == $(spec.stencil.dimensions)
                    lower = lower .+ $(lower_offset)
                else
                    lower = $(lower_offset) .+ 1
                end 

                if (length(upper) != 0)
                    # TODO: Add actual size checks?
                    @assert length(upper) == $(spec.stencil.dimensions)
                    upper = upper .+ $(lower_offset)
                else
                    upper = size($(kernel_args[first_grid_index])) .- $(upper_offset)                    
                end          

                if (threads == 0)
                    threads = Threads.nthreads()
                end

                range = (upper .- lower) .+ 1
                $(kernel_name)($(kernel_args...), lower, upper, range, threads)
            end
        end
    else
        kernel_caller = quote
            function $(caller_name)($(kernel_args...); lower, upper, threads, block, order)                
                if length(lower) != 0
                    @assert length(lower) == $(spec.stencil.dimensions)
                else
                    lower = ntuple(val -> 1, $(spec.stencil.dimensions))
                end 

                if length(upper) != 0
                    # TODO: Add actual size checks?
                    @assert length(upper) == $(spec.stencil.dimensions)
                else
                    upper = size($(kernel_args[first_grid_index]))                  
                end          

                if (threads == 0)
                    threads = $(spec.settings.thread_count)
                end

                # TODO, Two kernels, figorue out edge dispatch
                range = (upper .- lower) .+ 1
                $(kernel_name)($(kernel_args...), lower, upper, range, threads)
            end
        end
    end

    return CompilationResult(kernel_caller, generated_kernel, caller_name, kernel_name, spec)
end