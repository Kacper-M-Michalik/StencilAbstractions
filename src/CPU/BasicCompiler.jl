# Basic method does dispatch in rectangles
function compile_stencil(spec::StencilSpecification{CartesianGrid, B, KernelAbstractions.CPU, BasicSettings}) where {B}
    @assert spec.settings.thread_count > 0

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
            push!(kernel_args, Expr(Symbol("::"), arg[1], arg[2]))
        else
            push!(kernel_args, arg[1])            
        end
    end

    output_args = copy(spec.stencil.output_symbols)    
    while length(output_args) < spec.stencil.required_outputs || length(output_args) < 1
        output_symbol = gensym(Symbol("output"))
        push!(output_args, output_symbol)
        push!(kernel_args, output_symbol)
    end

    if (first_grid_index == 0)
        first_grid_index = length(kernel_args)
    end
        
    # Add aliasing hints
    const_inputs = Expr(:block)
    const_replacements = Dict{Symbol, Symbol}()
    for arg in spec.stencil.input_symbols
        const_name = gensym(arg)
        const_replacements[arg] = const_name
        push!(const_inputs.args, :($(const_name) = Base.Experimental.Const($(arg))))
    end

    # Compile kernel body    
    body_array = gen_kernel(spec.stencil.body, spec, output_args, cursor_symbol, !spec.structure.halo, const_replacements)
    @assert length(body_array) == 1
    body::Expr = body_array[1]
    push!(body.args, _KERNEL_END)

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

    cursor_assignment = Expr(:(=), cursor_symbol, :($(start_index):$(end_index)))
    main_loops = Expr(:for, cursor_assignment, body)    
    
    # Apply optimisation flags
    function_tags = Symbol[]
    if (spec.settings.enable_fastmath)
        # fastmath
        push!(function_tags, Symbol("@fastmath"))
    end
    if (!spec.settings.enable_bounds_checks)
        # inbounds
        main_loops = Expr(:macrocall, Symbol("@inbounds"), nothing, Expr(:block, main_loops))
    end

    # TODO: simd?
    generated_kernel = quote        
        function $kernel_name($(kernel_args...), $(_LOWER_BOUND), $(_UPPER_BOUND), $(_SIZE_SYMBOL))            
            # Hint that inputs do not alias
            Base.Experimental.@aliasscope begin
                $(const_inputs.args...)

                Threads.@threads for $(id_symbol) = 0:$(spec.settings.thread_count-1)
                    # Range calculations
                    $(step) = cld($(_SIZE_SYMBOL)[1], $(spec.settings.thread_count))
                    $(rng_low) =  $(_LOWER_BOUND)[1] + $(step)*$(id_symbol)
                    $(rng_high) = min($(rng_low)+$(step)-1, $(_UPPER_BOUND)[1])
                    $(start_index) = CartesianIndex($(rng_low), $(lower_access...))
                    $(end_index) = CartesianIndex($(rng_high), $(upper_access...))

                    # Main loops
                    $(main_loops)
                end
            end      
        end
    end

    # Add top level function macros, Kernel wrapped in block by default
    generated_kernel = generated_kernel.args[2]
    for tag in function_tags
        generated_kernel = Expr(:macrocall, tag, nothing, generated_kernel)
    end    
    specialised_kernel_ptr = eval(generated_kernel)
    
    local kernel_caller
    if (spec.structure.halo)
        kernel_caller = quote
            function $(caller_name)($(kernel_args...); lower=(), upper=())                
                if length(lower) != 0
                    @assert length(lower) == $(spec.stencil.dimensions)
                    lower = lower .+ $(lower_offset)
                else
                    lower = $(lower_offset) .+ 1
                end 

                if length(upper) != 0
                    # TODO: Add actual size checks?
                    @assert length(upper) == $(spec.stencil.dimensions)
                    upper = upper .+ $(lower_offset)
                else
                    upper = size($(kernel_args[first_grid_index])) .- $(upper_offset)                    
                end          

                range = (upper .- lower) .+ 1
                $(specialised_kernel_ptr)($(kernel_args...), lower, upper, range)
            end
        end
    else
        kernel_caller = quote
            function $(caller_name)($(kernel_args...); lower=(), upper=())                
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

                range = (upper .- lower) .+ 1
                $(specialised_kernel_ptr)($(kernel_args...), lower, upper, range)
            end
        end
    end

    print("\n\nGenerated kernel: ")
    print(kernel_name)
    print("\n")
    print(generated_kernel)
    print("\nCaller:\n")
    print(kernel_caller)
    print("\n")

    return CompiledStencil(eval(kernel_caller), spec)
end