# Dispatch in tiles, inplace
function internal_compile_stencil(spec::StencilSpecification{CartesianGrid, B, KernelAbstractions.CPU, SpatialSettings{Inplace_UNSAFE}}) where {B}
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

    # TODO: Allow non-first input?
    output_args = copy(spec.stencil.output_symbols)   
    @assert length(spec.stencil.input_symbols) > 0 "Inplace_UNSAFE stencils require at least one named @Input, as they write to it." 
    @assert first_grid_index > 0
    @assert length(output_args) == 0 "Inplace_UNSAFE stencils cannot have any named outputs, as they write to their first @Input." 
    @assert spec.stencil.required_outputs == 0 "Inplace_UNSAFE stencils don't support @returns, as they write to only their first @Input." 
    push!(output_args, spec.stencil.input_symbols[1])

    # Add aliasing hints
    const_inputs = Expr(:block)
    const_replacements = Dict{Symbol, Symbol}()
    for (i, arg) in enumerate(spec.stencil.input_symbols)
        if (i > 1)
            const_name = gensym(arg)
            const_replacements[arg] = const_name
            push!(const_inputs.args, :($(const_name) = Base.Experimental.Const($(arg))))
        end
    end

    # Compile kernel body    
    body_array = gen_kernel(spec.stencil.body, spec, output_args, cursor_symbol, !spec.structure.halo, const_replacements)
    @assert length(body_array) == 1
    body::Expr = body_array[1]
    push!(body.args, _KERNEL_END)

    # Generate iteration domain calculations  
    linear_limit = gensym(Symbol("linear_limit"))
    linear_index = gensym(Symbol("linear_index"))
    base_loc = gensym(Symbol("base_loc"))
    upper_loc = gensym(Symbol("upper_loc"))
    tile_id = gensym(Symbol("tile_id"))
    
    # Apply optimisation flags
    function_tags = Symbol[]
    if (spec.settings.enable_fastmath)
        # fastmath
        push!(function_tags, Symbol("@fastmath"))
    end
    if (!spec.settings.enable_bounds_checks)
        # inbounds
        body = Expr(:macrocall, Symbol("@inbounds"), nothing, body)
    end

    generated_kernel = quote        
        function $kernel_name($(kernel_args...), $(_LOWER_BOUND), $(_UPPER_BOUND), $(_SIZE_SYMBOL))            
            # Hint that inputs do not alias
            Base.Experimental.@aliasscope begin
                $(const_inputs.args...)

                Threads.@threads for $(id_symbol) = 1:$(spec.settings.thread_count)
                    # Range calculations
                    $(linear_limit) = prod($(_SIZE_SYMBOL))
                    for $(linear_index) in $(id_symbol):$(spec.settings.thread_count):$(linear_limit)
                        $(tile_id) = CartesianIndices($(_SIZE_SYMBOL))[$(linear_index)]
                        $(base_loc) = CartesianIndex(($(tile_id).I .- 1) .* $(spec.settings.tile_size) .+ $(_LOWER_BOUND))
                        $(upper_loc) = CartesianIndex(min.($(base_loc).I .+ $(spec.settings.tile_size) .- 1, $(_UPPER_BOUND)))
                        for $(cursor_symbol) in $(base_loc):$(upper_loc)
                            $(body.args...)
                        end
                    end
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
    
    kernel_caller = gen_spatial_caller(spec, caller_name, kernel_args, first_grid_index, lower_offset, upper_offset, specialised_kernel_ptr)    

    # print("\n\nGenerated kernel: ")
    # print(kernel_name)
    # print("\n")
    # print(generated_kernel)
    # print("\nCaller:\n")
    # print(kernel_caller)
    # print("\n")

    return CompiledStencil(eval(kernel_caller), spec)
end