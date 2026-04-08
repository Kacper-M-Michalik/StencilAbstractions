function gen_spatial_caller(spec::StencilSpecification, caller_name, kernel_args, first_grid_index, lower_offset, upper_offset, specialised_kernel_ptr)
    
end

# Dispatch in tiles
function internal_compile_stencil(spec::StencilSpecification{CartesianGrid, B, KernelAbstractions.CPU, SpatialSettings{Copy}}) where {B}

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
        
    # # Add aliasing hints
    # const_inputs = Expr(:block)
    # const_replacements = Dict{Symbol, Symbol}()
    # for arg in spec.stencil.input_symbols
    #     const_name = gensym(arg)
    #     const_replacements[arg] = const_name
    #     push!(const_inputs.args, :($(const_name) = Base.Experimental.Const($(arg))))
    # end

    # Generate iteration domain calculations  
    linear_limit = gensym(Symbol("linear_limit"))
    linear_index = gensym(Symbol("linear_index"))
    base_loc = gensym(Symbol("base_loc"))
    upper_loc = gensym(Symbol("upper_loc"))
    tile_id = gensym(Symbol("tile_id"))
    
    # Apply optimisation flags
    if (spec.settings.enable_fastmath)
        # fastmath
        body = Expr(:macrocall, Symbol("@fastmath"), nothing, body)
    end
    if (!spec.settings.enable_bounds_checks)
        # inbounds
        body = Expr(:macrocall, Symbol("@inbounds"), nothing, body)
    end

    # TODO: simd?
    generated_kernel = quote        
        function $kernel_name($(kernel_args...), $(_LOWER_BOUND), $(_UPPER_BOUND), $(_SIZE_SYMBOL))            
            # Hint that inputs do not alias
            Base.Experimental.@aliasscope begin
                #$(const_inputs.args...)

                #@batch for $(id_symbol) = 1:$(spec.settings.thread_count)
                    # Range calculations
                    $(linear_limit) = prod($(_SIZE_SYMBOL))
                    #for $(linear_index) in $(id_symbol):$(spec.settings.thread_count):$(linear_limit)
                    @batch for $(linear_index) in 1:$(linear_limit)
                        $(tile_id) = CartesianIndices($(_SIZE_SYMBOL))[$(linear_index)]
                        $(base_loc) = CartesianIndex(($(tile_id).I .- 1) .* $(spec.settings.tile_size) .+ $(_LOWER_BOUND))
                        $(upper_loc) = CartesianIndex(min.($(base_loc).I .+ $(spec.settings.tile_size) .- 1, $(_UPPER_BOUND)))
                        for $(cursor_symbol) in $(base_loc):$(upper_loc)
                            $(body.args...)
                        end
                    end
                #end
            end      
        end
    end

    # Add top level function macros, Kernel wrapped in block by default
    generated_kernel = generated_kernel.args[2]   

    # TODO: drop adding offset, do in global frame?
    local kernel_caller
    if (spec.structure.halo)
        kernel_caller = quote
            function $(caller_name)($(kernel_args...), lower, upper, threads, block, order)                
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

                range = cld.((upper .- lower) .+ 1, $(spec.settings.tile_size))
                $(kernel_name)($(kernel_args...), lower, upper, range)
            end
        end
    else
        kernel_caller = quote
            function $(caller_name)($(kernel_args...), lower, upper, threads, block, order)                
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

                range = cld.((upper .- lower) .+ 1, $(spec.settings.tile_size))
                $(kernel_name)($(kernel_args...), lower, upper, range)
            end
        end
    end

    return CompilationResult(kernel_caller, generated_kernel, caller_name, kernel_name, spec)
end