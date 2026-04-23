function get_tile_size(source, block_size)
    get    
end

# function compile_stencil(spec::StencilSpecification{ArrayBacking, B, D, OverlappedSettings}) where {B, D <: KernelAbstractions.GPU})
   
#     timesteps = spec.settings.timesteps

#     quote
#         @kernel unsafe_indices=true function $(kernel_name)($(wrapped_args...), $_LOWER_BOUND, $_UPPER_BOUND)     

#             # All inputs and outputs
#             $(buffer_allocations)

#             # For all outputs
#             $(register_variables)

#             $cursor_symbol = @index(Global, Cartesian)      
#             $cursor_symbol += CartesianIndex($_LOWER_BOUND)

#             $(load_loop)

#             while $(time_var) < $(timesteps)
                


#                 @synchronize()
#                 # flush register vars to buffers
#                 @synchronize()
#             end

#             # for reach output, flush buffer to global memory
#             if (all(($cursor_symbol).I .<= $_UPPER_BOUND) && TODO: TEST IF VALID TIME LOC)
#                 $() = $()[$]
#             end
#         end
#         $(kernel_name)($(spec.device), $(spec.settings.block_size))
#     end


# end

function internal_compile_stencil(spec::StencilSpecification{ArrayBacking, B, D, OverlappedSettings}) where {B, D <: KernelAbstractions.GPU}

    # Generate hygenic symbols for kernel function name, etc.
    kernel_name = gensym(Symbol(spec.stencil.name, "!"))
    caller_name = gensym(Symbol(kernel_name, "_caller"))   
    block_size = spec.settings.block_size
    eff_block_size = spec.settings.block_size .* spec.settings.elements_per_thread
    timesteps = spec.settings.timesteps

    # Process args   
    kernel_args = Any[]    
    stencil_args = Any[]
    for arg in spec.stencil.args
        if (in(arg[1], spec.stencil.input_symbols) && !in(arg[1], spec.stencil.output_symbols))           
            push!(kernel_args, Expr(:macrocall, Symbol("@Const"), nothing, arg[1]))
            push!(stencil_args, arg[1])
        elseif (!in(arg[1], spec.stencil.input_symbols) && !in(arg[1], spec.stencil.output_symbols) && arg[2] !== nothing)
            push!(kernel_args, Expr(:(::), arg[1], arg[2]))
            push!(stencil_args, Expr(:(::), arg[1], arg[2]))
        else
            push!(kernel_args, arg[1])            
            push!(stencil_args, arg[1])
        end
    end        
    caller_args = copy(stencil_args)
    setup_caller_args!(caller_args)
    
    push!(kernel_args, _LOWER_BOUND)
    push!(kernel_args, _UPPER_BOUND)

    # Check/get valid boundaries
    boundaries = get_boundaries(spec)

    shared_sizes = []
    max_timesteps = -1
    for (grid, source) in spec.stencil.sources
        shared_size, possible_timesteps = get_tile_size(source, eff_block_size)
        if (max_timesteps == -1 || possible_timesteps < max_timesteps) max_timesteps = possible_timesteps end
        push!(shared_sizes, shared_size)
    end

    if (timesteps > max_timesteps)
        error("")
    end

    logical_tile_size = 0

end





# function internal_compile_stencil(spec::StencilSpecification{CartesianGrid, B, D, BasicSettings{W}}) where {B, D <: KernelAbstractions.GPU, W}

#     # Generate hygenic symbols for kernel function name, etc.
#     kernel_name = gensym(Symbol(spec.stencil.name, "!"))
#     caller_name = gensym(Symbol(kernel_name, "_caller"))   
#     cursor_symbol = gensym(Symbol("cursor"))
#     first_grid_index = spec.stencil.first_grid_index

#     # Mark kernel args as constant for KernelAbstractions
#     wrapped_args = Any[]    
#     kernel_args = Any[]
#     for arg in spec.stencil.args
#         if (in(arg[1], spec.stencil.input_symbols) && !in(arg[1], spec.stencil.output_symbols))           
#             push!(wrapped_args, Expr(:macrocall, Symbol("@Const"), nothing, arg[1]))
#             push!(kernel_args, arg[1])
#         elseif (!in(arg[1], spec.stencil.input_symbols) && !in(arg[1], spec.stencil.output_symbols) && arg[2] !== nothing)
#             push!(wrapped_args, Expr(:(::), arg[1], arg[2]))
#             push!(kernel_args, Expr(:(::), arg[1], arg[2]))
#         else
#             push!(wrapped_args, arg[1])            
#             push!(kernel_args, arg[1])
#         end
#     end        

#     # Compile kernel body    
#     body_array = gen_kernel(spec.stencil.body, spec, cursor_symbol, !spec.structure.halo, Dict{Symbol, Symbol}())
#     @assert length(body_array) == 1
#     body::Expr = body_array[1]

#     # Apply optimisation flags
#     if (spec.settings.enable_fastmath)
#         # fastmath
#         body = Expr(:macrocall, Symbol("@fastmath"), nothing, body)
#     end
#     if (!spec.settings.enable_bounds_checks)
#         # inbounds
#         body = Expr(:macrocall, Symbol("@inbounds"), nothing, body)
#     end

#     # Generate kernel, eval into function table
#     generated_kernel = quote
#         @kernel unsafe_indices=true function $(kernel_name)($(wrapped_args...), $_LOWER_BOUND, $_UPPER_BOUND)        
#             $cursor_symbol = @index(Global, Cartesian)      
#             $cursor_symbol += CartesianIndex($_LOWER_BOUND)
            
#             if (all(($cursor_symbol).I .<= $_UPPER_BOUND))               
#                 $(body.args...)               
#             end 
#         end        
#     end
    
#     lower_offset = GetHaloLowerOffset(spec)
#     upper_offset = GetHaloUpperOffset(spec)

#     # Generate the dispatcher function
#     # Kernel abstractions automatically calculates the requried number of tiles
#     local kernel_caller
#     if (spec.structure.halo)
#         kernel_caller = quote
#             function $(caller_name)($(kernel_args...), lower, upper, threads, block, order)
#                 local offset
#                 if length(lower) != 0
#                     @assert length(lower) == $(spec.stencil.dimensions)
#                     offset = lower .+ $(lower_offset) .- 1
#                 else
#                     offset = $(lower_offset)
#                 end 

#                 if length(upper) != 0
#                     # TODO: Add actual size checks?
#                     @assert length(upper) == $(spec.stencil.dimensions)
#                     upper = upper .+ $(lower_offset)
#                 else
#                     upper = size($(kernel_args[first_grid_index])) .- $(upper_offset)                    
#                 end          

#                 if length(block) == 0
#                     block = $(spec.settings.default_block_size)
#                 end
                
#                 range = (upper .- offset)
#                 $(kernel_name)($(spec.device), block)($(kernel_args...), offset, upper; ndrange=range)  
#             end
#         end
#     else
#         kernel_caller = quote
#             function $(caller_name)($(kernel_args...), lower, upper, threads, block, order)
#                 local offset
#                 if length(lower) != 0
#                     @assert length(lower) == $(spec.stencil.dimensions)
#                     offset = lower .- 1
#                 else
#                     offset = ntuple(val -> 0, $(spec.stencil.dimensions))
#                 end 

#                 if length(upper) != 0
#                     # TODO: Add actual size checks?
#                     @assert length(upper) == $(spec.stencil.dimensions)
#                 else
#                     upper = size($(kernel_args[first_grid_index]))
#                 end          
                
#                 if length(block) == 0
#                     block = $(spec.settings.default_block_size)
#                 end

#                 range = (upper .- offset)
#                 $(kernel_name)($(spec.device), block)($(kernel_args...), offset, upper; ndrange=range)  
#             end
#         end
#     end    
    
#     return CompilationResult(kernel_caller, generated_kernel, caller_name, kernel_name, spec)
# end