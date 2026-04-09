function get_tile_size(spec, block_size, timesteps)
    get    
end

function compile_stencil(spec::StencilSpecification{CartesianGrid, B, D, OverlappedSettings}) where {B, D <: KernelAbstractions.GPU})
   
    timesteps = spec.settings.timesteps

    quote
        @kernel unsafe_indices=true function $(kernel_name)($(wrapped_args...), $_LOWER_BOUND, $_UPPER_BOUND)     

            # All inputs and outputs
            $(buffer_allocations)

            # For all outputs
            $(register_variables)

            $cursor_symbol = @index(Global, Cartesian)      
            $cursor_symbol += CartesianIndex($_LOWER_BOUND)

            $(load_loop)

            while $(time_var) < $(timesteps)
                


                @synchronize()
                # flush register vars to buffers
                @synchronize()
            end

            # for reach output, flush buffer to global memory
            if (all(($cursor_symbol).I .<= $_UPPER_BOUND) && TODO: TEST IF VALID TIME LOC)
                $() = $()[$]
            end
        end
        $(kernel_name)($(spec.device), $(spec.settings.block_size))
    end


end