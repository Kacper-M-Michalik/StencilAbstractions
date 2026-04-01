# Stencil transformations are performed here:
# 1. Replace returns
# 2. Replace input refs
# 3. Generate distribution/dispathc methods
#, lower_corner::NTuple{N, Int}=(), upper_corner::NTuple{N, Int}=()) where N

function compile_stencil(spec::StencilSpecification) 
    @assert spec.settings.warp_size == 32 || spec.settings.warp_size == 64
    #@assert spec.settings.tile_size == spec.settings.block_size
    # TODO max sure this holds true?

    # Generate hygenic symbols for kernel function name, etc.
    kernel_name = gensym(Symbol(stencil.name, "!"))
    caller_name = gensym(kernel_name, "_caller")
    cursor_symbol = gensym(Symbol("cursor"))
    id_symbol = gensym(Symbol("id"))

    # Mark kernel args as constant for KernelAbstractions
    # TODO: allow typing, insert missing outputs
    wrapped_args = Any[]
    for arg in stencil.args
        if (haskey(stencil.sources, arg))
            push!(wrapped_args, Expr(:macrocall, Symbol("@Const"), nothing, arg))
        else
            push!(wrapped_args, arg)
        end
    end    
    
    # Compile kernel    
    transformed_body::Expr = gen_jacobi_kernel(spec, cursor_symbol)

    # TODO: Fastmath, inbounds conditional #unsafe_indices=true
    # Generate kernel, eval into function table, last line returns specialised KernelAbstractions callable struct
    generated_kernel = quote
        @kernel function $(kernel_name)($(wrapped_args...), $_LOWER_BOUND, $_UPPER_BOUND)        
            $cursor_symbol = @index(Global, Cartesian)      
            $cursor_symbol += $_LOWER_BOUND

            if (all(($cursor_symbol).I .<= $_UPPER_BOUND))
                $(transformed_body.args...)               
            end 
            $_KERNEL_END

        end
        $(kernel_name)($(spec.backend), $(spec.settings.tile_size))
    end
    specialised_kernel_ptr = eval(generated_kernel)

    # TODO: Add actual size checks?
    kernel_caller = quote
        function $(caller_name)(args...; lower_corner=(), upper_corner=())     
                # TODO -> use output array, from it you can read size w/ padding and true size
                # alloc gets maximal bounds, then max with tile size (if enabled)

            if length(lower_corner) != 0
                @assert length(lower_corner) == $(spec.stencil.dimensions)
            else
                lower_corner = ntuple(i -> 0, $(spec.stencil.dimensions))
            end 
            lower_corner = lower_corner .+ bulk_start_index

            if length(upper_corner) != 0
                @assert length(lower_corner) == $(spec.stencil.dimensions)
            else
                upper_corner = size($()) .- bulk_end_index
            end           

            tile_count = cld.((upper_corner .- lower_corner), $(spec.settings.tile_size))  
            $(specialised_kernel_ptr)(args..., lower_corner, upper_corner; ndrange=tile_count)  
        end
    end
    caller_ptr = eval(kernel_caller)
    
    print("\n\nGenerated kernel:\n")
    print(generated_kernel)
    print("\nCaller:\n")
    print(kernel_caller)
    print("\n")

    return caller_ptr

    # specialised_kernel_ptr = eval(generated_kernel)
    # return CompiledStencil(specialised_kernel_ptr, settings)
end