# Stencil transformations are performed here:
# 1. Replace returns
# 2. Replace input refs
# 3. Generate distribution/dispatch methods
# Block size is used not tile size in this basic compiler
function compile_stencil(spec::StencilSpecification{CartesianGrid, B, D}) where {B, D <: KernelAbstractions.GPU}
    @assert spec.settings.warp_size == 32 || spec.settings.warp_size == 64

    # Generate hygenic symbols for kernel function name, etc.
    kernel_name = gensym(Symbol(spec.stencil.name, "!"))
    caller_name = gensym(Symbol(kernel_name, "_caller"))
    cursor_symbol = gensym(Symbol("cursor"))
    first_grid_index = spec.stencil.first_grid_index

    # Mark kernel args as constant for KernelAbstractions
    wrapped_args = Any[]
    for arg in spec.stencil.args
        if (arg[1] in spec.stencil.input_symbols)           
            push!(wrapped_args, Expr(:macrocall, Symbol("@Const"), nothing, arg[1]))
        elseif (!(arg[2] in spec.stencil.output_symbols) && arg[2] !== nothing)
            push!(wrapped_args, Expr(Symbol("::"), arg[1], arg[2]))
        else
            push!(wrapped_args, arg[1])            
        end
    end    
    
    output_args = copy(spec.stencil.output_symbols)    
    while length(output_args) < spec.stencil.required_outputs || length(output_args) < 1
        output_symbol = gensym(Symbol("output"))
        push!(output_args, output_symbol)
        push!(wrapped_args, output_symbol)
    end

    if (first_grid_index == 0)
        first_grid_index = length(wrapped_args)
    end

    # Compile kernel body    
    body_array = gen_kernel(spec.stencil.body, spec, output_args, cursor_symbol, !spec.structure.halo, Dict{Symbol, Symbol}())
    @assert length(body_array) == 1
    body::Expr = body_array[1]

    # TODO: Fastmath, inbounds
    # Generate kernel, eval into function table, last line returns specialised KernelAbstractions callable struct
    generated_kernel = quote
        @kernel unsafe_indices=true function $(kernel_name)($(wrapped_args...), $_LOWER_BOUND, $_UPPER_BOUND)        
            $cursor_symbol = @index(Global, Cartesian)      
            $cursor_symbol += CartesianIndex($_LOWER_BOUND)
            
            if (all(($cursor_symbol).I .<= $_UPPER_BOUND))               
                $(body.args...)               
            end 
            $_KERNEL_END
        end
        $(kernel_name)($(spec.device), $(spec.settings.block_size))
    end
    specialised_kernel_ptr = eval(generated_kernel)
    
    lower_offset = GetHaloLowerOffset(spec)
    upper_offset = GetHaloUpperOffset(spec)

    # Generate the dispatcher function
    # Kernel abstractions automatically calculates the requried number of tiles
    local kernel_caller
    if (spec.structure.halo)
        kernel_caller = quote
            function $(caller_name)(args...; lower=(), upper=())     
                local offset
                if length(lower) != 0
                    @assert length(lower) == $(spec.stencil.dimensions)
                    offset = lower .+ $(lower_offset) .- 1
                else
                    offset = $(lower_offset)
                end 

                if length(upper) != 0
                    # TODO: Add actual size checks?
                    @assert length(upper) == $(spec.stencil.dimensions)
                    upper = upper .+ $(lower_offset)
                else
                    upper = size(args[$(first_grid_index)]) .- $(upper_offset)                    
                end          
                
                range = (upper .- offset)
                $(specialised_kernel_ptr)(args..., offset, upper; ndrange=range)  
            end
        end
    else
        kernel_caller = quote
            function $(caller_name)(args...; lower=(), upper=())     
                local offset
                if length(lower) != 0
                    @assert length(lower) == $(spec.stencil.dimensions)
                    offset = lower .- 1
                else
                    offset = ntuple(val -> 0, $(spec.stencil.dimensions))
                end 

                if length(upper) != 0
                    # TODO: Add actual size checks?
                    @assert length(upper) == $(spec.stencil.dimensions)
                else
                    upper = size(args[$(first_grid_index)])
                end          
                
                range = (upper .- offset)
                $(specialised_kernel_ptr)(args..., offset, upper; ndrange=range)  
            end
        end
    end    
    
    # print("\n\nGenerated kernel: ")
    # print(kernel_name)
    # print("\n")
    # print(generated_kernel)
    # print("\nCaller:\n")
    # print(kernel_caller)
    # print("\n")

    return eval(kernel_caller)
end