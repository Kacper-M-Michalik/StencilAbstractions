function compile_stencil(stencil::StencilEquation, settings::CartesianGrid{B, Haloless, D, S}, optimisations::OptimisationSettings) where {B <: GridBoundary, D <: KernelAbstractions.GPU, S <: GPUStrategy}
    
    @assert optimisations.warp_size == 32 || optimisations.warp_size == 64
    # TODO: Tile and block size checks

    # Generate hygenic symbols for kernel function name, etc.
    kernel_name = gensym(Symbol(stencil.name, "!"))
    cursor_symbol = gensym(Symbol("cursor"))
    id_symbol = gensym(Symbol("id"))

    # Mark kernel args as constant for KernelAbstractions
    # TODO: allow typing
    wrapped_args = Any[]
    for arg in stencil.args
        if (haskey(stencil.sources, arg))
            push!(wrapped_args, Expr(:macrocall, Symbol("@Const"), nothing, arg))
        else
            push!(wrapped_args, arg)
        end
    end    
    
    # Compile body    
    local transformed_body::Expr
    if (settings.strategy isa GatherComputeGPU)
        transformed_body = gpu_body_gather(stencil, settings, optimisations, cursor_symbol)
    else
        transformed_body = gpu_body_basic(stencil, settings, cursor_symbol)
    end    

    # TODO: Fastmath, inbounds conditional #unsafe_indices=true
    # Generate kernel, eval into function table, last line returns specialised KernelAbstractions callable struct
    generated_kernel = quote
        @kernel function $(kernel_name)($(wrapped_args...))                  
            $(transformed_body.args...)
            $_KERNEL_END
        end
        $(kernel_name)($(settings.backend), $(optimisations.block_size))
    end

    print("\n\nGenerated kernel:\n")
    print(generated_kernel)
    print("\n")

    specialised_kernel_ptr = eval(generated_kernel)
    return CompiledStencil(specialised_kernel_ptr, settings)
end


function gpu_body_basic(stencil::StencilEquation, settings::StencilSettings, cursor_symbol::Symbol)
    
    body = generic_compile(stencil.body, stencil, settings, cursor_symbol)
    @assert length(body) == 1

    index_block = quote
        $cursor_symbol = @index(Global, Cartesian)

        $(body[1].args...)
    end

    return index_block
end


# Kernel split into gather/process stage, gather stage performs coalesced reads into shared memory for reduced global memory reads
# Block size must be known at stencil compile time to declare correct shared memory arrays
function gpu_body_gather(stencil::StencilEquation, settings::StencilSettings, optimisation::OptimisationSettings, cursor_symbol::Symbol)

    index_calculations_body = quote
        block_ID = @index(Group, NTuple) .- 1
        ID = @index(Local, Linear) - 1
        local_ID = @index(Local, Cartesian)

        block_offset = CartesianIndex(block_ID .* $(optimisations.block_size))      
        out_index = block_offset + local_ID
        #@print("LOOP $(block_ID) $(block_offset.I) $(ID) $(local_ID.I) $(out_index.I)\n")
    end

    # Generate the gather phase
    # For each source array, we generate a shared memory array for the block, where an appropriately sized neighbourhood is gathered in a coalesced manner
    gather_body = Expr[]  
    array_to_shared = Dict{Symbol, Symbol}()   
    array_to_cursor = Dict{Symbol, CartesianIndex}()   
    id_jump = prod(block_size)
    
    for (key, source) in stencil.sources

        array_name = gensym(source.symbol)
        array_to_shared[source.symbol] = array_name

        # Currently don't believe we need to do bank shift, could as an option?
        shared_dims = MVector{stencil.dimensions, Int64}(undef)
        radii_mins = MVector{stencil.dimensions, Int64}(undef)
        for (i, radius) in enumerate(source.radii)
            shared_dims[i] = block_size[i] - radius.min + radius.max
            radii_mins[i] = radius.min
        end       

        index_region = (Int(ceil(shared_dims[1]/optimisations.warp_size))*optimisations.warp_size, shared_dims[2:end]...)
        last_index = prod(index_region)        

        # TODO Get rid off?
        @assert (stencil.dimensions > 0)

        index_calculations = Any[]
        out_symbols = Symbol[]
        for i in (stencil.dimensions):-1:1
            cur_index_symbol = Symbol("linear$(i)d")
            prev_index_symbol = Symbol("linear$(i+1)d")
            out_symbol = Symbol("out$(i)d")
            dim_mod = prod(index_region[1:i])
            dim_div = prod(index_region[1:(i-1)])

            push!(out_symbols, out_symbol)
            
            local index::Expr
            if (i == stencil.dimensions)
                index = quote
                    $cur_index_symbol = linear_global
                    $out_symbol = div($cur_index_symbol, $dim_div) + 1
                end
            elseif (i == 1)
                index = quote
                    $cur_index_symbol = $prev_index_symbol % $dim_mod
                    $out_symbol = $cur_index_symbol + 1
                end
            else
                index = quote
                    $cur_index_symbol = $prev_index_symbol % $dim_mod
                    $out_symbol = div($cur_index_symbol, $dim_div) + 1
                end
            end

            for line in index.args
                push!(index_calculations, line)
            end            
        end

        load_declaration = quote              
            #TODO -> cursor should be recalculated when decoupling tile size from block size, cursor must also be uniquely named
            cursor = local_ID - CartesianIndex($(radii_mins.data)) 
            $array_name = @localmem eltype($(source.symbol)) ($(shared_dims.data))
            
            linear_global = ID
            while linear_global < $last_index
                $(index_calculations...)

                shared_index = CartesianIndex($(reverse(out_symbols)...))
                global_index = block_offset + CartesianIndex($(radii_mins...)) + shared_index
            
                linear_global += $id_jump
                if ($(out_symbols[end]) > $(shared_dims[1])) continue end

                # TODO handle gather correctly
                $array_name[shared_index] = $(_perform_gather(source.symbol, :global_index, settings))
            end

            # @synchronize
            # if (ID == 0)
            #     for i in 1:$(shared_dims[1])
            #         for j in 1:$(shared_dims[2])
            #             @print(" ")
            #             @print($array_name[i, j]) 
            #         end
            #         @print("\n")
            #     end
            # end
        end

        push!(gather_body, load_declaration)
    end
    # Sync threads to ensure filled shared memory
    sync = :(@synchronize)

    # Convert the kernel body
    kernel_body = shared_mem_compile(stencil.body, array_to_shared, stencil, settings)
    @assert length(kernel_body) == 1

    # TODO: PASS SIZE
    kernel_body = quote
        if (out_index in CartesianIndices((1:4, 1:4)))
            @print("PROCESSING $(out_index.I) $(cursor.I)")
            $(kernel_body[1].args...)
        end
    end
    
    # Combine
    transformed_body = Expr(:block)
    for line in index_calculations_body.args
        push!(transformed_body.args, line)
    end
    for block in gather_body
        for line in block.args
            push!(transformed_body.args, line)
        end
    end    
    push!(transformed_body.args, sync)     
    for line in kernel_body.args
        push!(transformed_body.args, line)
    end

    return transformed_body
end

function shared_mem_compile(expr::Expr, array_to_shared, stencil::StencilEquation, settings::StencilSettings)
    if !(expr isa Expr)
        return Any[expr]
    end
    
    if (expr.head === :(=) && expr.args[1] isa Symbol && in(expr.args[1], stencil.output_args))
        @assert length(expr.args) == 2          
        out_arg = expr.args[1]

        assign_ref = Expr(:ref, out_arg, :out_index)
        @assert length(assign_ref.args) == 2
        
        assign_line = Expr(:(=), assign_ref) 
        @assert length(assign_line.args) == 1
        
        rhs_old = expr.args[2]
        if (rhs_old isa Expr)
            rhs = shared_mem_compile(rhs_old, array_to_shared, stencil, settings)
            @assert length(rhs) == 1
            push!(assign_line.args, rhs[1]) 
        else
            push!(assign_line.args, rhs_old)  
        end

        return Any[assign_line]     

    # Replace return with output array write and early return
    elseif (expr.head === :return)        
        @assert length(expr.args) == 1
        
        #TODO: Find correct output if return is multi arg
        assign_ref = Expr(:ref, stencil.output_args[1], :out_index)
        @assert length(assign_ref.args) == 2
        
        assign_line = Expr(:(=), assign_ref) 
        @assert length(assign_line.args) == 1
        
        rhs_old = expr.args[1]
        if (rhs_old !== nothing)
            if (rhs_old isa Expr)
                rhs = shared_mem_compile(rhs_old, array_to_shared, stencil, settings)
                @assert length(rhs) == 1
                push!(assign_line.args, rhs[1]) 
            else
                push!(assign_line.args, rhs_old)  
            end
            return Any[assign_line, :(@goto _kernel_end)]
        end
        
        return Any[:(@goto _kernel_end)]     

    # Modify array ref with appropriate bounds checked access
    elseif (expr.head === :ref)        
        symbol = expr.args[1]
        indices = expr.args[2:end]
        offset = Expr(:call, :CartesianIndex, indices...)       
        
        ref_new = Expr(:ref, array_to_shared[symbol], :(cursor + CartesianIndex($(indices...))))

        return Any[ref_new]
    
    # Perform standard array access
    # TODO: Mark as inbounds?
    elseif (expr.head === :macrocall && expr.args[1] == Symbol("@abs"))
        symbol = expr.args[3].args[1]
        indices = expr.args[3].args[2:end]
        offset = Expr(:call, :CartesianIndex, indices...)                    
        ref_new = Expr(:ref, symbol, offset)

        return Any[ref_new]

    # Otherwise copy expr
    else  
        new_expr = Expr(expr.head)
        for line in expr.args
            new_lines = shared_mem_compile(line, array_to_shared, stencil, settings)
            for new_line in new_lines
                push!(new_expr.args, new_line)
            end            
        end

        return Any[new_expr]
    end
end


# function shared_mem_compile(expr, output_array_name, stencil::StencilEquation, settings::StencilSettings)
#     if !(expr isa Expr)
#         return Any[expr]
#     end
    
#     if (expr.head === :return)                    
#         # Replace return with output array write and early return
#         @assert length(expr.args) == 1
        
#         assign_ref = Expr(:ref, output_array_name, :i)
#         @assert length(assign_ref.args) == 2
        
#         assign_line = Expr(:(=), assign_ref) 
#         @assert length(assign_line.args) == 1
#         if (expr.args[1] isa Expr)
#             rhs = generic_compile(expr.args[1], output_array_name, settings)
#             @assert length(rhs) == 1
#             push!(assign_line.args, rhs[1])
#         else
#             push!(assign_line.args, expr.args[1])
#         end                                    

#         return Any[assign_line, :(@goto _kernel_end)]     
#     elseif (expr.head === :ref)
#         # Replace array ref with shared memory access
#         symbol = expr.args[1]
#         indices = expr.args[2:end]
#         offset = Expr(:call, :CartesianIndex, indices...)      
        
#         # TODO CHANGE ACCESS HERE
#         ref_new = _perform_gather(symbol, offset, settings)   

#         return Any[ref_new]
#     else  
#         new_expr = Expr(expr.head)
#         for line in expr.args
#             new_lines = generic_compile(line, output_array_name, settings)
#             for new_line in new_lines
#                 push!(new_expr.args, new_line)
#             end            
#         end

#         return Any[new_expr]
#     end
# end