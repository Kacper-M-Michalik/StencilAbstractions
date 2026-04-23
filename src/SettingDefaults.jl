function correct_default(spec::StencilSpecification{ArrayBacking, B, KernelAbstractions.CPU, BasicSettings{W, T}}) where {B, W, T}
    return spec.settings
end
function correct_default(spec::StencilSpecification{ArrayBacking, B, KernelAbstractions.CPU, SpatialSettings{W, T}}) where {B, W, T}
    target_size = 8192
    # TODO, extract from stencil
    element_size = 32
    if (spec.settings.default_tile_size == ())
        
    elseif (length(spec.settings.default_tile_size) != spec.stencil.dimensions)
        error("Mismatched tile dimensions")
    end
    return spec.settings
end
function correct_default(spec::StencilSpecification{ArrayBacking, B, D, BasicSettings{W, T}}) where {B, D <: KernelAbstractions.GPU, W, T}

end
function correct_default(spec::StencilSpecification{ArrayBacking, B, D, SpatialSettings{W, T}}) where {B, D <: KernelAbstractions.GPU, W, T}

end
# TODO overlapped