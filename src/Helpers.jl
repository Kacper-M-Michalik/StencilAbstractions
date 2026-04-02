function GetMaximalHaloSizes(spec::StencilSpecification)
    halos = [StencilDimension() for _ in 1:spec.stencil.dimensions]

    for source in values(spec.stencil.sources)
        for (i, radius) in enumerate(source.radii)
            if radius.min < halos[i].min && radius.min < 0
                halos[i].min = radius.min
            end
            if radius.max > halos[i].max && radius.max > 0
                halos[i].max = radius.max
            end
        end
    end

    return halos
end

function GetHaloLowerOffset(spec::StencilSpecification)
    offset = zeros(Int, spec.stencil.dimensions)

    for source in values(spec.stencil.sources)
        for (i, radius) in enumerate(source.radii)
            if radius.min < offset[i]
                offset[i] = -radius.min
            end
        end
    end

    return Tuple(offset)
end

function GetHaloUpperOffset(spec::StencilSpecification)
    offset = zeros(Int, spec.stencil.dimensions)

    for source in values(spec.stencil.sources)
        for (i, radius) in enumerate(source.radii)
            if radius.max > offset[i]
                offset[i] = radius.max
            end
        end
    end

    return Tuple(offset)
end