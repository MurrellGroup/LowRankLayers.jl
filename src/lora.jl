struct LoRADense
    primary::Dense
    proj1::Dense
    proj2::Dense
end

"""
    LoRADense(primary::Dense, hidden_dim::Int)

Create a LoRA wrapper around a Dense layer. The second projection matrix is initialized to zero, and only the two projections (and not the primary layer) are trainable.
"""
function LoRADense(primary::Dense, hidden_dim::Int; init=Flux.kaiming_uniform())
    dim1 = size(primary.weight, 2)
    dim2 = size(primary.weight, 1)
    ld = LoRADense(
        primary,
        Dense(dim1 => hidden_dim, bias=false, init = init),
        Dense(hidden_dim => dim2, bias=false)
    )
    ld.proj2.weight .= 0 
    return ld
end 

function (lora::LoRADense)(x)
    return lora.primary(x) .+ lora.proj2(lora.proj1(x))
end

Flux.@layer :expand LoRADense trainable=(proj1, proj2)