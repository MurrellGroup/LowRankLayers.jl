struct LoRADense{P0<:Dense,P1<:Dense,P2<:Dense}
    primary::P0
    proj1::P1
    proj2::P2
end

"""
    LoRADense(primary::Dense, hidden_dim::Int; init=Flux.kaiming_uniform())

Create a LoRA wrapper around a Dense layer. The second projection matrix is initialized to zero,
and only the two projections (and not the primary layer) are trainable.
"""
function LoRADense(primary::Dense, hidden_dim::Int; init=Flux.kaiming_uniform())
    dim2, dim1 = size(primary.weight)
    ld = LoRADense(
        primary,
        Dense(dim1 => hidden_dim, bias=false, init = init),
        Dense(hidden_dim => dim2, bias=false)
    )
    ld.proj2.weight .= 0 
    return ld
end 

(lora::LoRADense)(x) = lora.primary(x) .+ lora.proj2(lora.proj1(x))

Flux.@layer :expand LoRADense trainable=(proj1, proj2)
