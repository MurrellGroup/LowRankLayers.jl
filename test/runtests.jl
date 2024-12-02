using LowRankLayers
using Flux
using Test

@testset "LowRankLayers.jl" begin

    @testset "LoRADense Layer Tests" begin

        @testset "Initialization" begin
            primary_layer = Dense(10 => 5)
            hidden_dim = 3
            lora_layer = LoRADense(primary_layer, hidden_dim)
    
            @test size(lora_layer.primary.weight) == (5, 10)
            @test size(lora_layer.proj1.weight) == (3, 10)
            @test size(lora_layer.proj2.weight) == (5, 3)
            @test all(iszero, lora_layer.proj2.weight)
        end

        @testset "Forward Pass" begin
            primary_layer = Dense(10 => 5)
            hidden_dim = 3
            lora_layer = LoRADense(primary_layer, hidden_dim)

            x = rand(Float32, 10)
            output = lora_layer(x)
    
            @test size(output) == (5,)
        end

        @testset "Trainability" begin
            primary_layer = Dense(10 => 5)
            hidden_dim = 3
            lora_layer = LoRADense(primary_layer, hidden_dim)
            @test Flux.trainable(lora_layer) == (; lora_layer.proj1, lora_layer.proj2)
        end

    end

end
