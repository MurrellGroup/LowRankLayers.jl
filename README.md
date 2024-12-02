# LowRankLayers

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/LowRankLayers.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/LowRankLayers.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/LowRankLayers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/LowRankLayers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/LowRankLayers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/LowRankLayers.jl)

This package provides a simple implementation of Low-Rank Adaptations (LoRAs) for Flux layers.

## Usage

```julia
using Flux, LowRankLayers

dense_layer = Dense(10 => 5)
lora_layer = LoRADense(dense_layer, 3) # hidden_dim = 3
```
