using GameTracer
using GameTheory
using Random
using Test

@testset "GameTracer.jl" begin
    gs = []

    g = NormalFormGame(Player([3 3; 2 5; 0 6]),
                       Player([3 2 3; 2 6 1]))
    push!(gs, g)

    g = NormalFormGame((2, 2, 2))
    g[1, 1, 1] = 9, 8, 12
    g[2, 2, 1] = 9, 8, 2
    g[1, 2, 2] = 3, 4, 6
    g[2, 1, 2] = 3, 4, 4
    push!(gs, g)

    @testset "ipa_solve" begin
        seed = 1234
        rng = MersenneTwister(seed)
        fuzz_default = 1e-6
        for g in gs
            res = @inferred ipa_solve(rng, g)
            @test is_nash(g, res.NE, tol=fuzz_default)

            fuzz = 1e-8
            res = @inferred ipa_solve(rng, g, fuzz=fuzz)
            @test is_nash(g, res.NE, tol=fuzz)
        end
    end

    @testset "gnm_solve" begin
        seed = 1234
        rng = MersenneTwister(seed)
        for g in gs
            res = @inferred gnm_solve(rng, g)
            @test length(res.NEs) == res.num_NEs
            for NE in res.NEs
                @test is_nash(g, NE,)
            end
        end
    end

    @testset "1-player game" begin
        g = NormalFormGame([[1], [2], [3]])
        @test_throws ArgumentError ipa_solve(g)
        @test_throws ArgumentError gnm_solve(g)
    end
end
