module GameTracer

using GameTheory: NormalFormGame, GAMPayoffVector
using Random

# ------------------------------------------------------------------
# Library Path
# ------------------------------------------------------------------
# TODO: Remove hardcoded dylib path after Phase B
# Temporary: Using local dylib until Phase B is complete
const libgametracer = "/tmp/gametracer_prefix/lib/libgametracer.dylib"

# ------------------------------------------------------------------
# Public API & Result Types
# ------------------------------------------------------------------
export ipa_solve, gnm_solve


"""
    IPAResult

# Fields TBD

"""
struct IPAResult{N}
    NE::NTuple{N, Vector{Float64}}
end

"""
    GNMResult

# Fields TBD

"""
struct GNMResult{N}
    NEs::Vector{NTuple{N, Vector{Float64}}}
    num_NEs::Int
end


"""
    ipa_solve(g::NormalFormGame) -> IPAResult

# Arguments

# Keyword Arguments

# Returns

- IPAResult:

# References

"""
function ipa_solve(
    rng::AbstractRNG,
    g::NormalFormGame{N};
    ray::AbstractVector{<:Real} = rand(rng, sum(g.nums_actions)),
    zh::AbstractVector{<:Real} = ones(sum(g.nums_actions)),
    alpha::Real = 0.02,
    fuzz::Real = 1e-6,
) where {N}
    M = sum(g.nums_actions)
    actions = Cint[g.nums_actions...]
    p = GAMPayoffVector(Cdouble, g)
    ray = convert(Vector{Cdouble}, ray)
    zh = Vector{Cdouble}(zh)  # Copy
    out = Vector{Cdouble}(undef, M)
    out, ret = ipa!(
        N, actions, p.payoffs, ray, zh, Cdouble(alpha), Cdouble(fuzz), out
    )

    NE = _get_action_profile(out, g.nums_actions)

    return IPAResult(NE)
end

ipa_solve(g::NormalFormGame; kwargs...) = 
    ipa_solve(Random.GLOBAL_RNG, g; kwargs...)

function ipa_solve(rng::AbstractRNG, g::NormalFormGame{1}; kwargs...)
    throw(ArgumentError("not implemented for 1-player games"))
end


"""
    gnm_solve(g::NormalFormGame) -> GNMResult

# Arguments

# Keyword Arguments

# Returns

- GNMResult: 

# References

"""
function gnm_solve(
    rng::AbstractRNG,
    g::NormalFormGame{N};
    ray::AbstractVector{<:Real} = rand(rng, sum(g.nums_actions)),
    steps::Integer = 100,
    fuzz::Real = 1e-12,
    lnmfreq::Integer = 3,
    lnmmax::Integer = 10,
    lambdamin::Real = -10.0,
    wobble::Bool = false,
    threshold::Real = 1e-2
) where {N}
    actions = Cint[g.nums_actions...]
    p = GAMPayoffVector(Cdouble, g)
    ray = convert(Vector{Cdouble}, ray)
    answers, ret = gnm(
        N, actions, p.payoffs, ray,
        steps, Cdouble(fuzz), lnmfreq, lnmmax,
        Cdouble(lambdamin), wobble, Cdouble(threshold)
    )

    NEs = _get_action_profiles(answers, g.nums_actions)
    
    return GNMResult(NEs, Int(ret))
end

gnm_solve(g::NormalFormGame; kwargs...) = 
    gnm_solve(Random.GLOBAL_RNG, g; kwargs...)

function gnm_solve(rng::AbstractRNG, g::NormalFormGame{1}; kwargs...)
    throw(ArgumentError("not implemented for 1-player games"))
end


# ------------------------------------------------------------------
# Private API (C ABI wrappers)
# ------------------------------------------------------------------

function ipa!(
    N::Integer,
    actions::Vector{Cint},
    payoffs::Vector{Cdouble},
    ray::Vector{Cdouble},
    zh::Vector{Cdouble},
    alpha::Cdouble,
    fuzz::Cdouble,
    out::Vector{Cdouble}
)
    ret = ccall(
        (:ipa, libgametracer), Cint,
        (Cint, Ptr{Cint}, Ptr{Cdouble},
         Ptr{Cdouble}, Ptr{Cdouble},
         Cdouble, Cdouble,
         Ptr{Cdouble}),
        N, actions, payoffs,
        ray, zh,
        alpha, fuzz,
        out
    )

    ret <= 0 && error("IPA failed (ret = $ret)")

    return (out, ret)
end

function gnm(
    N::Integer,
    actions::Vector{Cint},
    payoffs::Vector{Cdouble},
    ray::Vector{Cdouble},
    steps::Integer,
    fuzz::Cdouble,
    lnmfreq::Integer,
    lnmmax::Integer,
    lambdamin::Cdouble,
    wobble::Integer,
    threshold::Cdouble,
)
    M = sum(actions)
    answers_ref = Ref{Ptr{Cdouble}}(C_NULL)

    ret = ccall(
        (:gnm, libgametracer), Cint,
        (Cint, Ptr{Cint}, Ptr{Cdouble},
         Ptr{Cdouble}, Ref{Ptr{Cdouble}},
         Cint, Cdouble, Cint, Cint, Cdouble, Cint, Cdouble),
        N, actions, payoffs,
        ray, answers_ref,
        steps, fuzz, lnmfreq, lnmmax, lambdamin, wobble, threshold
    )

    ret < 0 && error("GNM failed (ret = $ret)")

    # ret == 0: 0 equilibria, answers == NULL
    ret == 0 && return (Matrix{Cdouble}(undef, M, 0), ret)

    # ret > 0: num_eq equilibria, answers is malloc'd buffer
    ptr = answers_ref[]
    ptr != C_NULL || error("GNM returned ret=$ret but answers pointer was NULL")

    answers = try
        answers_view = unsafe_wrap(Array, ptr, (M, Int(ret)); own=false)
        copy(answers_view)
    finally
        ccall((:gametracer_free, libgametracer), Cvoid, (Ptr{Cvoid},), ptr)
    end

    return (answers, ret)
end

function _get_action_profile(x::AbstractVector{T},
                             nums_actions::NTuple{N,Integer}) where {N,T}
    out = ntuple(i -> Vector{T}(undef, nums_actions[i]), Val(N))
    ind = 1
    @inbounds for i in 1:N
        len = nums_actions[i]
        copyto!(out[i], 1, x, ind, len)
        ind += len
    end
    return out
end

function _get_action_profiles(x::AbstractMatrix{T},
                              nums_actions::NTuple{N,Integer}) where {N,T}
    num_NEs = size(x, 2)
    out = Vector{NTuple{N,Vector{T}}}(undef, num_NEs)
    @inbounds for j in 1:num_NEs
        out[j] = _get_action_profile(@view(x[:, j]), nums_actions)
    end
    return out
end

end
