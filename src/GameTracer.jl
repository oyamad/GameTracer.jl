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

Result of [`ipa_solve`](@ref).

# Fields
- `NE::NTuple{N, Vector{Float64}}`: Mixed-action profile of a Nash equilibrium 
    computed by `ipa_solve`, stored as one probability vector per player.
- `ret_code::Int`: Raw positive return value from the underlying C function. 
    See `ipa_solve` for details.
"""
struct IPAResult{N}
    NE::NTuple{N, Vector{Float64}}
    ret_code::Int
end

"""
    GNMResult

Result of [`gnm_solve`](@ref).

# Fields
- `NEs::Vector{NTuple{N, Vector{Float64}}}`: Mixed-action profiles of Nash 
    equilibria computed by `gnm_solve`.
- `ret_code::Int`: Raw return value from the underlying C function. In the 
    current implementation, this is equal to the number of equilibria returned 
    in `NEs`. See `gnm_solve` for details.
"""
struct GNMResult{N}
    NEs::Vector{NTuple{N, Vector{Float64}}}
    ret_code::Int
end


"""
    ipa_solve(rng, g; 
              ray = rand(rng, sum(g.nums_actions)), 
              z_init = ones(sum(g.nums_actions)), 
              alpha = 0.02,
              fuzz = 1e-6)

Compute one mixed-action Nash equilibrium of `g` with the iterated polymatrix 
approximation (IPA) algorithm (Govindan and Wilson, 2004).

# Arguments
- `rng::AbstractRNG`: Random number generator used.
- `g::NormalFormGame`: A `NormalFormGame` instance with `N >=2` players.
- `ray::AbstractVector{<:Real}`: Pertubation ray. Its length must 
    correspond to `sum(g.nums_actions)`.
- `z_init::AbstractVector{<:Real}`: Initial point for the iteration. Its 
    length must correspond to `sum(g.nums_actions)`.
- `alpha::Real`: Step size parameter. Must satisfy `0 < alpha < 1`.
- `fuzz::Real`: Convergence tolerance for an equilibrium.

# Returns
- `res::IPAResult{N}`: Result object containing information about 
    the computed equilibrium and status code returned from the underlying C 
    routine. See [`IPAResult`](@ref) for details.

# Notes
* Pass an explicit `rng` or `ray` to obtain reproducible results.

# References
- S. Govindan and R. Wilson, "Computing Nash equilibria by iterated
  polymatrix approximation," Journal of Economic Dynamics and Control, 28 (2004),
  1229-1241.
"""
function ipa_solve(
    rng::AbstractRNG,
    g::NormalFormGame{N};
    ray::AbstractVector{<:Real} = rand(rng, sum(g.nums_actions)),
    z_init::AbstractVector{<:Real} = ones(sum(g.nums_actions)),
    alpha::Real = 0.02,
    fuzz::Real = 1e-6,
) where {N}
    M = sum(g.nums_actions)

    length(ray) == M ||
        throw(ArgumentError("length(ray) must equal sum(g.nums_actions)"))
    length(z_init) == M || 
        throw(ArgumentError("length(z_init) must equal sum(g.nums_actions)"))
    0 < alpha < 1 || 
        throw(ArgumentError("alpha must satisfy 0 < alpha < 1"))
    
    actions = Cint[g.nums_actions...]
    p = GAMPayoffVector(Cdouble, g)
    ray = convert(Vector{Cdouble}, ray)
    z_init = Vector{Cdouble}(z_init)  # Copy
    out = Vector{Cdouble}(undef, M)
    out, ret_code = ipa!(
        N, actions, p.payoffs, ray, z_init, Cdouble(alpha), Cdouble(fuzz), out
    )

    NE = _get_action_profile(out, g.nums_actions)

    return IPAResult(NE, Int(ret_code))
end

ipa_solve(g::NormalFormGame; kwargs...) = 
    ipa_solve(Random.GLOBAL_RNG, g; kwargs...)

function ipa_solve(rng::AbstractRNG, g::NormalFormGame{1}; kwargs...)
    throw(ArgumentError("not implemented for 1-player games"))
end


"""
    gnm_solve(rng, g;
              ray = rand(rng, sum(g.nums_actions)),
              steps = 100,
              fuzz = 1e-12,
              lnmfreq = 3,
              lnmmax = 10,
              lambdamin = -10.0,
              wobble = false,
              threshold = 1e-2)

Compute mixed-action Nash equilibria of `g` with the global Newton method (GNM) 
algorithm (Govindan and Wilson, 2003). 

# Arguments
- `rng::AbstractRNG`: Random number generator used.
- `g::NormalFormGame`: A `NormalFormGame` instance with `N >=2` players.
- `ray::AbstractVector{<:Real}`: Pertubation ray. Its length must 
    correspond to `sum(g.nums_actions)`.
- `steps::Integer`: Maximum number of steps; higher values of this parameter 
    slow GNM down, but may help it avoid getting off the path.
- `fuzz::Real`: Convergence tolerance for an equilibrium.
- `lnmfreq::Integer`: Frequency of the local Newton method (LNM) subroutines.
    Higher values decreases accumulated error.
- `lnmmax::Integer`: Maximum number of iterations within the LNM algorithm.
- `lambdamin::Real`: Minimum lambda value for the LNM algorithm. The algorithm
    terminates if lambda falls below this value. Must be negative.
- `wobble::Bool`: Whether to use "wobbles" of the perturbation vector to remove
    accumulated errors.
- `threshold::Real`: The equilibrium error tolerance for doing a wobble. If 
    wobbles are disabled, the GNM algorithm terminates if the error reaches this
    threshold.

# Returns
- `res::GNMResult{N}`: Result object containing information about 
    the computed equilibria and status code returned from the underlying C 
    routine. See [`GNMResult`](@ref) for details.

# Notes
* Pass an explicit `rng` or `ray` to obtain reproducible results.

# References
- S. Govindan and R. Wilson, "A global Newton method to compute Nash 
    equilibria," Journal of Economic Theory, 110 (2003), 65-86.
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
    M = sum(g.nums_actions)

    length(ray) == M ||
        throw(ArgumentError("length(ray) must equal sum(g.nums_actions)"))
    lambdamin < 0 || 
        throw(ArgumentError("lambdamin must be negative"))

    actions = Cint[g.nums_actions...]
    p = GAMPayoffVector(Cdouble, g)
    ray = convert(Vector{Cdouble}, ray)
    answers, ret_code = gnm(
        N, actions, p.payoffs, ray,
        steps, Cdouble(fuzz), lnmfreq, lnmmax,
        Cdouble(lambdamin), wobble, Cdouble(threshold)
    )

    NEs = _get_action_profiles(answers, g.nums_actions)
    
    return GNMResult(NEs, Int(ret_code))
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
