"""
    DFLimitIntelVectorMathExt

The package extension of DFLimit for IntelVectorMath (MKL's vector mathematics). Julia loads it when DFLimit and
IntelVectorMath are both loaded in a session, in either order. On x86-64 Linux and Windows the batched sums of the line
integral then take the exponential, sine and cosine of all their nodes at once from IntelVectorMath; on macOS, where the
engine takes them from Apple's Accelerate framework, and on other processors, the extension changes nothing.
`DFLimit.DFExact.vecmath()` reports the library in use.
"""
module DFLimitIntelVectorMathExt

using DFLimit: DFExact
import IntelVectorMath

# the engine's hook (DFExact.ivm_expsincos!): e = exp(a), sn = sin(b), cs = cos(b) for the first n entries
function DFExact.ivm_expsincos!(e::Vector{Float64}, sn::Vector{Float64}, cs::Vector{Float64}, a::Vector{Float64},
                                b::Vector{Float64}, n::Int)
    av = view(a, 1:n); bv = view(b, 1:n)
    IntelVectorMath.exp!(view(e, 1:n), av); IntelVectorMath.sin!(view(sn, 1:n), bv); IntelVectorMath.cos!(view(cs, 1:n), bv)
    nothing
end

# runs after the engine's __init__, which has looked for Accelerate: IntelVectorMath serves where Accelerate does not, on
# x86-64 (MKL's platforms) other than macOS
function __init__()
    DFExact.HAVE_IVM[] = Sys.ARCH === :x86_64 && !Sys.isapple() && !DFExact.HAVE_ACCEL[]
    nothing
end

end # module
