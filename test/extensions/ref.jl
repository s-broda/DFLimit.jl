# The reference values of the extensions for the ports (test/ref/ext_ref.txt, written by dev/ext_ref.jl, the tags in its
# header), recomputed: a regression test of the Julia code and a check that the file is that of the code.
using DFLimit, Test
const DD = DFLimit.DFDet
const REFX = joinpath(@__DIR__, "..", "ref", "ext_ref.txt")
errmsg(f) = try; f(); ""; catch e; e isa ArgumentError ? e.msg : "other: " * sprint(showerror, e); end

@testset "the reference values of the extensions (test/ref/ext_ref.txt)" begin
    lines = [split(l) for l in eachline(REFX) if !startswith(l, "#") && !isempty(strip(l))]
    num(s) = parse(Float64, s)
    cpx(v, k) = complex(num(v[k]), num(v[k+1]))
    designs = Dict{Int,Determ}(); vols = Dict{Int,VolPath}()
    for v in lines
        if v[1] == "design"
            id = parse(Int, v[2])
            designs[id] = v[3] == "trend" ? trend(parse(Int, v[4])) : perron(Symbol(v[4]), parse(Int, v[5]) == 1 ? num(v[6]) : num.(v[6:end]))
        elseif v[1] == "vol"
            nb = parse(Int, v[3])
            vols[parse(Int, v[2])] = volpath(num.(v[4:3+nb]), num.(v[4+nb:4+2nb]))
        end
    end
    # a design id, a volatility id and the c of its regimes from position k: (D, V, c, next position)
    function dvc(v, k)
        D = designs[parse(Int, v[k])]; V = vols[parse(Int, v[k+1])]; nc = parse(Int, v[k+2])
        D, V, num.(v[k+3:k+2+nc]), k + 3 + nc
    end
    vecs(v, k) = (n = parse(Int, v[k]); (num.(v[k+1:k+n]), k + 1 + n))
    # a regime variant from position k (did vid nc c.. start gls first): (design, V, c, keywords, next position)
    function rvar(v, k)
        D, V, c, k = dvc(v, k)
        g = num(v[k+1])
        (parse(Int, v[k+2]) == 1 ? detrend_first(D) : D), V, c, (; start = parse(Int, v[k]) == 2 ? :stationary : :zero, gls = isnan(g) ? nothing : g), k + 3
    end
    counts = Dict{String,Int}()
    for v in lines
        tag = v[1]; counts[tag] = get(counts, tag, 0) + 1
        if tag == "vdval"
            D, V, c, k = dvc(v, 2)
            s = cpx(v, k); t = cpx(v, k + 2); d = cpx(v, k + 4)
            @test abs(DD.vol_dval(DD.vol_model(D, V, c), s, t) - d) <= num(v[k+6])
        elseif tag == "vroots"
            D, V, c, k = dvc(v, 2)
            VM = DD.vol_model(D, V, c)
            _, ld, mu, _ = DD.mgfparts(VM, 0.0, cpx(v, k), DD.MWork(VM))
            r = parse(Int, v[k+3]); ref = [cpx(v, k + 4 + 2(j - 1)) for j in 1:r]
            @test abs(real(ld) - num(v[k+2])) <= num(v[end])
            @test maximum(abs, sort(mu; by = z -> (real(z), imag(z))) .- sort(ref; by = z -> (real(z), imag(z)))) <= num(v[end]) * maximum(abs, ref)
        elseif tag == "vtstar"
            D, V, c, k = dvc(v, 2)
            @test abs(DD.tstar(DD.vol_model(D, V, c), minimum(c))[1] - num(v[k])) <= num(v[k+1])
        elseif tag == "vval"
            D, V, c, k = dvc(v, 3)
            c0 = num(v[k]); x, k = vecs(v, k + 1); ref, k = vecs(v, k)
            val = v[2] == "cdf" ? dist_cdf(D, x, c; vol = V, c0) : dist_pdf(D, x, c; vol = V, c0)
            @test maximum(abs, val .- ref) <= num(v[k])
        elseif tag == "vquant"
            D, V, c, k = dvc(v, 2)
            @test abs(dist_quantile(D, num(v[k+1]), c; vol = V, c0 = num(v[k])) - num(v[k+2])) <= num(v[k+3])
        elseif tag == "rdval"
            D, V, c, kw, k = rvar(v, 2)
            s = cpx(v, k); t = cpx(v, k + 2); d = cpx(v, k + 4)
            @test abs(DD.vol_dval(DD.vol_model(D, V, c; kw...), s, t) - d) <= num(v[k+6])
        elseif tag == "rroots"
            D, V, c, kw, k = rvar(v, 2)
            VM = DD.vol_model(D, V, c; kw...)
            _, ld, mu, _ = DD.mgfparts(VM, 0.0, cpx(v, k), DD.MWork(VM))
            r = parse(Int, v[k+3]); ref = [cpx(v, k + 4 + 2(j - 1)) for j in 1:r]
            @test abs(real(ld) - num(v[k+2])) <= num(v[end])
            @test maximum(abs, sort(mu; by = z -> (real(z), imag(z))) .- sort(ref; by = z -> (real(z), imag(z)))) <= num(v[end]) * maximum(abs, ref)
        elseif tag == "rtstar"
            D, V, c, kw, k = rvar(v, 2)
            @test abs(DD.tstar(DD.vol_model(D, V, c; kw...), minimum(c))[1] - num(v[k])) <= num(v[k+1])
        elseif tag == "rval"
            D, V, c, kw, k = rvar(v, 3)
            c0 = num(v[k]); x, k = vecs(v, k + 1); ref, k = vecs(v, k)
            val = v[2] == "cdf" ? dist_cdf(D, x, c; vol = V, c0, kw...) : dist_pdf(D, x, c; vol = V, c0, kw...)
            @test maximum(abs, val .- ref) <= num(v[k])
        elseif tag == "rquant"
            D, V, c, kw, k = rvar(v, 2)
            @test abs(dist_quantile(D, num(v[k+1]), c; vol = V, c0 = num(v[k]), kw...) - num(v[k+2])) <= num(v[k+3])
        elseif tag == "rcase"
            V = vols[parse(Int, v[4])]; nc = parse(Int, v[5]); c = num.(v[6:5+nc]); k = 6 + nc
            c0 = num(v[k]); x, k = vecs(v, k + 1); ref, k = vecs(v, k)
            val = v[2] == "cdf" ? ltratio_cdf(x, v[3], c; vol = V, c0) : v[2] == "pdf" ? ltratio_pdf(x, v[3], c; vol = V, c0) :
                                  [ltratio_icdf(p, v[3], c; vol = V, c0) for p in x]
            @test maximum(abs, val .- ref) <= num(v[k])
        elseif tag == "rwarn"
            D, V, c, kw, k = rvar(v, 2)
            logger = Test.TestLogger(min_level = Base.CoreLogging.Warn)
            F = Base.CoreLogging.with_logger(() -> dist_cdf(D, [num(v[k+1])], c; vol = V, c0 = num(v[k]), kw...), logger)
            @test any(r -> occursin(join(v[findfirst(==("|"), v)+1:end], " "), r.message), logger.logs)
            @test abs(F[1] - num(v[k+2])) <= num(v[k+3])
        elseif tag == "cval"
            D = designs[parse(Int, v[3])]; st = (:zero, :fixed, :stationary)[parse(Int, v[4])+1]
            g0, c, c0, rho = num.(v[5:8]); m = Symbol(v[9])
            x, k = vecs(v, 10); ref, k = vecs(v, k)
            kw = (; c0, start = st, gamma0 = g0, rho, method = rho == 0 ? :auto : m)
            val = v[2] == "cdf" ? dist_cdf(D, x, c; kw...) : dist_pdf(D, x, c; kw...)
            @test maximum(abs, val .- ref) <= num(v[k])
        elseif tag == "cquant"
            D = designs[parse(Int, v[2])]; c, c0, rho, p, x, tol = num.(v[5:10])
            @test abs(dist_quantile(D, p, c; c0, rho) - x) <= tol
        elseif tag == "sval"
            D = designs[parse(Int, v[3])]; c = num(v[4]); c0 = num(v[5]); vid = parse(Int, v[6])
            x, k = vecs(v, 7); ref, k = vecs(v, k)
            f = v[2]
            val = f == "t3cdf" ? (vid == 0 ? hegy_t3_cdf(D, x, c; c0) : hegy_t3_cdf(D, x, [c, c]; c0, vol = vols[vid])) :
                                 (vid == 0 ? hegy_t3_pdf(D, x, c; c0) : hegy_t3_pdf(D, x, [c, c]; c0, vol = vols[vid]))
            @test maximum(abs, val .- ref) <= num(v[k])
        elseif tag == "squant"
            D = designs[parse(Int, v[3])]; c, p, x, tol = num.(v[4:7])
            q = hegy_t3_quantile(D, p, c)
            @test abs(q - x) <= tol
        end
    end
    @test get(counts, "vval", 0) == 20 && get(counts, "cval", 0) == 20 && get(counts, "sval", 0) == 12 && get(counts, "squant", 0) == 2
    @test get(counts, "rdval", 0) == 30 && get(counts, "rroots", 0) == 30 && get(counts, "rtstar", 0) == 10 && get(counts, "rval", 0) == 38
    @test get(counts, "rquant", 0) == 12 && get(counts, "rcase", 0) == 12 && get(counts, "rwarn", 0) == 1
    @test length([v for v in lines if v[1] == "err"]) == 18 && !haskey(counts, "sdval")
end
