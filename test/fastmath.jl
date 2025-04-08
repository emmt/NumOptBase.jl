"""

This module runs several tests to check the behavior of the `@fastmath` macro. See
`README.md` for a summary of the results and also See code in `base/fastmath.jl` for the
Julia implementation of the `@fastmath` macro.

"""
module TestingFastMath

function test_predicates(::Type{T} = Float64) where {T}
    println("Test predicates for `T = $T`:")
    for (f, com) in ((:isnan,    "always false with @fastmath"),
                     (:isfinite, "always true with @fastmath"),
                     (:isinf,    "always false with @fastmath"),
                     (:iszero,   ""),
                     (:isone,    ""))
        print(" • `$f(x)`")
        if length(com) > 0
            print(" (")
            printstyled(com; color=:yellow)
            println("):")
        else
            println(":")
        end
        for x in (T(NaN), +T(Inf), -T(Inf), zero(T), one(T), -one(T))
            slow_result = @eval $f($x)
            fast_result = @eval @fastmath($f($x))
            print("   • `@fastmath($f($(prt(x))))` -> ")
            printstyled(fast_result; color=(fast_result == slow_result ? :green : :red))
            println()
        end
    end
end

function test_binary_operators(::Type{T} = Float64) where {T}
    println("Test binary operators for `T = $T`:")
    for (f, com) in ((:(+),   "arithmetic operations ok for all"),
                     (:(-),   "arithmetic operations ok for all"),
                     (:(*),   "arithmetic operations ok for all"),
                     (:(/),   "arithmetic operations ok for all"),
                     (:(%),   "arithmetic operations ok for all"),
                     (:(===), "ok for all"),
                     (:(==),  "comparisons fail for NaN"),
                     (:(!=),  "comparisons fail for NaN"),
                     (:(<),   "comparisons fail for NaN"),
                     (:(<=),  "comparisons fail for NaN"),
                     (:(>),   "comparisons fail for NaN"),
                     (:(>=),  "comparisons fail for NaN"))
        print(" • `x $f y`")
        if length(com) > 0
            print(" (")
            printstyled(com; color=:yellow)
            println("):")
        else
            println(":")
        end
        for x in (T(NaN), +T(Inf), -T(Inf), zero(T), one(T), -one(T))
            for y in (T(NaN), +T(Inf), -T(Inf), zero(T), one(T), -one(T))
                slow_result = @eval $f($x,$y)
                fast_result = @eval @fastmath($f($x,$y))
                print("   • `@fastmath($(prt(x)) $f $(prt(y)))` -> ")
                if fast_result === slow_result
                    printstyled(prt(fast_result); color=:green)
                else
                    printstyled(prt(fast_result); color=:red)
                    print(", truth is ")
                    printstyled(prt(slow_result); color=:blue)
                end
                println()
            end
        end
    end
end

function test_binary_functions(::Type{T} = Float64) where {T}
    println("Test binary functions for `T = $T`:")
    for (f, com) in ((:isless, ""),
                     (:isequal, ""),
                     (:min, ""),
                     (:max, ""),
                     (:cmp, "without fastmath, NaN is considered as the largest possible value; with fastmath, NaN is considered as equal to any value "),
                     (:hypot, "fastmath yield more intuitive result"))
        print(" • `$f(x, y)`")
        if length(com) > 0
            print(" (")
            printstyled(com; color=:yellow)
            println("):")
        else
            println(":")
        end
        for x in (T(NaN), +T(Inf), -T(Inf), zero(T), one(T), -one(T))
            for y in (T(NaN), +T(Inf), -T(Inf), zero(T), one(T), -one(T))
                slow_result = @eval $f($x,$y)
                fast_result = @eval @fastmath($f($x,$y))
                print("   • `@fastmath($f($(prt(x)), $(prt(y))))` -> ")
                if fast_result === slow_result
                    printstyled(prt(fast_result); color=:green)
                else
                    printstyled(prt(fast_result); color=:red)
                    print(", truth is ")
                    printstyled(prt(slow_result); color=:blue)
                end
                println()
            end
        end
    end
end

function test_unary_functions(::Type{T} = Float64) where {T}
    println("Test unary functions for `T = $T`:")
    for (f, com) in ((:sign, ""),
                     (:abs, ""),
                     (:abs2, ""),
                     (:inv, ""),
                     (:sign, ""),
                     (:sqrt, "fastmath yield more practical result"),
                     (:exp, ""),
                     (:log, ""),
                     (:sin, ""),
                     (:cos, ""))
        print(" • `$f(x)`")
        if length(com) > 0
            print(" (")
            printstyled(com; color=:yellow)
            println("):")
        else
            println(":")
        end
        for x in (T(NaN), +T(Inf), -T(Inf), zero(T), one(T), -one(T))
            slow_result= try; @eval $f($x); catch; :throws ; end
            fast_result = try; @eval @fastmath($f($x)); catch; :throws; end
            print("   • `@fastmath($f($(prt(x))))` -> ")
            if fast_result === slow_result || (_isnan(slow_result) && _isnan(fast_result))
                printstyled(prt(fast_result); color=:green)
            else
                printstyled(prt(fast_result); color=:red)
                print(", truth is ")
                printstyled(prt(slow_result); color=:blue)
            end
            println()
        end
    end
end

function test_consistency(::Type{T} = Float64) where {T}
    println("Test consistency of results for `T = $T`:")
    for (f, nf) in ((:(==), :(!=)),
                    (:(<), :(>=)),
                    (:(>), :(<=)))
        println(" • `x $f y` versus `x $nf y`:")
        for x in (T(NaN), +T(Inf), -T(Inf), zero(T), one(T), -one(T))
            for y in (T(NaN), +T(Inf), -T(Inf), zero(T), one(T), -one(T))
                r = @eval @fastmath($f($x,$y))
                nr = @eval @fastmath($nf($x,$y))
                print("   • `@fastmath($(prt(x)) $f $(prt(y)))` -> $(prt(r)),")
                print(" `@fastmath($(prt(x)) $nf $(prt(y)))` -> $(prt(nr)), ")
                if nr == !r
                    printstyled("ok"; color=:green)
                else
                    printstyled("inconsistent"; color=:red)
                end
                println()
            end
        end
    end
end

prt(x) = repr(x)
prt(x::Symbol) = x
_isnan(x::Number) = isnan(x)
_isnan(x::Symbol) = false

end # module
if isinteractive()
    TestingFastMath.test_predicates()
    println()
    TestingFastMath.test_binary_operators()
    println()
    TestingFastMath.test_binary_functions()
    println()
    TestingFastMath.test_unary_functions()
    println()
    TestingFastMath.test_consistency()
end
nothing
