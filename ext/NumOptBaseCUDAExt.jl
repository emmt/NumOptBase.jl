module NumOptBaseCUDAExt

if isdefined(Base, :get_extension)
    using CUDA
    import NumOptBase
    using NumOptBase: RealComplex, inner, norm1, norminf, convert_multiplier, unsafe_map!, unsafe_scale!
else
    using ..CUDA
    import ..NumOptBase
    using ..NumOptBase: RealComplex, inner, norm1, norminf, convert_multiplier, unsafe_map!, unsafe_scale!
end

flatten(A::CuArray{<:Real,1}) = A
@inline function flatten(A::CuArray{T,N}) where {R<:Real,T<:RealComplex{R},N}
    # NOTE: This is similar to `_derived_array` in `CUDA.jl/src/array.jl`.
    refcount = A.storage.refcount[]
    refcount > 0 || throw(AssertionError("unexpected reference count"))
    dims = (eltype(A) <: Complex ? 2*length(A) : length(A),)
    offset = (A.offset*Base.elsize(A))÷sizeof(R)
    maxsize = A.maxsize
    Threads.atomic_add!(A.storage.refcount, 1)
    B = CuArray{R,1}(A.storage, dims; maxsize, offset)
    return finalizer(CUDA.unsafe_finalize!, B)
end

"""
    @gpu_range(A)

yields the linear range of indices relevant for element-wise operations on GPU
arrays of same dimensions as `A`. These thread-wise computations cannot be done
by a regular function.

"""
macro gpu_range(A)
    A isa Symbol || error("expecting a symbolic name")
    esc(:(
        #= first =# ((blockIdx().x - 1)*blockDim().x + threadIdx().x) :
        #= step  =# (gridDim().x*blockDim().x) :
        #= last  =# length($A)
    ))
end

"""
    gpu_config(fun, len) -> threads, blocks

yields suitable numbers of threads and blocks for applying GPU kernel/function
`fun` on arrays with `len` elements. If second argment is an array, its length
is used.

"""
gpu_config(ker::Union{CUDA.HostKernel,CuFunction}, arr::AbstractArray) =
    gpu_config(ker, length(arr))
gpu_config(ker::CUDA.HostKernel, len::Int) = gpu_config(ker.fun, len)
function gpu_config(fun::CuFunction, len::Int)
    config = launch_configuration(fun)
    threads = min(len, config.threads)
    blocks = cld(len, threads)
    return threads, blocks
end

# The "device" version of a given method takes array arguments of type
# `CuDeviceArray` and return `nothing` while the "host" version takes array
# arguments of type `CuArray`.

function unsafe_ax!(dst::CuArray{T,N},
                    α::R, x::CuArray{T,N}) where {R<:Real,T<:RealComplex{R},N}
    function func!(dst, α, x)
        for i in @gpu_range(dst)
            @inbounds dst[i] = α*x[i]
        end
        return nothing
    end
    kernel = @cuda launch=false func!(dst, α, x)
    threads, blocks = gpu_config(kernel, dst)
    kernel(dst, α, x; threads, blocks)
    return nothing
end

function unsafe_axpy!(α::R, x::CuArray{T,N},
                      y::CuArray{T,N}) where {R<:Real,T<:RealComplex{R},N}
    function func!(α, x, y)
        for i in @gpu_range(y)
            @inbounds y[i] += α*x[i]
        end
        return nothing
    end
    kernel = @cuda launch=false func!(α, x, y)
    threads, blocks = gpu_config(kernel, y)
    kernel(α, x, y; threads, blocks)
    return nothing
end

function unsafe_xpby!(dst::CuArray{T,N},
                      x::CuArray{T,N},
                      β::R, y::CuArray{T,N}) where {R<:Real,T<:RealComplex{R},N}
    function func!(dst, x, β, y)
        for i in @gpu_range(dst)
            @inbounds dst[i] = x[i] + β*y[i]
        end
        return nothing
    end
    kernel = @cuda launch=false func!(dst, x, β, y)
    threads, blocks = gpu_config(kernel, dst)
    kernel(dst, x, β, y; threads, blocks)
    return nothing
end

function unsafe_axpby!(dst::CuArray{T,N},
                       α::R, x::CuArray{T,N},
                       β::R, y::CuArray{T,N}) where {R<:Real,T<:RealComplex{R},N}
    function func!(dst, α, x, β, y)
        for i in @gpu_range(dst)
            @inbounds dst[i] = α*x[i] + β*y[i]
        end
        return nothing
    end
    kernel = @cuda launch=false func!(dst, α, x, β, y)
    threads, blocks = gpu_config(kernel, dst)
    kernel(dst, α, x, β, y; threads, blocks)
    return nothing
end

function NumOptBase.unsafe_scale!(dst::CuArray{T,N},
                                  α::Real, x::CuArray{T,N}) where {R<:Real,T<:RealComplex{R},N}
    if iszero(α)
        fill!(dst, zero(eltype(dst)))
    elseif isone(α)
        dst !== x && copyto!(dst, x)
    elseif α == -one(α)
        unsafe_map!(-, dst, x)
    else
        unsafe_ax!(dst, convert_multiplier(α, x), x)
    end
    nothing
end

function NumOptBase.unsafe_update!(dst::CuArray{T,N},
                                   α::Real, x::CuArray{T,N}) where {R<:Real,T<:RealComplex{R},N}
    if isone(α)
        unsafe_map!(+, dst, dst, x)
    elseif α == -one(α)
        unsafe_map!(-, dst, dst, x)
    elseif !iszero(α)
        unsafe_axpy!(convert_multiplier(α, x), x, dst)
    end
    nothing
end

function NumOptBase.unsafe_combine!(dst::CuArray{T,N},
                                    α::Real, x::CuArray{T,N},
                                    β::Real, y::CuArray{T,N}) where {R<:Real,T<:RealComplex{R},N}
    if iszero(β)
        unsafe_scale!(dst, α, x)
    elseif iszero(α)
        unsafe_scale!(dst, β, y)
    elseif isone(α)
        if isone(β)
            # dst .= x .+ y
            unsafe_map!(+, dst, x, y)
        elseif β == -one(β)
            # dst .= x .- y
            unsafe_map!(-, dst, x, y)
        else
            # dst .= x .+ β.*y
            unsafe_xpby!(dst, x, convert_multiplier(β, y), y)
        end
    else
        # dst .= α.*x .+ β.*y
        unsafe_axpby!(dst,
                      convert_multiplier(α, x), x,
                      convert_multiplier(β, y), y)
    end
    nothing
end

function NumOptBase.unsafe_inner(x::CuArray{T,N},
                                 y::CuArray{T,N}) where {R<:Real,T<:RealComplex{R},N}
    #return mapreduce(inner, +, x, y; init=inner(zero(R),zero(R)))
    return flatten(x)'*flatten(y) # NOTE: this is a faster than mapreduce
end

function NumOptBase.unsafe_inner(w::CuArray{R,N},
                                 x::CuArray{R,N},
                                 y::CuArray{R,N}) where {R<:Real,N}
    return mapreduce(inner, +, w, x, y; init=inner(zero(R),zero(R),zero(R)))
end

function NumOptBase.norm1(x::CuArray{T,N}) where {R<:Real,T<:RealComplex{R},N}
    return mapreduce(norm1, +, x; init=norm1(zero(R)))
end

function NumOptBase.norm2(x::CuArray{T,N}) where {R<:Real,T<:RealComplex{R},N}
    return sqrt(mapreduce(abs2, +, x; init=abs2(zero(R))))
end

function NumOptBase.norminf(x::CuArray{T,N}) where {R<:Real,T<:RealComplex{R},N}
    return mapreduce(norminf, max, x; init=norminf(zero(R)))
end

function NumOptBase.unsafe_map!(f::Function, dst::CuArray,
                                x::CuArray)
    function func!(f, dst, x)
        for i in @gpu_range(dst)
            @inbounds dst[i] = f(x[i])
        end
        nothing
    end
    kernel = @cuda launch=false func!(f, dst, x)
    threads, blocks = gpu_config(kernel, dst)
    kernel(f, dst, x; threads, blocks)
    return nothing
end

function NumOptBase.unsafe_map!(f::Function, dst::CuArray,
                                x::CuArray, y::CuArray)
    function func!(f, dst, x, y)
        for i in @gpu_range(dst)
            @inbounds dst[i] = f(x[i], y[i])
        end
        nothing
    end
    kernel = @cuda launch=false func!(f, dst, x, y)
    threads, blocks = gpu_config(kernel, dst)
    kernel(f, dst, x, y; threads, blocks)
    return nothing
end

end # module
