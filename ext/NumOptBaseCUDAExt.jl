module NumOptBaseCUDAExt

if isdefined(Base, :get_extension)
    using CUDA
    using NumOptBase
    using NumOptBase:
        CudaEngine,
        PlusOrMinus,
        RealComplex,
        convert_multiplier,
        get_bound,
        is_unblocked,
        project,
        αx, αxpy, αxmy, αxpβy
    import NumOptBase:
        engine,
        flatten,
        norm1,
        norm2,
        norminf,
        unsafe_copy!,
        unsafe_inner,
        unsafe_linesearch_limits,
        unsafe_linesearch_stepmax,
        unsafe_linesearch_stepmin,
        unsafe_map!,
        unsafe_project_direction!,
        unsafe_project_variables!,
        unsafe_unblocked_variables!,
        unsafe_update!,
        zerofill!
else
    using ..CUDA
    using ..NumOptBase
    using ..NumOptBase:
        CudaEngine,
        PlusOrMinus,
        RealComplex,
        convert_multiplier,
        get_bound,
        is_unblocked,
        project,
        αx, αxpy, αxmy, αxpβy
    import ..NumOptBase:
        engine,
        flatten,
        norm1,
        norm2,
        norminf,
        unsafe_copy!,
        unsafe_inner,
        unsafe_linesearch_limits,
        unsafe_linesearch_stepmax,
        unsafe_linesearch_stepmin,
        unsafe_map!,
        unsafe_project_direction!,
        unsafe_project_variables!,
        unsafe_unblocked_variables!,
        unsafe_update!,
        zerofill!
end

engine(args::CuArray...) = CudaEngine

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
arrays of same dimensions as `A`. This facility is provided by a macro because
these thread-wise computations cannot be done by a regular function.

"""
macro gpu_range(A)
    A isa Symbol || error("expecting a symbolic name")
    esc(:(
        #= first =# ((blockIdx().x - 1)*blockDim().x + threadIdx().x) :
        #= step  =# (blockDim().x*gridDim().x) :
        #= last  =# length($A)
    ))
end

"""
    gpu_config(fun, len) -> threads, blocks

yields suitable numbers of threads and blocks for applying GPU kernel/function
`fun` on arrays with `len` elements. If second argument is an array, its length
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

unsafe_copy!(dst::CuArray, x::CuArray) = unsafe_copy!(dst, x)
zerofill!(dst::CuArray) = fill!(dst, zero(eltype(dst)))

# The "device" version of a given method takes array arguments of type
# `CuDeviceArray` and return `nothing` while the "host" version takes array
# arguments of type `CuArray`.

function unsafe_map!(::Type{<:CudaEngine}, f::Function, dst::CuArray, x::CuArray)
    function func!(f, dst, x)
        for i in @gpu_range(dst)
            @inbounds dst[i] = f(x[i])
        end
        return nothing # GPU kernels return nothing
    end
    kernel = @cuda launch=false func!(f, dst, x)
    threads, blocks = gpu_config(kernel, dst)
    kernel(f, dst, x; threads, blocks)
    return nothing
end

function unsafe_map!(::Type{<:CudaEngine}, f::Function, dst::CuArray, x::CuArray, y::CuArray)
    function func!(f, dst, x, y)
        for i in @gpu_range(dst)
            @inbounds dst[i] = f(x[i], y[i])
        end
        return nothing # GPU kernels return nothing
    end
    kernel = @cuda launch=false func!(f, dst, x, y)
    threads, blocks = gpu_config(kernel, dst)
    kernel(f, dst, x, y; threads, blocks)
    return nothing
end

function unsafe_map!(::Type{<:CudaEngine}, f::Function, dst::CuArray, x::CuArray, y::CuArray, z::CuArray)
    function func!(f, dst, x, y, z)
        for i in @gpu_range(dst)
            @inbounds dst[i] = f(x[i], y[i], z[i])
        end
        return nothing # GPU kernels return nothing
    end
    kernel = @cuda launch=false func!(f, dst, x, y, z)
    threads, blocks = gpu_config(kernel, dst)
    kernel(f, dst, x, y, z; threads, blocks)
    return nothing
end

function unsafe_map!(::Type{<:CudaEngine}, f::αx, dst::CuArray, x::CuArray)
    function func!(dst, α, x)
        for i in @gpu_range(dst)
            @inbounds dst[i] = α*x[i]
        end
        return nothing # GPU kernels return nothing
    end
    kernel = @cuda launch=false func!(dst, f.α, x)
    threads, blocks = gpu_config(kernel, dst)
    kernel(dst, f.α, x; threads, blocks)
    return nothing
end

function unsafe_map!(::Type{<:CudaEngine}, f::αxpy, dst::CuArray, x::CuArray, y::CuArray)
    function func!(dst, α, x, y)
        for i in @gpu_range(y)
            @inbounds dst[i] = α*x[i] + y[i]
        end
        return nothing # GPU kernels return nothing
    end
    kernel = @cuda launch=false func!(dst, f.α, x, y)
    threads, blocks = gpu_config(kernel, dst)
    kernel(dst, f.α, x, y; threads, blocks)
    return nothing
end

function unsafe_map!(::Type{<:CudaEngine}, f::αxmy, dst::CuArray, x::CuArray, y::CuArray)
    function func!(dst, α, x, y)
        for i in @gpu_range(y)
            @inbounds dst[i] = α*x[i] - y[i]
        end
        return nothing # GPU kernels return nothing
    end
    kernel = @cuda launch=false func!(dst, f.α, x, y)
    threads, blocks = gpu_config(kernel, dst)
    kernel(dst, f.α, x, y; threads, blocks)
    return nothing
end

function unsafe_map!(::Type{<:CudaEngine}, f::αxpβy, dst::CuArray, x::CuArray, y::CuArray)
    function func!(dst, α, x, β, y)
        for i in @gpu_range(dst)
            @inbounds dst[i] = α*x[i] + β*y[i]
        end
        return nothing # GPU kernels return nothing
    end
    kernel = @cuda launch=false func!(dst, f.α, x, f.β, y)
    threads, blocks = gpu_config(kernel, dst)
    kernel(dst, f.α, x, f.β, y; threads, blocks)
    return nothing
end

function device_unsafe_update!(dst::CuDeviceArray,
                               α::Real,
                               x::CuDeviceArray)
    @inbounds for i in @gpu_range(dst)
        dst[i] += α*x[i]
    end
    return nothing # GPU kernels return nothing
end

function device_unsafe_update!(dst::CuDeviceArray,
                               α::Real,
                               x::CuDeviceArray,
                               y::CuDeviceArray)
    @inbounds for i in @gpu_range(dst)
        dst[i] += α*x[i]*y[i]
    end
    return nothing # GPU kernels return nothing
end

function unsafe_update!(::Type{<:CudaEngine},
                        dst::CuArray,
                        α::Real,
                        x::CuArray)
    kernel = @cuda launch=false device_unsafe_update!(dst, α, x)
    threads, blocks = gpu_config(kernel, dst)
    kernel(dst, α, x; threads, blocks)
    return nothing
end

function unsafe_update!(::Type{<:CudaEngine},
                        dst::CuArray,
                        α::Real,
                        x::CuArray,
                        y::CuArray)
    kernel = @cuda launch=false device_unsafe_update!(dst, α, x, y)
    threads, blocks = gpu_config(kernel, dst)
    kernel(dst, α, x, y; threads, blocks)
    return nothing
end

function unsafe_inner(::Type{<:CudaEngine},
                      x::CuArray{T,N},
                      y::CuArray{T,N}) where {R<:Real,T<:RealComplex{R},N}
    #return mapreduce(inner, +, x, y; init=inner(zero(R),zero(R)))
    return flatten(x)'*flatten(y) # NOTE: this is faster than mapreduce
end

function unsafe_inner(::Type{<:CudaEngine},
                      w::CuArray{R,N},
                      x::CuArray{R,N},
                      y::CuArray{R,N}) where {R<:Real,N}
    return mapreduce(inner, +, w, x, y; init=inner(zero(R),zero(R),zero(R)))
end

function norm1(::Type{<:CudaEngine}, x::CuArray{T,N}) where {R<:Real,T<:RealComplex{R},N}
    return mapreduce(norm1, +, x; init=norm1(zero(R)))
end

function norm2(::Type{<:CudaEngine}, x::CuArray{T,N}) where {R<:Real,T<:RealComplex{R},N}
    return sqrt(mapreduce(abs2, +, x; init=abs2(zero(R))))
end

function norminf(::Type{<:CudaEngine}, x::CuArray{T,N}) where {R<:Real,T<:RealComplex{R},N}
    return mapreduce(norminf, max, x; init=norminf(zero(R)))
end

# Type of arguments suitable to represent a bound for CUDA arrays, see
# `Bound{T,N}`.
const CuBound{T,N} = Union{Nothing,T,CuArray{T,N}}
const CuDeviceBound{T,N} = Union{Nothing,T,CuDeviceArray{T,N}}

function device_project_variables!(dst::CuDeviceArray{T,N},
                                   x::CuDeviceArray{T,N},
                                   lower::CuDeviceBound{T,N},
                                   upper::CuDeviceBound{T,N}) where {T,N}
    @inbounds for i in @gpu_range(dst)
        dst[i] = project(x[i], get_bound(lower, i), get_bound(upper, i))
    end
    return nothing # GPU kernels return nothing
end

function unsafe_project_variables!(::Type{<:CudaEngine},
                                   dst::CuArray{T,N},
                                   x::CuArray{T,N},
                                   lower::CuBound{T,N},
                                   upper::CuBound{T,N}) where {T,N}
    kernel = @cuda launch=false device_project_variables!(dst, x, lower, upper)
    threads, blocks = gpu_config(kernel, dst)
    kernel(dst, x, lower, upper; threads, blocks)
    return nothing
end

function device_project_direction!(dst::CuDeviceArray{T,N},
                                   x::CuDeviceArray{T,N},
                                   pm::PlusOrMinus,
                                   d::CuDeviceArray{T,N},
                                   lower::CuDeviceBound{T,N},
                                   upper::CuDeviceBound{T,N}) where {T,N}
    @inbounds for i in @gpu_range(dst)
        dst[i] = project(x[i], pm, d[i], get_bound(lower, i), get_bound(upper, i))
    end
    return nothing # GPU kernels return nothing
end

function unsafe_project_direction!(::Type{<:CudaEngine},
                                   dst::CuArray{T,N},
                                   x::CuArray{T,N},
                                   pm::PlusOrMinus,
                                   d::CuArray{T,N},
                                   lower::CuBound{T,N},
                                   upper::CuBound{T,N}) where {T,N}
    kernel = @cuda launch=false device_project_direction!(dst, x, pm, d, lower, upper)
    threads, blocks = gpu_config(kernel, dst)
    kernel(dst, x, pm, d, lower, upper; threads, blocks)
    return nothing
end

function device_unblocked_variables!(dst::CuDeviceArray{B,N},
                                     x::CuDeviceArray{T,N},
                                     pm::PlusOrMinus,
                                     d::CuDeviceArray{T,N},
                                     lower::CuDeviceBound{T,N},
                                     upper::CuDeviceBound{T,N}) where {B,T,N}
    @inbounds for i in @gpu_range(dst)
        dst[i] = is_unblocked(B, x[i], pm, d[i],
                              get_bound(lower, i), get_bound(upper, i))
    end
    return nothing # GPU kernels return nothing
end

function unsafe_unblocked_variables!(::Type{<:CudaEngine},
                                     dst::CuArray{B,N},
                                     x::CuArray{T,N},
                                     pm::PlusOrMinus,
                                     d::CuArray{T,N},
                                     lower::CuBound{T,N},
                                     upper::CuBound{T,N}) where {B,T,N}
    kernel = @cuda launch=false device_unblocked_variables!(dst, x, pm, d, lower, upper)
    threads, blocks = gpu_config(kernel, dst)
    kernel(dst, x, pm, d, lower, upper; threads, blocks)
    return nothing
end

# FIXME: The returned value is conservative.
function unsafe_linesearch_stepmin(::Type{<:CudaEngine},
                                   x::CuArray{T,N},
                                   pm::PlusOrMinus,
                                   d::CuArray{T,N},
                                   lower::CuBound{T,N},
                                   upper::CuBound{T,N}) where {T,N}
    return zero(T)
end

# FIXME: The returned value is conservative.
function unsafe_linesearch_stepmax(::Type{<:CudaEngine},
                                   x::CuArray{T,N},
                                   pm::PlusOrMinus,
                                   d::CuArray{T,N},
                                   lower::CuBound{T,N},
                                   upper::CuBound{T,N}) where {T,N}
    return typemax(T)
end

# FIXME: The returned values are conservative.
function unsafe_linesearch_limits(::Type{<:CudaEngine},
                                  x::CuArray{T,N},
                                  pm::PlusOrMinus,
                                  d::CuArray{T,N},
                                  lower::CuBound{T,N},
                                  upper::CuBound{T,N}) where {T,N}
    return (unsafe_linesearch_stepmin(CudaEngine, x, pm, d, lower, upper),
            unsafe_linesearch_stepmax(CudaEngine, x, pm, d, lower, upper))
end

end # module
