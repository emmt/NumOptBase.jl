module NumOptBaseCUDAExt

if isdefined(Base, :get_extension)
    using StructuredArrays
    using CUDA
    using NumOptBase
    using NumOptBase:
        CudaEngine,
        PlusOrMinus,
        RealComplex,
        convert_multiplier,
        get_bound,
        can_vary,
        project_direction,
        project_variable,
        αx, αxpy, αxmy, αxpβy
    import NumOptBase:
        engine,
        flatten,
        norm1,
        norm2,
        norminf,
        unsafe_copy!,
        unsafe_inner,
        unsafe_map!,
        unsafe_project_direction!,
        unsafe_project_variables!,
        unsafe_changing_variables!,
        unsafe_update!,
        zerofill!
else
    using ..StructuredArrays
    using ..CUDA
    using ..NumOptBase
    using ..NumOptBase:
        CudaEngine,
        PlusOrMinus,
        RealComplex,
        convert_multiplier,
        get_bound,
        can_vary,
        project_direction,
        project_variable,
        αx, αxpy, αxmy, αxpβy
    import ..NumOptBase:
        engine,
        flatten,
        norm1,
        norm2,
        norminf,
        unsafe_copy!,
        unsafe_inner,
        unsafe_map!,
        unsafe_project_direction!,
        unsafe_project_variables!,
        unsafe_changing_variables!,
        unsafe_update!,
        zerofill!
end

engine(::CuArray, ::Union{CuArray,AbstractUniformArray}...) = CudaEngine

flatten(A::CuArray{T,1}) where {T} = A
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

# The "device" version of a given method takes array arguments of type
# `CuDeviceArray` and return `nothing` while the "host" version takes array
# arguments of type `CuArray`.

for n in 1:3
    vars = (:x, :y, :z)[1:n] # list of variables
    args = (:($(var)::AbstractArray) for var in vars) # list of arguments with types
    vals = (:($(var)[i]) for var in vars) # list of values (indexed arguments)
    @eval begin
        function unsafe_gpu_map!(f::Function, dst::CuDeviceArray, $(args...))
            @inbounds for i in @gpu_range(dst)
                dst[i] = f($(vals...))
            end
            return nothing # GPU kernels return nothing
        end
        function unsafe_map!(::Type{<:CudaEngine}, f::Function, dst::CuArray, $(args...))
            kernel = @cuda launch=false unsafe_gpu_map!(f, dst, $(vars...))
            threads, blocks = gpu_config(kernel, dst)
            kernel(f, dst, $(vars...); threads, blocks)
            return nothing
        end
    end
end

function unsafe_gpu_αx!(dst::CuDeviceArray, α::Real, x::AbstractArray)
    @inbounds for i in @gpu_range(dst)
        dst[i] = α*x[i]
    end
    return nothing # GPU kernels return nothing
end

function unsafe_gpu_αxpy!(dst::CuDeviceArray, α::Real, x::AbstractArray, y::AbstractArray)
    @inbounds for i in @gpu_range(y)
        dst[i] = α*x[i] + y[i]
    end
    return nothing # GPU kernels return nothing
end

function unsafe_gpu_αxmy!(dst::CuDeviceArray, α::Real, x::AbstractArray, y::AbstractArray)
    @inbounds for i in @gpu_range(y)
        dst[i] = α*x[i] - y[i]
    end
    return nothing # GPU kernels return nothing
end

function unsafe_gpu_αxpβy!(dst::CuDeviceArray, α::Real, x::AbstractArray, β::Real, y::AbstractArray)
    @inbounds for i in @gpu_range(dst)
        dst[i] = α*x[i] + β*y[i]
    end
    return nothing # GPU kernels return nothing
end

function unsafe_map!(::Type{<:CudaEngine}, f::αx, dst::CuArray, x::AbstractArray)
    kernel = @cuda launch=false unsafe_gpu_αx!(dst, f.α, x)
    threads, blocks = gpu_config(kernel, dst)
    kernel(dst, f.α, x; threads, blocks)
    return nothing
end

function unsafe_map!(::Type{<:CudaEngine}, f::αxpy, dst::CuArray, x::AbstractArray, y::AbstractArray)
    kernel = @cuda launch=false unsafe_gpu_αxpy!(dst, f.α, x, y)
    threads, blocks = gpu_config(kernel, dst)
    kernel(dst, f.α, x, y; threads, blocks)
    return nothing
end

function unsafe_map!(::Type{<:CudaEngine}, f::αxmy, dst::CuArray, x::AbstractArray, y::AbstractArray)
    kernel = @cuda launch=false unsafe_gpu_αxmy!(dst, f.α, x, y)
    threads, blocks = gpu_config(kernel, dst)
    kernel(dst, f.α, x, y; threads, blocks)
    return nothing
end

function unsafe_map!(::Type{<:CudaEngine}, f::αxpβy, dst::CuArray, x::AbstractArray, y::AbstractArray)
    kernel = @cuda launch=false unsafe_gpu_αxpβy!(dst, f.α, x, f.β, y)
    threads, blocks = gpu_config(kernel, dst)
    kernel(dst, f.α, x, f.β, y; threads, blocks)
    return nothing
end

function unsafe_gpu_update!(dst::CuDeviceArray, α::Real, x::CuDeviceArray)
    @inbounds for i in @gpu_range(dst)
        dst[i] += α*x[i]
    end
    return nothing # GPU kernels return nothing
end

function unsafe_gpu_update!(dst::CuDeviceArray, α::Real, x::AbstractArray, y::AbstractArray)
    @inbounds for i in @gpu_range(dst)
        dst[i] += α*x[i]*y[i]
    end
    return nothing # GPU kernels return nothing
end

function unsafe_update!(::Type{<:CudaEngine}, dst::CuArray, α::Real, x::AbstractArray)
    kernel = @cuda launch=false unsafe_gpu_update!(dst, α, x)
    threads, blocks = gpu_config(kernel, dst)
    kernel(dst, α, x; threads, blocks)
    return nothing
end

function unsafe_update!(::Type{<:CudaEngine}, dst::CuArray, α::Real, x::AbstractArray, y::AbstractArray)
    kernel = @cuda launch=false unsafe_gpu_update!(dst, α, x, y)
    threads, blocks = gpu_config(kernel, dst)
    kernel(dst, α, x, y; threads, blocks)
    return nothing
end

function unsafe_inner(::Type{<:CudaEngine}, x::CuArray, y::CuArray)
    #return mapreduce(inner, +, x, y; init=inner(zero(eltype(x)),zero(eltype(y))))
    return flatten(x)'*flatten(y) # NOTE: this is faster than mapreduce
end

function unsafe_inner(::Type{<:CudaEngine}, w::CuArray, x::CuArray, y::CuArray)
    return mapreduce(inner, +, w, x, y; init=inner(zero(eltype(w)),zero(eltype(x)),zero(eltype(y))))
end

function norm1(::Type{<:CudaEngine}, x::CuArray)
    return mapreduce(norm1, +, x; init=norm1(zero(eltype(x))))
end

function norm2(::Type{<:CudaEngine}, x::CuArray)
    return sqrt(mapreduce(abs2, +, x; init=abs2(zero(eltype(x)))))
end

function norminf(::Type{<:CudaEngine}, x::CuArray)
    return mapreduce(norminf, max, x; init=norminf(zero(eltype(x))))
end

function unsafe_gpu_project_variables!(dst::CuDeviceArray, x, lower, upper)
    @inbounds for i in @gpu_range(dst)
        dst[i] = project_variable(x[i], get_bound(lower, i), get_bound(upper, i))
    end
    return nothing # GPU kernels return nothing
end

function unsafe_project_variables!(::Type{<:CudaEngine}, dst::CuArray, x, lower, upper)
    kernel = @cuda launch=false unsafe_gpu_project_variables!(dst, x, lower, upper)
    threads, blocks = gpu_config(kernel, dst)
    kernel(dst, x, lower, upper; threads, blocks)
    return nothing
end

function unsafe_gpu_project_direction!(dst::CuDeviceArray, x, pm::PlusOrMinus, d, lower, upper)
    @inbounds for i in @gpu_range(dst)
        dst[i] = project_direction(x[i], pm, d[i], get_bound(lower, i), get_bound(upper, i))
    end
    return nothing # GPU kernels return nothing
end

function unsafe_project_direction!(::Type{<:CudaEngine}, dst::CuArray, x, pm::PlusOrMinus, d, lower, upper)
    kernel = @cuda launch=false unsafe_gpu_project_direction!(dst, x, pm, d, lower, upper)
    threads, blocks = gpu_config(kernel, dst)
    kernel(dst, x, pm, d, lower, upper; threads, blocks)
    return nothing
end

function unsafe_gpu_changing_variables!(dst::CuDeviceArray, x, pm::PlusOrMinus, d, lower, upper)
    @inbounds for i in @gpu_range(dst)
        dst[i] = can_vary(eltype(dst), x[i], pm, d[i], get_bound(lower, i), get_bound(upper, i))
    end
    return nothing # GPU kernels return nothing
end

function unsafe_changing_variables!(::Type{<:CudaEngine}, dst::CuArray, x, pm::PlusOrMinus, d, lower, upper)
    kernel = @cuda launch=false unsafe_gpu_changing_variables!(dst, x, pm, d, lower, upper)
    threads, blocks = gpu_config(kernel, dst)
    kernel(dst, x, pm, d, lower, upper; threads, blocks)
    return nothing
end

end # module
