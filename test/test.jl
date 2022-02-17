using CUDA

function arrayDef(a,datLen)
    i = threadIdx().x + (blockIdx().x-1)*blockDim().x
    j = threadIdx().y + (blockIdx().y-1)*blockDim().y
    if i <= datLen && j <=datLen
        a[i,j] = i+j
    end
    return
end

function arrayDef2(a,datLen)
    i = threadIdx().x + (blockIdx().x-1)*blockDim().x
    j = threadIdx().y + (blockIdx().y-1)*blockDim().y
    if i <= datLen && j <=datLen
        a[i,j] = (i+j)*2
    end
    return
end

function vadd(c, a, b)
    i = threadIdx().x + (blockIdx().x-1)*blockDim().x
    j = threadIdx().y + (blockIdx().y-1)*blockDim().y
    if i <= length(a) && j <= length(a)
        c[i,j] = a[i,j] + b[i,j]
    end
    return
end
a = CuArray{Int}(undef,(1024,1024))
b = CuArray{Int}(undef,(1024,1024))
@cuda threads = (32,32) blocks = (cld(1024,32),cld(1024,32)) arrayDef(a,1024)
@cuda threads = (32,32) blocks = (cld(1024,32),cld(1024,32)) arrayDef2(b,1024)
c = similar(a)
@cuda threads = (32,32) blocks = (cld(1024,32),cld(1024,32)) vadd(c,a,b)
c