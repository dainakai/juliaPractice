using ImageView
using Images
using CUDA
using CUDA.CUFFT

function loadholo(path)
    out = Float32.(channelview(Gray.(load(path))))
end

function CuTransSqr(datLen, wavLen, dx, Plane)
    x = (blockIdx().x-1)*blockDim().x + threadIdx().x
    y = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if x <= datLen && y <= datLen
        Plane[x,y] = 1.0 - ((x-datLen/2)*wavLen/datLen/dx)^2 - ((y-datLen/2)*wavLen/datLen/dx)^2
    end
    return
end

function CuTransFunc(z0, wavLen, datLen, d_sqrPart, Plane)
    x = (blockIdx().x-1)*blockDim().x + threadIdx().x
    y = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if x <= datLen && y <= datLen
        Plane[x,y] = exp(2im*pi*z0/wavLen*sqrt(d_sqrPart[x,y]))
    end
    return
end

function main()
    dx = 10.0
    zs = 45000.0
    ze = 55000.0
    datLen = 1024
    wavLen = 0.6328
    depth = 500
    dz = 20.0
    threads = (32,32)
    blocks = (cld(1024,32),cld(1024,32))

    img = cu(loadholo("./test3.png"))

    vol = CuArray{Float32}(undef,(datLen,datLen,depth))
    sqr = CuArray{Float32}(undef,(datLen,datLen))
    transZs = CuArray{ComplexF32}(undef,(datLen,datLen))
    transDz = CuArray{ComplexF32}(undef,(datLen,datLen))
    holo = CuArray{ComplexF32}(undef,(datLen,datLen))
    @cuda threads = threads blocks = blocks CuTransSqr(datLen,wavLen,dx,sqr)
    @cuda threads = threads blocks = blocks CuTransFunc(zs-dz,wavLen,datLen,sqr,transZs)
    @cuda threads = threads blocks = blocks CuTransFunc(dz,wavLen,datLen,sqr,transDz)

    holo = fftshift(fft(img)) .* transZs

    for idx in 1:Int((ze-zs)/dz)
        holo = holo .* transDz
        intout = fftshift(ifft(holo))
        vol[:,:,idx] = Float32.(abs.(intout))
        println("$idx was processed")
    end
end

@time main()