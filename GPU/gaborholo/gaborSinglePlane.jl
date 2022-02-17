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
        Plane[x,y] = exp(2im*pi*(z0)/wavLen*sqrt(d_sqrPart[x,y]))
    end
    return
end

function CuUpdateImposed(datLen, input,imposed)
    x = (blockIdx().x-1)*blockDim().x + threadIdx().x
    y = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if x <= datLen && y <= datLen
        if input[x,y] < imposed[x,y]
            imposed[x,y] = input[x,y]
        end
    end
    return
end

function main()
    dx = 10.0
    zF = 120.0*1000 # init place
    datLen = 1024
    wavLen = 0.6328
    depth = 500
    dz = 50.0
    threads = (32,32)
    blocks = (cld(1024,32),cld(1024,32))

    img = cu(loadholo("./backrem.bmp"))

    vol = CuArray{Float32}(undef,(datLen,datLen,depth))
    sqr = CuArray{Float32}(undef,(datLen,datLen))
    transF = CuArray{ComplexF32}(undef,(datLen,datLen))
    transInt = CuArray{ComplexF32}(undef,(datLen,datLen))
    holo = CuArray{ComplexF32}(undef,(datLen,datLen))
    impImg = CUDA.ones(datLen,datLen)
    @cuda threads = threads blocks = blocks CuTransSqr(datLen,wavLen,dx,sqr)
    @cuda threads = threads blocks = blocks CuTransFunc(zF,wavLen,datLen,sqr,transF)
    @cuda threads = threads blocks = blocks CuTransFunc(dz,wavLen,datLen,sqr,transInt)

    holo = fftshift(fft(img)) .* transF
    for idx in 1:1000
        holo = holo .* transInt
        img = Float32.(abs.(ifft(fftshift(holo))))
        @cuda threads = threads blocks = blocks CuUpdateImposed(datLen,img,impImg)
        # save(string("./reconstInv/",lpad(idx,5,"0"),".bmp"),img)
    end
    save("./imposedImage2_1.bmp",Array(impImg))
end

@time main()