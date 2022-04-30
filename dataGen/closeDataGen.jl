using ImageView
using Images
using CUDA
using CUDA.CUFFT
using Random
using Term

function particleInfo(pNum, datLen, zMin, zWid, cond)
    diam = randn(pNum)*10.0 .+ 50.0 # um
    x = rand(pNum)*datLen # pixel
    y = rand(pNum)*datLen # pixel
    z = sort(rand(pNum)*zWid .+ zMin) # um

    if cond == "close"
        z[2] = z[1]
        x[1] = rand()*datLen/2.0 + datLen/4.0
        y[1] = rand()*datLen/2.0 + datLen/4.0
        dis = (diam[1] + diam[2])/2.0*(1 + rand()*0.1)
        phase = rand() * 2pi
        x[2] = x[1] + dis*cos(phase)
        y[2] = y[1] + dis*sin(phase)
    end
    return diam,x,y,z
end

function CuObjPlane(rad, x0, y0, dx, datLen, Plane)
    x = (blockIdx().x-1)*blockDim().x + threadIdx().x
    y = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if x <= datLen && y <= datLen
        if (x0-Float32(x))^2 + (y0-Float32(y))^2 < (rad/dx)^2
            Plane[x,y] = 0
        else
            Plane[x,y] = 1
        end
    end
    return
end

function CuObjPlaneDouble(rad1, x1, y1, rad2, x2, y2, dx, datLen, Plane)
    x = (blockIdx().x-1)*blockDim().x + threadIdx().x
    y = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if x <= datLen && y <= datLen
        if (x1-Float32(x))^2 + (y1-Float32(y))^2 < (rad1/dx)^2 || (x2-Float32(x))^2 + (y2-Float32(y))^2 < (rad2/dx)^2
            Plane[x,y] = 0
        else
            Plane[x,y] = 1
        end
    end
    return
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
    dx = 10.0 # um
    wavLen = 0.6328 # um
    datLen = 2048 # pixel
    zMin = 120000.0 # um
    zWid = 100000.0 # um

    status = "train"
    cond = "close"
    pNum = 5

    println("closeDataGen.jl starting...")
    
    outputDir = "./fullhologram/" + status + "/" + cond + "/pNum$pNum/"
    mkpath(outputDir)

    threads = (32,32)
    blocks = (cld(datLen,32),cld(datLen,32))



    obj = CuArray{Float32}(undef,(datLen,datLen))
    sqr = CuArray{Float32}(undef,(datLen,datLen))
    trans = CuArray{ComplexF32}(undef,(datLen,datLen))
    holo = CuArray{ComplexF32}(undef,(datLen,datLen))

    diam,x,y,z = particleInfo(pNum, datLen/2, zMin, zWid, cond)

    @cuda threads = threads blocks = blocks CuTransSqr(datLen,wavLen,dx,sqr)

    if cond == "close"
        closePlaneIdx = rand(1:pNum-1)
        @cuda threads = threads blocks = blocks CuObjPlaneDouble(diam[1]/2,x[1],y[1],diam[2]/2,x[2],y[2],dx,datLen,obj)
        @cuda threads = threads blocks = blocks CuTransFunc(z[1], wavLen, datLen, sqr, trans)



    for idx in 1:pNum
        if cond == "close"
            @cuda threads = threads blocks = blocks CuObjPlaneDouble(diam[1])
        @cuda threads = threads blocks = blocks CuObjPlane(diams[idx]/2.0, xs[idx], ys[idx],dx,datLen,obj)
        @cuda threads = threads blocks = blocks Cu

    end

    @cuda threads = threads blocks = blocks CuObjPlane(50.0,512,512,dx,datLen,obj)
    @cuda threads = threads blocks = blocks CuTransFunc(z0,wavLen,datLen,sqr,trans)
    holo = fftshift(fft(obj)) .* trans
    holo = ifft(fftshift(holo))
    img = Array(real.(holo .* conj.(holo)))
    save("./test3.png",float.(img)/2)
end

# @time main()