using Images
using FFTW

function loadholo(path)
    out = Float32.(channelview(Gray.(load(path))))
end

function transSqr(datLen,wavLen,dx)
    plane = Array{Float32}(undef,datLen,datLen)
    for yi in 1:datLen
        for xi in 1:datLen
            plane[xi,yi] = 1.0 - ((xi-datLen/2)*wavLen/datLen/dx)^2 - ((yi-datLen/2)*wavLen/datLen/dx)^2 
        end
    end
    return plane
end

function transFunc(z0, wavLen, datLen, sqrPart)
    out = Array{ComplexF32}(undef,datLen,datLen)
    out = exp.(2im*pi*z0/wavLen*sqrt.(sqrPart))
    return out
end

function main()
    dx = 10.0
    zs = 45000.0
    ze = 55000.0
    datLen = 1024
    wavLen = 0.6328
    depth = 500
    dz = 20.0

    img = loadholo("./test.png")

    vol = Array{Float32}(undef,datLen,datLen,depth)
    sqr = transSqr(datLen,wavLen,dx)
    transZs = transFunc(zs-dz,wavLen,datLen,sqr)
    transDz = transFunc(dz,wavLen,datLen,sqr)
    holo = fftshift(fft(img)) .* transZs

    for idx in 1:Int((ze-zs)/dz)
        holo = holo .* transDz
        intout = fftshift(ifft(holo))
        vol[:,:,idx] = Float32.(abs.(intout))
        println("$idx was processed")
    end
end

main()