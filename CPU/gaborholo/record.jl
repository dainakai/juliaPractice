using ImageView
using Images
using FFTW

function objectPlane(rad,x0,y0,dx,datLen)
    # x and y are array indexes, not coordinates.
    plane = ones(UInt8,datLen,datLen)
    for yi in 1:datLen
        for xi in 1:datLen
            if (xi-x0)^2 + (yi-y0)^2 < (rad/dx)^2
                plane[xi,yi] = 0
            end
        end
    end
    return plane
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


dx = 10.0
z0 = 50000.0
datLen = 1024
wavLen = 0.6328

obj = objectPlane(50.0,512,512,10.0,1024)
sqr = transSqr(datLen,wavLen,dx)
trans = transFunc(z0,wavLen,datLen,sqr)
holo = fftshift(fft(obj)) .* trans
holo = ifft(fftshift(holo))
img = real.(holo .* conj.(holo))
save("./test.png",float.(img)/2)
