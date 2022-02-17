using Images
using Statistics

function loadholo(path)
    out = Float32.(channelview(Gray.(load(path))))
end

function main()
    back = loadholo("./background.bmp")
    img = loadholo("./particle.bmp")
    img = img .- back
    img .+= mean(back)
    save("./backrem.bmp",img)
end

main()