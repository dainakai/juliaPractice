using Images

function loadholo(path)
    out = Float32.(channelview(Gray.(load(path))))
end

function main()
    img1 = loadholo("./imposedImage2.bmp")
    img2 = loadholo("./imposedImage2_1.bmp")
    out = img1 .- img2
    for j in 1:1024
        for i in 1:1024
            if abs(out[i,j]) > 0.1
                println("x: ",i," y: ",j," value: ", Int64(floor(abs(out[i,j])*255)))
            end
        end
    end
end

main()