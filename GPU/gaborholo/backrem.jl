using Glob
using CUDA
using Images

function loadholo(path)
    out = Float32.(channelview(Gray.(load(path))))
end

function CuUpdateVote(vote, img, datLen)
    x = (blockIdx().x-1)*blockDim().x + threadIdx().x
    y = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if x <= datLen && y <= datLen
        vote[Int(img[x,y]*255)+1,x,y] += 1
    end
    return
end

function main()
    threads = (32,32)
    blocks = (cld(1024,32),cld(1024,32))

    files = glob("../../DropExp/exp220217/C001H001S0003/*.bmp")
    vote = CUDA.zeros(256,1024,1024)
    for item in files
        println(string("processing: ",item))
        tmpImg = cu(loadholo(item))
        @cuda threads = threads blocks = blocks CuUpdateVote(vote,tmpImg,1024)
    end
    img = Array{Float32}(undef,1024,1024)
    for j in 1:1024
        for i in 1:1024
            img[i,j] = (argmax(vote[:,i,j])-1)/255
        end
    end
    save("./background.bmp",img)

end

main()