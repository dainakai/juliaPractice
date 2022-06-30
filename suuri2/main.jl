using Statistics

function func(start::Int, n::Int)
    ans::Int = 0
    now = start
    while true
        if now == 0
            break
        end
        if now == n
            s = rand([1,2])
            if s == 1
                now = now-1
                ans = ans+1
            else
                ans = ans+1
            end
        else
            s = rand([1,2])
            if s == 1
                now = now -1
                ans = ans+1
            else
                now = now+1
                ans = ans+1
            end
        end
    end
    return ans
end



function main()
    n = 5
    counts = 300
    array = zeros(n+1,counts)


    for itr in 1:counts
        for start in 0:n
            array[start+1,itr] = func(start,n)
        end
    end

    meanSD = zeros(n+1,2)
    for start in 0:n
        meanSD[start+1,1] = mean(array[start+1,:])
        meanSD[start+1,2] = std(array[start+1,:])
    end

    display(meanSD)
    
end

# main()