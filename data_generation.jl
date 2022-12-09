# data_generation.jl
# Functions for generating phase retrieval data.
# Requires: SpecialPolynomials, Distributions

#----------------------------------------------------------------------
# General routines

function symAngFourierRange(n)
    # Grid for self-dual discrete Fourier transform of length n, angular frequency convention. 
    return (0:(n-1)) .* sqrt(2π/n) .- ((n-1)*sqrt(π/(2n)))
end
function symFourierRange(n)
    # Grid for self-dual discrete Fourier transform of length n.
    return (0:(n-1))/sqrt(n) .- (n-1)/(2*sqrt(n))
end

#----------------------------------------------------------------------
# Random 1D amplitudes

function HGmode(n::Integer,x::Number,z::Number,w0::Number,lambda::Number)
    # Hermite-Gaussian mode.
    zr = pi*w0^2/lambda    # Rayleigh range
    q = z + im*zr          # Complex beam parameter.
    w = w0 * sqrt(1+(z/zr)^2)    # Beam waist
    a = sqrt(sqrt(2/pi)/(2^n * factorial(n) * w0))
    b = sqrt(im*zr/q) * sqrt(-conj(q)/q)^n
    c = basis(Hermite,n)(sqrt(2)*x/w)
    d = exp(-im*pi*x^2/(lambda*q))
    return a*b*c*d
end


function randomHGAmplitude(N::Integer,modes,ampDist::Distribution,powerSpec,z::Number,w0::Number,lambda::Number)
    # Random amplitude, generated from a list of modes, a distribution `ampDist` from which to draw mode amplitudes, 
        # a power spectrum to multiply the mode amplitude by, and some standard Gaussian beam parameters. 
    xs = symFourierRange(N)
    f = sum(HGmode.(i,xs,z,w0,lambda)*rand(ampDist)*powerSpec(i) for i in modes)
    return abs.(f)/sqrt(sum(abs.(f).^2))
end
randomHGAmplitude(N::Integer,ampDist::Distribution,powerSpec) = randomHGAmplitude(N,(0:10...,),ampDist,powerSpec,0,1,1)
randomHGAmplitude(N::Integer) = randomHGAmplitude(N,(0:10...,),Cauchy(),x->1,0,1,1)

#----------------------------------------------------------------------
# Random convex phase, for supervised learning. 

function randomConvexPhase(N::Integer,d::Distribution)
    # The distribution d determines the second derivative of the phase at each point. 
    # The characteristic scale of the distribution d should be a bit smaller than pi/N,
        # in order to have an input and output beam both with characterstic length scale ~1. 
    d2p = abs.(rand(d,N))
    dp = cumsum(d2p)
    p = cumsum(dp .- dp[floor(Integer,(N+1)/2)])    #cumsum(dp .- dp[rand(1:N)])
    return p
end

#----------------------------------------------------------------------
# Random laser field, for supervised learning.

function randomEField(N::Integer,d::Distribution)
    # The scale of d should be about pi/N for an identity transformation
    f = randomHGAmplitude(N) .* exp.(2π*im*randomConvexPhase(N,d))
    return f
end

#----------------------------------------------------------------------
# Random 2D amplitudes

function random2DAmplitude(N::Integer,modes,ampDist::Distribution,powerSpec,z::Number,w0::Number,lambda::Number)
    # Inputs are the same as for randomHGAmplitude, except that now modes should be an iterable of pairs (n,m),
        # representing the mode along x and y directions. 
    xs = symFourierRange(N)
    ys = transpose(symFourierRange(N))
    f = sum(HGmode.(m[1],xs,z,w0,lambda)*rand(ampDist)*powerSpec(m[1])*powerSpec(m[2]) .* 
        HGmode.(m[2],ys,z,w0,lambda) for m in modes)
    return abs.(f)/sqrt(sum(abs.(f).^2))
end
random2DAmplitude(N::Integer,ampDist::Distribution,powerSpec) = random2DAmplitude(N,(((i,j) for i=0:5,j=0:5)...,),ampDist,powerSpec,0,1,1)
random2DAmplitude(N::Integer) = random2DAmplitude(N,(((i,j) for i=0:5,j=0:5)...,),Cauchy(),x->1,0,1,1)