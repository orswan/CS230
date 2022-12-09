# utility_functions.jl
# General purpose functions.
# Requires: FFTW

#--------------------------------------------------------------------------------------
# Custom DFT matrices
DFTmatrix(N::Integer) = [exp(-2π*im*i*j/N)/sqrt(N) for i=0:(N-1),j=0:(N-1)]
DFTcentered(N::Integer) = [exp(-2π*im*i*j/N)/sqrt(N) for i=ceil(Int,-(N-1)/2):ceil(Int,(N-1)/2),j=0:(N-1)]

#--------------------------------------------------------------------------------------
# Phase retrieval error

function PRerror(g::Array{T,N},G::Array{T,N},psi::Array{T,N},ft) where N where T<:Real
    # ft is an FFT object.  Use this method if you already have an FFT computed.
    sum(abs2.( abs.( ft*(g .* exp.(2*pi*im*psi)) ./ sqrt(*(size(g)...)) ) - G ))
end
function PRerror(g::Array{T,N},G::Array{T,N},psi::Array{T,N}) where N where T<:Real
    # Use this method to automatically get the FFT. 
    sum(abs2.( abs.( fft*(g .* exp.(2*pi*im*psi)) ./ sqrt(*(size(g)...)) ) - G ))
end

#--------------------------------------------------------------------------------------
# Gerchberg-Saxton algorithm

function unitize(x::Number) 
    # Turns arbitrary number into complex unit vector
    if x==0
        return zero(ComplexF64)
    else
        return convert(ComplexF64,x/abs(x))
    end
end

function GSA(g::Array{T,N},G::Array{T,N},nit::Integer,ft,ift) where N where T<:Real
    # `ampIn` is input beam amplitude, `ampOut` is output beam amplitude, 
    # `nit` is number of iterations, ft is an FFT object, ift is an IFFT object.
    phi = randn(size(g))
    for _ = 1:nit
        phi = unitize.( ift*(G .* unitize.(ft*(g .* phi))) )
    end
    return angle.(phi) ./ (2*pi)
end