# models.jl
# Methods for generating data and training Flux.jl models, and some prebuilt (but not trained) model architectures. 
# Requires: Flux, FFTW, SpecialPolynomials,utility_functions.jl,data_generation.jl

#----------------------------------------------------------------------
# Global parameters

const nfft = 100    # Grid size
m = 10000     # Number of data points
ft = plan_fft(zeros((nfft,)),flags=FFTW.MEASURE)    # Fourier transform

#----------------------------------------------------------------------
# Dataset generators

function supervised_dataset(N::Integer,m::Integer)
    # N is grid size, m is number of data points
    field_amps = [randomHGAmplitude(nfft) for i=1:m]
    y_data = [randomConvexPhase(nfft,Cauchy(0,(10^(-0.5-rand()) * pi/nfft))) for i=1:m]
    fft_amps = [abs.(ft*(field_amps[i] .* exp.(im*2*pi*y_data[i]))) for i=1:m]
    x_data_formatted = hcat((vcat(field_amps[i],fft_amps[i]) for i=1:m)...)
    y_data_formatted = hcat(y_data...)
    return x_data_formatted, y_data_formatted
end

function unsupervised_dataset(N::Integer,m::Integer)
    # N is grid size, m is number of data points
    slm_amps = [randomHGAmplitude(nfft,Normal(),x-> (x==0 ? 1 : 0.01)) for i=1:m]
    cam_amps = [randomHGAmplitude(nfft,Normal(),x-> (x==0 ? 1 : 0.1)) for i=1:m]
    return hcat( (vcat(slm_amps[i],cam_amps[i]) for i=1:m)... )
end

function format_conv_data(data)
    # Reshapes 1D data into the format required for convolutional layers. 
    return convert.(Float32,reshape(data,size(data)[1],1,size(data)[2]))
end

#----------------------------------------------------------------------
# Training loop

function train_and_monitor!(loss,model,batched_data,learning_rate,nit,xs,ys,interval=1)
    ls = []    # Loss, recorded every `interval` iterations. 
    for i=1:nit
        Flux.train!(loss,Flux.params(model),batched_data,Flux.Optimise.Adam(learning_rate))
        (i-1) % interval == 0 && append!(ls,loss(xs,ys))
    end
    return ls
end

#----------------------------------------------------------------------
# Supervised learning models
# Naming conventions: 
    # sMLP# for supervised MLP
    # sConv# for supervised convolutional network

# Baseline model
nHidden = 400
sMLP1 = Chain(Dense(2*nfft,nHidden,relu),Dense(nHidden,nHidden,relu),Dense(nHidden,nHidden,relu),
    Dense(nHidden,nHidden,relu),Dense(nHidden,nHidden,relu), Dense(nHidden,nfft))
loss_sMLP1(x,y) = Flux.Losses.mse(sMLP1(x),y)
# data_x,data_y = supervised_dataset(nfft,m)
# batched_data_sMLP1 = Flux.Data.DataLoader((data_x,data_y),batchsize=64)
# ls = train_and_monitor!(loss_sMLP1,sMLP1,batched_data_sMLP1,0.001,100,data_x,data_y,1)

# CNN
sCNN1 = Chain(Conv((5,), 1 => 8, relu; pad = SamePad()),Conv((5,), 8 => 16, relu; pad = SamePad()),MaxPool((4,)),
    Conv((5,), 16 => 32, relu; pad = SamePad()),Conv((5,), 32 => 64, relu; pad = SamePad()),MaxPool((5,)),
    Flux.flatten,Dense(640,640,relu),Dense(10*64,100))
loss_sCNN1(x,y) = Flux.Losses.mse(sCNN1(x),y)
# data_x,data_y = supervised_dataset(nfft,m)
# data_x = format_conv_data(data_x)
# batched_data_sCNN1 = Flux.Data.DataLoader((data_x,data_y),batchsize=64)
# ls = train_and_monitor!(loss_sCNN1,sCNN1,batched_data_sCNN1,0.001,100,data_x,data_y,1)

#----------------------------------------------------------------------
# Unsupervised learning models
# Naming conventions: 
    # uMLP# for supervised MLP
    # uConv# for supervised convolutional network

const D = DFTcentered(nfft);   # DFT matrix

# MLP
nHidden = 400
uMLP1 = Chain(Dense(2*nfft,nHidden,relu),Dense(nHidden,nHidden,relu),Dense(nHidden,nHidden,relu),
    Dense(nHidden,nHidden,relu),Dense(nHidden,nHidden,relu), Dense(nHidden,nfft))
loss_uMLP1(x,y) = Flux.Losses.mse( abs.(D * (y[1:nfft,:] .* exp.(2*pi*im*uMLP1(x)))) ,y[nfft+1:end,:])
# data = unsupervised_dataset(nfft,m)
# batched_data_uMLP1 = Flux.Data.DataLoader((data,data),batchsize=64)
# ls = train_and_monitor!(loss_uMLP1,uMLP1,batched_data_uMLP1,0.001,100,data,data,1)

# CNN
uCNN1 = Chain(Conv((5,), 1 => 8, relu; pad = SamePad()),Conv((5,), 8 => 16, relu; pad = SamePad()),MaxPool((4,)),
    Conv((5,), 16 => 32, relu; pad = SamePad()),Conv((5,), 32 => 64, relu; pad = SamePad()),MaxPool((5,)),
    Flux.flatten,Dense(10*64,100))
loss_uCNN1(x,y) = Flux.Losses.mse( abs.(D * (y[1:nfft,1,:] .* exp.(2*pi*im*uCNN1(x)))) ,y[nfft+1:end,1,:])
# data = format_conv_data(unsupervised_dataset(nfft,m))
# batched_data_uCNN1 = Flux.Data.DataLoader((data,data),batchsize=64)
# ls = train_and_monitor!(loss_uCNN1,uCNN1,batched_data_uCNN1,0.001,100,data,data,1)