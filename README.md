"# CS230" 
The main code is divided into three .jl files:
	- data_generation.jl contains functions for producing random amplitudes and phases.
	- utility_functions.jl contains functions for differentiable DFTs, phase retrieval error computation, and the Gerchberg-Saxton algorithm.
	- models.jl defines several network architectures, along with functions for formatting data to be compatible with those networks.  

The training of each of the models in models.jl is carried out in three notebooks--one containing the two supervised models, and another for each of the unsupervised models. 
The trained models are saved in .bson files.  To load the models, use e.g.
	using BSON: @load
	@load "uMLP1.bson" uMLP