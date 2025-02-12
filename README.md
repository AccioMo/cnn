# To Start...
the chain rule allows us to compute gradients of nested or composite functions:
for example: for a composite function `L = f(g(h(x)))`, its gradient with respect to `x` is:
	ðL/ðx = ðL/ðf * ðf/ðg * ðg/ðh * ðh/ðx
in a neural network, our goal is to get to the gradient: ðL/ðx

in the context of NNs, we often have what i called a delta, which is an intermediate quantity used to calculate both loss with respect to weights, and loss with repect to the inputs (ðL/ðw and ðL/ðx respectively), where:

	ðL/ðw = delta • x.T (dot product of delta and the layer inputs transposed)
		and
	ðL/ðx = w.T • delta (dot product of weights transposed and delta)

ðL/ðw is used to update the weights:
	W-new = W-old - n • ðL/ðw		where `n` is learning rate
this allows us to optimize the gradient descent

ðL/ðx is the backpropagated error; during backpropagation, we calculate the error `e`, where:
	e-l = delta-(l+1) • w-(l+1).T		where `l` is the current layer
we then use this error to calculate delta for this layer, and the process continues until we reach the earliest layer.

# In This Network...
for each layer from l = 0 to l = n - 1 (where n is the number of layers: `_size`), i first implement a feedforward function which does the following:
	z = x • w + b	where:
		`z` is the pre-activation output
		`x` is the input matrix to the layer
		'w' is the weight matrix
		`b` is the bias
	then:
	a = f(z)	where:
		`a` is the post activation output
		`f()` is the activation function (ReLU, sigmoid, tanh, ...)
keep in mind that for each consequent layer, `x = a-(l-1)`
meaning the inputs to each layer is the activated output of the one before it.

following that is a backpropagation function, which, starting from the last layer and moving backwards, does the following:
	

