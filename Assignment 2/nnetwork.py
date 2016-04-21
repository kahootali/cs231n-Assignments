import numpy as np
import math
from layers import *

#####################################################################
# TODO: Implement init_three_layer_neuralnet and 
# three_layer_neuralnetwork functions
#####################################################################


def init_three_layer_neuralnet(weight_scale=1, bias_scale=0, input_feat_dim=786,
                           num_classes=10, num_neurons=(20, 30)):
  """
  Initialize the weights for a three-layer NeurAlnet.

  Inputs:
  - weight_scale: Scale at which weights are initialized. Default 1e-3.
  - bias_scale: Scale at which biases are initialized. Default is 0.
  - input_feat_dim: number of features of input examples..
  - num_classes: The number of classes for this network. Default is 10   (for MNIST)
  - num_neurons: A tuple containing number of neurons in each layer...
  
  Returns:
  A dictionary mapping parameter names to numpy arrays containing:
    - W1 (input_feat_dim,num_neurons[0]), b1: Weights and biases for the affine layer
    - W2, b2: Weights and biases for the affine layer
    - W3, b3: Weights and biases for the affine layer    
  """
  
  assert len(num_neurons)  == 2, 'You must provide number of neurons for two layers...'

  model = {}
  #model['W1'] = np.random.randn((num_neurons[0],(input_feat_dim) * weight_scale) * math.sqrt(2.0/input_feat_dim)) # Initialize from a Gaussian With scaling of sqrt(2.0/fanin)
  
  model['W1'] = (np.random.rand(input_feat_dim,num_neurons[0])*weight_scale) * math.sqrt(2.0/input_feat_dim)
  model['b1'] = np.zeros(num_neurons[0])# Initialize with zeros
  
  #model['W2'] = (np.random.randn(input_feat_dim) * weight_scale) * math.sqrt(2.0/input_feat_dim)# Initialize from a Gaussian With scaling of sqrt(2.0/fanin)
  #print ((model['W1'])[0,:]).shape
  #numcols = len(input[0])
  t=len((model['W1'])[0])
  #print t
  model['W2'] = (np.random.rand(num_neurons[0],num_neurons[1])*weight_scale) * math.sqrt(2.0/t)
  model['b2'] = np.zeros(num_neurons[1])# Initialize with zeros

  t=len((model['W2'])[0])
  #model['W3'] = (np.random.randn(input_feat_dim) * weight_scale) * math.sqrt(2.0/input_feat_dim)# Initialize from a Gaussian With scaling of sqrt(2.0/fanin)
  model['W3'] = (np.random.rand(num_neurons[1],num_classes)*weight_scale) * math.sqrt(2.0/t)
  model['b3'] = np.zeros(num_classes)# Initialize with zeros

  return model



def three_layer_neuralnetwork(X, model, y=None, reg=0.0,verbose=0):
  """
  Compute the loss and gradient for a simple three-layer NeurAlnet. The architecture
  is affine-relu-affine-relu-affine-softmax. We use L2 regularization on 
  for the affine layer weights.

  Inputs:
  - X: Input data, of shape (N,D), N examples of D dimensions
  - model: Dictionary mapping parameter names to parameters. A three-layer Neuralnet
    expects the model to have the following parameters:
    - W1, b1: Weights and biases for the affine layer
    - W2, b2: Weights and biases for the affine layer
    - W3, b3: Weights and biases for the affine layer
    
  - y: Vector of labels of shape (D,). y[i] gives the label for the point X[i].
  - reg: Regularization strength.

  Returns:
  If y is None, then returns:
  - scores: Matrix of scores, where scores[i, c] is the classification score for
    the ith input and class c.

  If y is not None, then returns a tuple of:
  - loss: Scalar value giving the loss.
  - grads: Dictionary with the same keys as model, mapping parameter names to
    their gradients.
  """
  
  # Unpack weights
  W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'],model['W3'],model['b3']
  N,D= X.shape

  assert W1.shape[0] == D, ' W1 2nd dimenions must match number of features'
  
  dW1,dW2,dW3,db1,db2,db3=np.zeros_like(W1),np.zeros_like(W2),np.zeros_like(W3),np.zeros_like(b1),np.zeros_like(b2),np.zeros_like(b3)
  # Compute the forward pass
  
  '''
  AffineLayer = X.dot(W1)+b1  
  ReluLayer,_ = relu_forward(AffineLayer)
  AffineLayer2 = ReluLayer.dot(W2) + b2
  ReluLayer2,_ = relu_forward(AffineLayer2)
  AffineLayer3 = ReluLayer2.dot(W3) + b3
  scores = AffineLayer3
  
  print X.shape
  print W1.shape
  print b1.shape
  print W2.shape
  print b2.shape
  print W3.shape
  print b3.shape
  '''
  affine_out1,cache1 = affine_forward(X, W1, b1)
  relu_out1,cache_relu1 = relu_forward(affine_out1)
  
  affine_out2,cache2 = affine_forward(relu_out1, W2, b2)
  relu_out2,cache_relu2 = relu_forward(affine_out2)
  
  affine_out3,cache3 = affine_forward(relu_out2, W3, b3)
  scores = affine_out3

  #if verbose:
    #print ['Layer {} Variance = {}'.format(i+1, np.var(l[:])) for i,l in enumerate([a1, a2, cache3[0]])][:]
  if y is None:
    return scores
  data_loss,d_softmax = softmax_loss(scores,y)
  data_loss += reg * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))
  '''
  max_scores = np.max(scores)
  scores -= max_scores
  correct_class_scores = scores[y,np.arange(N)]
  exp_score = np.exp(scores)
  sumexp = np.sum(exp_score,axis=0)
  loss_i = -correct_class_scores + np.log(sumexp)
  loss = np.sum(loss_i) / N  
  '''  	
  # Compute the backward pass
  
  d_affine_out3, dW3, db3 = affine_backward(d_softmax, cache3)  
  d_relu2 = relu_backward(d_affine_out3, cache_relu2)
  
  d_affine_out2, dW2, db2 = affine_backward(d_relu2, cache2)  
  d_relu1 = relu_backward(d_affine_out2, cache_relu1)
  
  d_affine_out1, dW1, db1 = affine_backward(d_relu1, cache1)    
    
  #
  reg_loss = 0

  loss = data_loss + reg_loss
  grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2,'W3':dW3,'b3':db3}
  
  return loss, grads

