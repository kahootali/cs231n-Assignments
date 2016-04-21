import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # might need to reshape the input into rows.                                 #
  #############################################################################
  N = x.shape[0]
  out = x.reshape(N, np.prod(x.shape[1:])).dot(w)+b

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  N = x.shape[0]
  dx = dout.dot(w.T).reshape(x.shape)
  dw = x.reshape(N, np.prod(x.shape[1:])).T.dot(dout)
  db = np.sum(dout, axis=0)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  check=(x>0)
  out = x*check

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  check=(x>0)
  dx=dout*check
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx


def im2col(A, B, skip=(1,1)):
  """
    A naive implementation of im2col that split
    the given image "A" into tiles of "B" with
    skipping provided by the skip Parameter

    A = np.random.randint(0,9,(4,4)) # Sample input array
                        # Sample blocksize (rows x columns)
    B = [2,2]
    skip=[2,2]
   """
  
  # Parameters
  # Parameters
  D,M,N = A.shape
  col_extent = N - B[1] + 1
  row_extent = M - B[0] + 1


  # Get Starting block indices
  start_idx = np.arange(B[0])[:,None]*N + np.arange(B[1])

  # Get Depth indeces
  cidx=M*N*np.arange(D)
  start_idx=(cidx[:,None]+start_idx.ravel()).reshape((-1,B[0],B[1]))

  # Get offsetted indices across the height and width of input array
  offset_idx = np.arange(row_extent)[:,None]*N + np.arange(col_extent)

  # Get all actual indices & index into input array for final output
  out = np.take (A,start_idx.ravel()[:,None] + offset_idx[::skip[0],::skip[1]].ravel())

def im2colidx(A, B, skip=(1,1)):
  """
    Same as im2col but returns the index 
   """
  
  # Parameters
  # Parameters
  D,M,N = A.shape
  col_extent = N - B[1] + 1
  row_extent = M - B[0] + 1


  # Get Starting block indices
  start_idx = np.arange(B[0])[:,None]*N + np.arange(B[1])

  # Get Depth indeces
  cidx=M*N*np.arange(D)
  start_idx=(cidx[:,None]+start_idx.ravel()).reshape((-1,B[0],B[1]))

  # Get offsetted indices across the height and width of input array
  offset_idx = np.arange(row_extent)[:,None]*N + np.arange(col_extent)

  # Get all actual indices & index into input array for final output
  out_idx=start_idx.ravel()[:,None] + offset_idx[::skip[0],::skip[1]].ravel()
  
  return out_idx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  # Hint: Furthermore you can look at np.ravel_index, np.unravel and np.take functions #
  #############################################################################
  pad = conv_param['pad']
  stride = conv_param['stride']
  F, C, HH, WW = w.shape
  N, C, H, W = x.shape
  Hp = 1 + (H + 2 * pad - HH) / stride
  Wp = 1 + (W + 2 * pad - WW) / stride
  out = np.zeros((N, F, Hp, Wp))

  #print out.shape
  
  padded = np.pad(x, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')
 
  for i in xrange(N):		
    for j in xrange(F):      
      for k in xrange(Hp):		
	    #print k
        he = k * stride			
        for l in xrange(Wp):		
          wi = l * stride		            
          win = padded[i, :, he:he+HH, wi:wi+WW]	
          out[i, j, k, l] = np.sum(win*w[j]) + b[j]		
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive, where
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  # number of images X number of Filters X Filter Rows X Filter Height...

  # db is sum of all the derivaties of each filter for all the examples...
  # sum over examples, then sum over rows, sum over columns...

  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  x, w, b, conv_param = cache
  pad = conv_param['pad']
  stride = conv_param['stride']
  F, C, HH, WW = w.shape
  N, C, H, W = x.shape
  Hp = 1 + (H + 2 * pad - HH) / stride
  Wp = 1 + (W + 2 * pad - WW) / stride

  dx = np.zeros_like(x)
  dw = np.zeros_like(w)
  db = np.zeros_like(b)
  
  padded = np.pad(x, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')
  padded_dx = np.pad(dx, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')

  for i in xrange(N): 
    for j in xrange(F):      
      for k in xrange(Hp):
        he = k * stride
        for l in xrange(Wp):
          wi = l * stride          
          win = padded[i, :, he:he+HH, wi:wi+WW]
          
          db[j] += dout[i, j, k, l]
          dw[j] += win*dout[i, j, k, l]
          padded_dx[i, :, he:he+HH, wi:wi+WW] += w[j] * dout[i, j, k, l]
  
  dx = padded_dx[:, :, pad:pad+H, pad:pad+W]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  HH = pool_param['pool_height']
  WW = pool_param['pool_width']
  stride = pool_param['stride']
  N, C, H, W = x.shape
  Hp = 1 + (H - HH) / stride
  Wp = 1 + (W - WW) / stride

  out = np.zeros((N, C, Hp, Wp))

  for i in xrange(N):
    for j in xrange(C):
      for k in xrange(Hp):
        he = k * stride
        for l in xrange(Wp):
          wi = l * stride
          
          win = x[i, j, he:he+HH, wi:wi+WW]
          out[i, j, k, l] = np.max(win)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x, pool_param = cache
  HH = pool_param['pool_height']
  WW = pool_param['pool_width']
  stride = pool_param['stride']
  N, C, H, W = x.shape
  Hp = 1 + (H - HH) / stride
  Wp = 1 + (W - WW) / stride

  dx = np.zeros_like(x)

  for i in xrange(N):
    for j in xrange(C):
      for k in xrange(Hp):
        he = k * stride
        for l in xrange(Wp):
          wi = l * stride

          win = x[i, j, he:he+HH, wi:wi+WW]
          t = np.max(win)

          dx[i, j, he:he+HH, wi:wi+WW] += (win == t) * dout[i, j, k, l]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx

