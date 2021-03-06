import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[1]
  scores = np.dot(W,X)
  scores_max = np.max(scores, axis=0)
  scores -= scores_max
  exp_scores = np.exp(scores)
  sumexp = np.sum(exp_scores, axis=0)
  log_sums = np.log(sumexp)
  correct_class_scores = np.array([scores[y[i], i] for i in xrange(X.shape[1])])
  loss_i = -correct_class_scores + log_sums
  total_loss = np.sum(loss_i)
  loss =  total_loss/ N
  loss += reg * np.sum(W * W)   
  for i in xrange(N):
    dW_i = np.exp(scores[:,i]) / sumexp[i]
    dW_i[y[i]] -= 1
    dW += np.outer(dW_i, X[:,i])

  dW /= N
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[1]
  scores = np.dot(W,X)
  max_scores = np.max(scores)
  scores -= max_scores
  correct_class_scores = scores[y,np.arange(N)]
  exp_score = np.exp(scores)
  sumexp = np.sum(exp_score,axis=0)
  loss_i = -correct_class_scores + np.log(sumexp)
  loss = np.sum(loss_i) / N
  loss += reg * np.sum(W*W)

  dW = np.exp(scores) / sumexp
  dW[y, np.arange(N)] -= 1
  dW = np.dot(dW,X.T)
  
  dW /= N
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
