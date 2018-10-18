import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
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
  D = W.shape[0]
  C = W.shape[1]
  N = X.shape[0]
    
  score = np.zeros((N,C))
  out = np.zeros(N)
    
  for i in range(N):
      for j in range(C):
          for k in range(D):
              score[i,j] += X[i,k] * W[k,j]
      score[i] = np.exp(score[i])
      score[i] /= np.sum(score[i])
      out[i] += score[i,y[i]]

      #gradient
      # dw_j = 1/num_train * \sum_i[x_i * (p(y_i = j)-Ind{y_i = j} )]
      score[i,y[i]] -= 1
      for j in range(C):
          for k in range(D):
              dW[k,j] += X[i,k]*score[i,j]
    
  loss -= np.sum(np.log(out)) 
  loss /= N 
  loss += reg * np.sum(W*W) * 0.5

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
  D = W.shape[0]
  C = W.shape[1]
  N = X.shape[0]

  score = np.zeros((N,C))
  out = np.zeros(N)
    
  score = np.exp(X.dot(W)) #(N, C) / X -> (N, D) / W -> (D, C)
  score /= np.sum(score, axis = 1, keepdims=True)
  out += score[np.arange(N),y]
  loss -= np.sum(np.log(out))
  loss /= N
  loss += reg * np.sum(W*W) * 0.5

  score[np.arange(N),y] -= 1
  dW = X.T.dot(score) #(D, C)
  dW /= N
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

