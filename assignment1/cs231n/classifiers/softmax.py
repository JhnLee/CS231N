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
  N = X.shape[0]
  C = W.shape[1]
  D = W.shape[0]

  for i in range(N):
    scores = X[i].dot(W) #(C, )
    scores -= np.max(scores) #for numerical stability (http://cs231n.github.io/linear-classify/#softmax)
    p = np.exp(scores) / np.sum(np.exp(scores)) #(C, )
    
    loss -= np.log(p[y[i]])
    
    #dW
    p[y[i]] -= 1
    for j in range(C):
        dW[:,j] += X[i,:] * p[j]
    
  loss /= N
  dW /= N
    
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
        
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
  N = X.shape[0]
  C = W.shape[1]
    
  scores = np.dot(X, W) #(N,C)
  scores -= np.max(scores)#for numerical stability
  scores = np.exp(scores)
  correct_score = scores[np.arange(N),y] #(N, )
  p = scores / np.sum(scores, axis = 1, keepdims = True) #(N, C)
  loss -= np.sum(np.log(p)[np.arange(N), y])
  loss /= N
  loss += 0.5*reg*np.sum(W*W)

  #dW  (D, C)
  dscores = p
  dscores[np.arange(N), y] -= 1  #(N, C)
  dW = np.dot(X.T, dscores) / N #(D, C)          
  dW += reg*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

