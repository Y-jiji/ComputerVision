from builtins import range
import numpy as np
from random import shuffle
# from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero
    #############################################################################
    # TODO：
    # 计算损失函数的梯度并将其存储为dW。
    # 与其先计算损失再计算梯度，还不如在计算损失的同时计算梯度更简单。
    # 因此，您可能需要修改上面的一些代码来计算梯度。
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:,j] += X[i] # dW计算
                dW[:,y[i]] += -X[i] # dW计算

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    dW += 2 * reg * W

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    nbatch = X.shape[0]
    nclass = W.shape[1]

    #############################################################################
    # TODO: 
    # 实现一个向量化SVM损失计算方法,并将结果存储到loss中
    #############################################################################
    mask_true_type = (np.cumsum(np.ones(shape=(nbatch, nclass)), axis=-1) == (y+1).reshape(nbatch, 1))

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    score = np.dot(X, W)                                                            # score[i, j] : score for sample n in class j
    loss  = score - (score*mask_true_type).sum(-1).reshape(nbatch, 1) + 1           # loss[i, j] : score[i, j] - score[i, y[i]] + 1
    mask  = (loss > 1e-12) * 1                                                      # mask[i, j] : loss[i, j] > 0
    loss_sum = ((loss * mask).sum() - nbatch) / nbatch + np.sum(reg * W * W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                              
    # 实现一个向量化的梯度计算方法,并将结果存储到dW中                       
    # 提示:与其从头计算梯度,不如利用一些计算loss时的中间变量                                    
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # dscore[i, j] : mask[i, j] - (j==y[i]) * (\sum_t mask[i, t])
    dscore = mask - mask_true_type * mask.sum(-1).reshape(nbatch, 1)
    # dW[i, j] : \sum_n dscore[n, j] * X[n, i] (something stupid will happen with einsum)
    dW = np.matmul(dscore.transpose(1, 0), X / nbatch).transpose(1, 0) + reg * 2 * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss_sum, dW

if __name__ == '__main__':
    import time
    for p in range(2, 8):
        n = 5*int(10**p) + 1
        X = np.random.randn(n, 32*32*5)
        y = np.random.randint(10, size=(n,))
        W = np.random.randn(32*32*5, 10)
        start = time.time()
        loss, grad = svm_loss_vectorized(W, X, y, 5)
        end   = time.time()
        print(f'vector time: {end - start}')
        start = time.time()
        loss, grad = svm_loss_naive(W, X, y, 5)
        end   = time.time()
        print(f'naive time: {end - start}')
