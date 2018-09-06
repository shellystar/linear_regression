import numpy as np
import matplotlib.pyplot as plt

def polynomialModel(train_x, train_y, degree=9, epochs=10, learningRate=0.1):
    """
    select polynomial as base function 
    """
    N = len(train_x)

    # expand train_x so that each row contains a degree of all the input data
    train_x = np.expand_dims(train_x, axis=1)
    for i in range(2, degree+1):
        xi = np.array(np.power(train_x[:,0], i))
        train_x = np.concatenate((train_x, np.expand_dims(xi, axis=1)), axis=1)
    # add bias
    train_x = np.concatenate((train_x, np.ones((N,1), dtype=float)), axis=1)

    # parameters
    w = np.random.uniform(-1, 1, size=degree+1)

    # train loop
    for epoch in range(epochs):
        # calculate predict y
        predict_y = np.dot(train_x, w)

        # loss: squared error + L2 normalization
        loss = np.sum(np.power(predict_y - train_y, 2)) / 2.0 + np.sum(np.power(w, 2)) * 0.001
        if epoch % 1000 == 0:
            print ("Epoch: %04d Loss: %.4f"%(epoch, loss))

        # gradient desecent
        gradient = np.dot(predict_y - train_y, train_x) + 0.002 * w
        w = w - learningRate * gradient
    
    predict_y = np.dot(train_x, w)
    return w, predict_y
