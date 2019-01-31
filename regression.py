import numpy as np

def gradient_decent(X, y, w, alpha,  num_iters, train=None,
        weights=None, lambda_=2, verbose=True):

    """ function to determine optimal weights, w (r,g,b values) 
    by stochasticgradient decent """

    if train == 'r':
        w = weights.get('wr')
    
    if train == 'g':
        w = weights.get('wg')

    if train == 'b':
        w = weights.get('wb')

    m = len(y)

    for _ in range(num_iters):

        # update parameters
        for i in range(m):
        	w = w -  (alpha*compute_gradient(X[i], y[i], w, m, lambda_, train, weights)).reshape((len(w),1))

        if verbose:
            print(f"Loss = {round(compute_loss(X, y, w, m, lambda_, train, weights),2)}")

    return w


def compute_gradient(x,y,w,m,lambda_, train, weights):
    f = lambda x, w: sum([w[j]*x[j] for j in range(len(w))])
    gradient = (1/m)*(f(x, w)-y)*x
    gradient += (lambda_/m)*sum(w)
    if train == 'r':
        wg = weights.get('wg')
        wb = weights.get('wb')
        gradient += (1/m)*(0.21*f(x,w) + 0.71*f(x,wg) + 0.07*f(x,wb) - x[4])
    elif train == 'g':
        wr = weights.get('wr')
        wb = weights.get('wb')
        gradient += (1/m)*(0.21*f(x,wr) + 0.71*f(x,w) + 0.07*f(x,wb) - x[4])
    elif train == 'b':
        wr = weights.get('wr')
        wg = weights.get('wg')
        gradient += (1/m)*(0.21*f(x,wr) + 0.71*f(x,wg) + 0.07*f(x,w) - x[4])

    return gradient

def compute_loss(X, y ,w, m, lambda_, train, weights):
    """ function to compute mean squared error between
    predicted and actual """
    m = len(y)
    f = lambda x, w: sum([w[j]*x[j] for j in range(len(w))]) # predicted grayscale
    loss = []
    grayloss = 0
    for i in range(m):
        if train == 'r':
            wg = weights.get('wg')
            wb = weights.get('wb')
            grayloss += (1/2*m)*(0.21*f(X[i],w) + 0.71*f(X[i],wg) + 0.07*f(X[i],wb) - X[i][4])**2
        elif train == 'g':
            wr = weights.get('wr')
            wb = weights.get('wb')
            grayloss += (1/2*m)*(0.21*f(X[i],wr) + 0.71*f(X[i],w) + 0.07*f(X[i],wb) - X[i][4])**2
        elif train == 'b':
            wr = weights.get('wr')
            wg = weights.get('wg')
            grayloss += (1/m)*(0.21*f(X[i],wr) + 0.71*f(X[i],wg) + 0.07*f(X[i],w) - X[i][4])**2
        loss.append((f(X[i], w)-y[i])**2 + (lambda_/m)*sum(w) + grayloss)
    return np.sum(loss)
    

def grayscale(w):
    """ function to convert rgb to grayscale """
    rgb = np.array([0.21, 0.72, 0.07])
    return np.sum(np.round_(np.dot(rgb,w)))


if __name__ == "__main__":

    # y = grayscale values

    y = np.array([
        [91, 92, 93],
        [92, 93, 92],
        [93, 92, 91]
    ]).flatten().reshape((-1,1))
    # should give (approximately) rgb = [34, 116, 11] and grayscale = 92


    """
    y = np.array([
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255]
    ]).flatten().reshape((-1,1))
    # should give (approximately) rgb = [255, 255, 255] and grayscale = 255
    """

    """
    y = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]).flatten().reshape((-1,1))
    # should give (approximately) rgb = [0, 0, 0] and grayscale = 0
    """

    # X = rgb scaling values
    X = np.repeat([[0.21, 0.72, 0.07]], 9, axis=0)

    # w = the predicted r, g, b values
    w = np.zeros((3,1))

    alpha = 0.05
    num_iters = 1000
    ww = gradient_decent(X, y, w, alpha, num_iters, verbose=False)
    print("Final:")
    print(f"rgb: {np.round_(ww.flatten())}  " + \
        f"True Avg Grayscale: {round(y.mean())}  " + \
        f"Predicted Grayscale: {grayscale(ww)} ")