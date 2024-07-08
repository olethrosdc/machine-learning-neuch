# Let us now consider the gradient D_w I[a \neq y] 
# D_w I[a \neq y] = D_w sgn(a \neq y)
#

def cost(w):
    if (w > 0.5):
        return 1
    else:
        return 0

import numpy as np    
W = np.linspace(-1,1)
DW = W + 0.01
cost(W - DW)

