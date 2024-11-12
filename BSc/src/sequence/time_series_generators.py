import numpy as np


class GaussianTimeSeries:
    """A Gaussian time series.

    This generates data from the distribution
    $$x_t =  x_{t-1} +  \epsilon_t$$,
    where $\epsilon_t \sim N(0, \sigma^2)$.

    Atrributes
    ----------
    scale : float
       The amount $\sigma$ by which to scale the noise.


    Methods
    -------

    __init__(scale)
        Initialise with specific scale

    generate()
        Generate a value from the time series

    """
    def __init__(self, scale):
        """Initialise the time series."""
        self.state = 0
        self.scale = scale
        
    def generate(self):
        """Generate a value from the time series."""
        
        x = self.state + self.scale * np.random.normal()
        self.state = x
        return x
    
class LinearGaussianTimeSeries:
    """A Gaussian time series.

    This generates a series from the distribution
    $$x_t = \sum_{i=1}^{n} w_i x_{t-i} + \epsilon_t$$,
    where $\epsilon_t \sim N(0, \sigma^2)$.

    The coefficients $w_i$ are randomly initialised in the range $[-0.5, 0.5]$.

    Atrributes
    ----------
    order : int
       The order $n$ of the dependency in the past
    scale : float
       The amount $\sigma$ by which to scale the noise.


    Methods
    -------

    __init__(scale)
        Initialise with specific scale

    generate()
        Generate a value from the time series

    """
    def __init__(self, scale, order):
        self.order = order
        self.state = np.zeros(order)
        self.scale = scale
        self.coeffs = np.random.uniform(size=order) - 0.5
        
    def generate(self):
        x = np.dot(self.state, self.coeffs) + self.scale * np.random.normal()
        self.state[:-1] = self.state[1:]
        self.state[-1] = x
        return x
    
class DiscreteTimeSeries:
    """A Gaussian time series.

    This generates a series from the distribution
    $$x_t \sim Mult(p_t)$$, 
    with $x_t \in \{1, \ldots, k\}$, where
    $$p_t = \theta_{x_{t-1}, ldots, x_{t-n}}$$,
    are multinomial coefficients.

    There is a different $k$-sized vector of multinomial coefficients
    The coefficients $w_i$ are randomly initialised in the range $[-0.5, 0.5]$.

    Atrributes
    ----------
    n_symbols : int
       $k$: The amount of symbols in the alphabet

    order : int
       $n$: The order of the dependency in the past


    Methods
    -------

    __init__(scale)
        Initialise with specific scale

    generate()
        Generate a value from the time series

    """
    def __init__(self, n_symbols, order):
        self.state = np.zeros(order, dtype=int)
        self.n_symbols = n_symbols
        self.order = order
        shape = np.ones(order, dtype=int) * n_symbols
        alpha = np.ones(n_symbols)
        self.transitions = np.random.dirichlet(alpha, size=shape)
    def generate(self):
        P = self.transitions[tuple(self.state)]
        x = np.random.choice(self.n_symbols, p = P)
        self.state[:-1] = self.state[1:]
        self.state[-1] = x
        return x


if __name__ == '__main__':
    T = 200

    gts = GaussianTimeSeries(0.1)
    x = np.zeros(T)
    for t in range(T):
        x[t] = gts.generate()
    print(x)

    import matplotlib.pyplot as plt
    plt.plot(x)
    plt.title("Gaussian time series")
    plt.show()

    for order in range(1,4):
        lgts = LinearGaussianTimeSeries(0.1, order)
        for t in range(T):
            x[t] = lgts.generate()
        plt.plot(x)
    plt.title("Linear-Gaussian time series")
    plt.legend(["1", "2", "3"])
    plt.show()


    dts = DiscreteTimeSeries(2, 3)
    for t in range(T):
        x[t] = dts.generate()
    print(x)








