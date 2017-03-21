import numpy as np
from scipy.stats import kurtosis, norm, rankdata, boxcox
from scipy.optimize import fmin  # TODO: Explore efficacy of other opt. methods

np.seterr(all='warn')

from .helpers import w_d, w_t, inverse, igmm, delta_gmm, delta_init

class Gaussianize(object):
    """
    Gaussianize data using various methods.
    Conventions
    ----------
    This class is a wrapper that follows sklearn naming/style (e.g. fit(X) to train).
    In this code, x is the input, y is the output. But in the functions outside the class, I follow
    Georg's convention that Y is the input and X is the output (Gaussianized) data.
    Parameters
    ----------
    tol : float, default = 1e-4
    max_iter : int, default = 200
        Maximum number of iterations to search for correct parameters of Lambert transform.
    strategy : str, default='lambert'
        Possibilities are 'lambert'[1], 'brute'[2] and 'boxcox'[3].
    Attributes
    ----------
    taus : list of tuples
        For each variable, we have transformation parameters.
        For Lambert, e.g., a tuple consisting of (mu, sigma, delta), corresponding to the parameters of the
        appropriate Lambert transform. Eq. 6 and 8 in the paper below.
    References
    ----------
    [1] Georg Goerg. The Lambert Way to Gaussianize heavy tailed data with
                        the inverse of Tukey's h transformation as a special case
        Author generously provides code in R: https://cran.r-project.org/web/packages/LambertW/
    [2] Valero Laparra, Gustavo Camps-Valls, and Jesus Malo. Iterative Gaussianization: From ICA to Random Rotations
    [3] Box cox transformation and references: https://en.wikipedia.org/wiki/Power_transform
    """

    def __init__(self, tol=1.22e-4, max_iter=100, strategy='lambert'):
        self.tol = tol
        self.max_iter = max_iter
        self.strategy = strategy
        self.taus = []  # Store tau for each transformed variable

    def fit(self, x):
        x = np.asarray(x)
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        elif len(x.shape) != 2:
            print "Data should be a 1-d list of samples to transform or a 2d array with samples as rows."

        if self.strategy == 'lambert':
            for x_i in x.T:
                self.taus.append(igmm(x_i, tol=self.tol, max_iter=self.max_iter))
        elif self.strategy == 'brute':
            for x_i in x.T:
                self.taus.append(None)  # TODO: In principle, we could store parameters to do a quasi-invert
        elif self.strategy == 'boxcox':
            for x_i in x.T:
                self.taus.append(boxcox(x_i)[1])
        else:
            raise NotImplementedError


    def transform(self, x):
        x = np.asarray(x)
        if len(x.shape) == 1:
            x = x[:, np.newaxis]
        elif len(x.shape) != 2:
            print "Data should be a 1-d list of samples to transform or a 2d array with samples as rows."
        if x.shape[1] != len(self.taus):
            print "%d variables in test data, but %d variables were in training data." % (x.shape[1], len(self.taus))

        if self.strategy == 'lambert':
            return np.array([w_t(x_i, tau_i) for x_i, tau_i in zip(x.T, self.taus)]).T
        elif self.strategy == 'brute':
            return np.array([norm.ppf((rankdata(x_i) - 0.5) / len(x_i)) for x_i in x.T]).T
        elif self.strategy == 'boxcox':
            return np.array([boxcox(x_i, lmbda=lmbda_i) for x_i, lmbda_i in zip(x.T, self.taus)]).T
        else:
            raise NotImplementedError

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def invert(self, y):
        if self.strategy == 'lambert':
            return np.array([inverse(y_i, tau_i) for y_i, tau_i in zip(y.T, self.taus)]).T
        elif self.strategy == 'boxcox':
            return np.array([(1. + lmbda_i * y_i)**(1./lmbda_i) for y_i, lmbda_i in zip(y.T, self.taus)]).T
        else:
            print 'Inversion not supported for this gaussianization transform.'
            raise NotImplementedError

    def qqplot(self, x, prefix='qq'):
        """Show qq plots compared to normal before and after the transform."""
        import pylab
        from scipy.stats import probplot
        y = self.transform(x)

        for i, (x_i, y_i) in enumerate(zip(x.T, y.T)):
            probplot(x_i, dist="norm", plot=pylab)
            pylab.savefig(prefix + '_%d_before.png' % i)
            pylab.clf()

            probplot(y_i, dist="norm", plot=pylab)
            pylab.savefig(prefix + '_%d_after.png' % i)
            pylab.clf()
