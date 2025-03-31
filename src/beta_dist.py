import torch
from torch.special import beta, betainc, betaincinv
import logging

class Beta():
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def from_sample(x):
        '''Estimate parameters for Beta distribution

        :param x: 1d torch array of samples
        :return: two singular torch array of a and beta
        '''

        #Based off of 2210.05536v1.pdf, PAGE 7


        xs = x
        ys = 1 - xs

        _get_means = lambda v: \
            (torch.mean(v), \
            torch.mean(torch.log(v)), \
            torch.mean(torch.xlogy(v, v)))
        mx, mlnx, mxlnx = _get_means(xs)
        my, mlny, mylny = _get_means(ys)

        a = mx/(mxlnx - mx*mlnx + mylny - my*mlny)
        b = my/(mxlnx - mx*mlnx + mylny - my*mlny)

        return Beta(a, b)

    def cdf(self, x):
        '''
        Return CDF of beta distribution

        :param x: threshold
        :return: torch percentile
        '''

        #NOTE must autograd for pr/pa pr/pb and pr/px (three param function)
        return betainc(self.a, self.b, x)

    def icdf(self, x):
        '''
        Return CDF of beta distribution

        :param x: percentile
        :return: torch of threshold
        '''

        #NOTE must autograd for pr/pa pr/pb and pr/px (three param function)
        return betaincinv(self.a, self.b, x)


