"""
Unified inference algorithm for loopy belief propagation and dual decomposition in factor graphs.
"""
from __future__ import division

from numpy import zeros, zeros_like, array
from collections import defaultdict

# TODO: backtrack-free search for optimal assignment to resolve issue ties for
# the optimal assignment.

def solve(dd, fg, maxiter=50, a=1.0, A=1, gamma=1.0, verbose=1):

    fg.push_unary()

    if not dd:
        # index settings for efficient lookup of consistent assigments to a
        # variable. DD doesn't need this index.
        for phi in fg.potentials:
            phi.index_neighbor_settings()

    for t in xrange(maxiter):

        nu = a * 1.0 / (t + A)**gamma

        for phi in fg.potentials:

            if dd:
                phi.argmax(phi.lmda)

            else:
                phi.update(phi.lmda)

        # TODO: implement convergence checks; e.g. DD primal-dual gap.

        # v->f message removes influence of f on belief

        # mu is the variables "belief", in BP terms.
        mu = zeros_like(fg.theta, dtype=float)
        for phi in fg.potentials:
            mu[phi.args] += phi.v

        if dd:
            # DD averages message, but seems to work fine with out it. Is this
            # step required? I don't think we can simply absorb this constant
            # into the learning rate.

            # Enabling averaging breaks BP (empirically). I'm not too surprised
            # because messages are now on different scales [revisit].

            mu /= fg.d

        # BP's v->f messages are DD's subgradient step.

        # subgradient step on lagrange multipliers
        for phi in fg.potentials:

            # Interesting: "mu[..]-phi.v" corresponds to subtracting the
            # influence of this potential from the variable's belief, just we do
            # in like BP's var->factor messages.

            phi.lmda += nu * (mu[phi.args,:] - phi.v)


    result = list(mu.argmax(axis=1).astype(int))

    if verbose:
        from arsenal.terminal import yellow, green
        print yellow % '*****************************************'
        print yellow % 'mu ='
        print mu #/ mu.sum(axis=0)
        print
        for phi in fg.potentials:
            print green % ('Potential%s lambda' % phi.args)
            print phi.lmda
            print
        print yellow % 'result ='
        print result
        print yellow % '*****************************************'

    # decode solution from "agreement vector," mu.
    return result


class FactorGraph(object):

    def __init__(self, unary):
        self.theta = array(unary)
        # degree of each variable (number of potentials it participates
        # in). Note: we don't count unary potentials in degree.
        self.d = zeros_like(self.theta)
        self.potentials = []

    def __iadd__(self, phi):
        "Allows nice syntax for 'adding' a potential to the factor graph."
        self.potentials.append(phi)
        self.d[phi.args] += 1.0
        return self

    def push_unary(self):
        """
        Push unary potentials into higher-order factors.

        Higher-order potentials will receive a fraction of unary potential
        proportional to the degree of the variable. This is just a heuristic to
        avoid making some potentials much stronger than others. We could also
        just picked one potential to push into per variable.
        """
        theta = self.theta / self.d
        for phi in self.potentials:
            phi.add_unary(theta[phi.args, :])  # make indices local to potential


class Potential(object):

    def __init__(self, vals, args):

        # This potential stores local and global indices for its variables.
        # Implementation note: Potential should only make use of local indexing
        self.args = args
        self.localargs = range(len(self.args))

        self.settings = vals.items()

        # figure out variable domains...
        self.domain = dom = {i: set() for i in self.localargs}
        for x, _ in self.settings:
            for i in self.localargs:
                dom[i].add(x[i])

        maxdomainsize = max(map(len, dom.values()))

        # ddecomp stuff
        self.lmda = zeros((len(args), maxdomainsize))
        self.v = zeros_like(self.lmda)      # DD "message" from factor to variable

        # bp stuff
        self.ns = None    # neighbor settings
        self.N_ne = None  # indices of neighboring variables != i

    def index_neighbor_settings(self):
        """Index clique assignments on variable-value pairs. Used BP."""
        self.ns = ns = defaultdict(list)
        localargs = self.localargs
        for x, score in self.settings:
            for i in localargs:
                ns[i, x[i]].append((array(x), score))
        # indices of neighboring variables != i
        self.N_ne = [array([j for j in localargs if j != i]) for i in localargs]

    def add_unary(self, theta):
        "Bake-in unary potentials by modifying potential table appropriately."
        assert theta.shape == self.v.shape, [theta.shape, self.v.shape]
        localargs = self.localargs
        self.settings = [(x, score + theta[localargs, list(x)].sum()) for x, score in self.settings]

    def argmax(self, omega):
        localargs = self.localargs
        [_, argmax] = max((score + omega[localargs, list(x)].sum(), x) for x, score in self.settings)
        # massage solutions into var-val matrix (Can be generalized to any
        # features of the argmax of this potential not just this binary
        # variable-value thing we've implemented below).
        self.v.fill(0)
        self.v[localargs, argmax] = 1

    def update(self, omega):
        """
        Compute message from potential to each of its arguments.
        """

        # TODO: Overload this method and argmax to implement structured BP/DD,
        # e.g. custom potential which use dynamic programming to compute
        # messages.

        self.v.fill(None)
        localargs = self.localargs
        for i in localargs:
            N_ne_i = self.N_ne[i]  # indices of neighboring variables != i
            for v in self.domain[i]:
                # Solve subproblem with variable i constrained to value, v,
                # while incorporating bias from variable messages. The variable
                # messages are exactly the "hacked unary potentials" (the
                # lagrange multiplers) in DD. The only difference how we use the
                # message: BP leaves out the message from variable `i` for this
                # subproblem's objective function.
                self.v[i,v] = max(score + omega[N_ne_i, x[N_ne_i]].sum() for x, score in self.ns[i, v])

                # Contrast BP message with DD: this use of cavity marginal, we
                # send a message back to each variable which ignores it's
                # influence in this factor. DD just sends features of the max
                # assignment.

                # I believe there's a simple generalization of the
                # variable-value messages we have which sends feature
                # expectation (i.e. first-order statistcs), There is an
                # analogous generalization of DD's messages. Although, because
                # we're doing max-sum it's not actually an expectation - it's
                # features of the max? [todo: check]. Of course, we're still
                # using the cavity marginal so it's different from DD. This
                # generalization is commonly discussed in expectation
                # propagation, because messages are the sufficient statistics of
                # parametric distribution (e.g., exponential family), which
                # approximates the continuous analogue the discrete messages we
                # have here.

        # Another difference between DD and BP, is that BP fills the matrix with
        # the max value consistent with the variable assignment, whereas DD's
        # message is a binary matrix indicating whether a variable assignment is
        # optimal for the subproblem.

        # It's tempting to think that DD messages are simply annealed BP message
        # (i.e.,"DD is just BP run at a very low temperature"). This is *not*
        # the case because subproblem objectives are different. BP sends
        # messages which are max-*marginals* (a max for each variable-value). DD
        # only sends messages pertaining to one assignment to the clique.
