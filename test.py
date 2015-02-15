from inference import FactorGraph, Potential, solve
from numpy import zeros


# TODO: test examples against the output libDAI; BP should produce exactly the
# same marginals.

def chain_example(DD, n=5, biasfirst=0.1, biasall=0.0):

    # unary
    unary = [[0, biasfirst + biasall]]
    for i in xrange(1, n):
        unary.append([0, biasall])

    g = FactorGraph(unary)

    # xor potential
    Xor = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 1,
        (1, 1): 0,
    }

    tbl = Xor

    # pairwise chain potentials
    for i in xrange(1, n):
        g += Potential(tbl, [i-1, i])

    return solve(DD, g, maxiter=100)


def test_chain_example(DD):

    print 'Chain:'

    result = chain_example(DD, n=5, biasfirst=1)
    assert result == [1, 0, 1, 0, 1], result

    result = chain_example(DD, n=3, biasfirst=1)
    assert result == [1, 0, 1], result

    result = chain_example(DD, n=8, biasfirst=1)
    assert result == [1, 0]*4, result

    result = chain_example(DD, n=5, biasfirst=0.0, biasall=0.1)
    assert result == [1, 0, 1, 0, 1], result


def triangle_example(DD, a=0, b=0, c=0):
    """
    args (a,b,c) pick which variable will break the symmetry (i.e. which will be
    zero).
    """

    # unary
    unary = [
        (a, 0, 0),   # break ties.
        (b, 0, 0),
        (c, 0, 0),
    ]

    g = FactorGraph(unary)

    # 'equals' and 'different' potential
    Diff = {}
    for a in [0, 1, 2]:
        for b in [0, 1, 2]:
            Diff[a, b] = (float(a != b) - 1) * 10000
    tbl = Diff

    # triangle
    g += Potential(tbl, [0, 1])
    g += Potential(tbl, [1, 2])
    g += Potential(tbl, [2, 0])

    return solve(DD, g, maxiter=50, A=1, a=.1)


def test_triangle_example(DD):

    print 'Triangle:'

    result = triangle_example(DD, a=1)
    assert result == [0, 2, 1] or result == [0, 1, 2], result

    result = triangle_example(DD, b=1)
    assert result == [2, 0, 1] or result == [1, 0, 2], result

    result = triangle_example(DD, c=1)
    assert result == [2, 1, 0] or result == [1, 2, 0], result



def sparse_sparse_simple(DD, a=0, b=0, c=0, d=0):
    """
    args (a,b,c) pick which variable will break the symmetry (i.e. which will be
    zero).
    """

    # unary
    unary = [
        (0, a),   # break ties.
        (0, b),
        (0, c),
        (0, d),
    ]

    g = FactorGraph(unary)

    k = 4

    # 'equals' and 'different' potential
    ExactlyOne = {}
    for i in xrange(k):
        x = zeros(k, dtype=int)
        x[i] = 1
        ExactlyOne[tuple(x)] = 10.0
    tbl = ExactlyOne

    g += Potential(tbl, [0, 1, 2, 3])

    return solve(DD, g, maxiter=5, A=1, a=1)


def test_sparse_simple(DD):
    got = sparse_sparse_simple(DD, a=1)
    assert [1, 0, 0, 0] == got, got

    got = sparse_sparse_simple(DD, b=1)
    assert [0, 1, 0, 0] == got

    got = sparse_sparse_simple(DD, c=1)
    assert [0, 0, 1, 0] == got

    got = sparse_sparse_simple(DD, d=1)
    assert [0, 0, 0, 1] == got



def sparse_tree(DD, a=0, b=0, c=0, d=0, e=0):

    # unary
    unary = [
        (0, a),   # break ties.
        (0, b),
        (0, c),
        (0, d),
        (0, e),
        (0, 0),
        (0, 0),
    ]

    g = FactorGraph(unary)

    k = 3

    # 'equals' and 'different' potential
    ExactlyOne = {}
    for i in xrange(k):
        x = zeros(k, dtype=int)
        x[i] = 1
        ExactlyOne[tuple(x)] = 1.0
    tbl = ExactlyOne

    g += Potential(tbl, [0, 1, 2])
    g += Potential(tbl, [2, 3, 4])
    g += Potential(tbl, [5, 2, 6])

    return solve(DD, g, maxiter=10, A=1, a=0.1)


def test_sparse_tree(DD):
    got = sparse_tree(DD, c=1)
    assert [0, 0, 1, 0, 0, 0, 0] == got, ('DD' if DD else 'BP', got)


def main():
    if 1:
        print '=== dual decomposition tests ==='
        test_chain_example(DD=True)
        test_triangle_example(DD=True)
        test_sparse_simple(DD=True)
        test_sparse_tree(DD=True)

    if 1:
        print '=== belief propagation tests ==='
        test_chain_example(DD=False)
        test_sparse_simple(DD=False)
        test_sparse_tree(DD=False)

        try:
            test_triangle_example(DD=False)
        except AssertionError:
            print 'triangle BP failed..'


if __name__ == '__main__':
    main()
