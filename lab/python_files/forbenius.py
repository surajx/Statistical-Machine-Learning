import numpy
def frobenius_norm(A):
    """ Accepts Matrix A and returns the Forbenius Norm of A

        (matrix) -> (number)
    """
    return numpy.sqrt(numpy.trace(numpy.transpose(A) * A))

def frobenius_norm_efficient(A):
    """
    """
    print(A.size())

print (frobenius_norm(numpy.mat([[1,2,3],[4,5,6],[7,8,9],[9,5,3]])))
print (frobenius_norm_efficient(numpy.mat([[1,2,3],[4,5,6],[7,8,9],[9,5,3]])))


def cost_function_for_matrix(X, mu):
    C = construct_random_matrix()
    N = construct_diagonal_matrix()
    cost = 0.5 * numpy.trace(numpy.transpose(X) * C * X * N) + \
        mu * 0.25 * (frobenius_norm_efficient(N - numpy.transpose(X) * X) ** 2)
    return cost
