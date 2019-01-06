import numpy as np 


def vector_multiplication_using_numpy(v1,v2):
    '''
     
     this is where we would be performing multiplication
     operation using only 1D tensors. and we would be
     using numpy operations on np tensors

     :param v1: 1D np tenspr
     :param v2: 1D np tensor

     :return: 1D vector multiplication results in a scalar value

     this algorithm would perform:
     element wise multiplication of each vector v1 and v2

    '''
    
    #check if inputs are 1D tensors
    assert len(v1.shape) == 1
    assert len(v2.shape) == 1

    #check if vectors are of same size
    assert v1.shape[0] == v2.shape[0]
    
    return np.dot(v1,v2)


def vector_multiplication_using_python(v1,v2):
    '''
     
     this is where we would be performing multiplication
     operation using only 1D tensors. and we would be
     using numpy operations on np tensors

     :param v1: 1D np tenspr
     :param v2: 1D np tensor

     :return: 1D vector multiplication results in a scalar value

     this algorithm would perform:
     element wise multiplication of each vector v1 and v2

    '''
    
    
    #check if inputs are 1D tensors
    assert len(v1.shape) == 1
    assert len(v2.shape) == 1

    #check if vectors are of same size
    assert v1.shape[0] == v2.shape[0]
    z =  0

    for i in range(v1.shape[0]):
        z += v1[i] * v2[i]
    
    return z


def vector_matrix_multiplication_using_numpy(m, v):
    ''' 
        this is where we multiply a matrix with a vector
        remember it is important that m.shape[1] == v.shape[0]
        also m is a 2D tensor

        resultant will be a vector of the shape
        (m.shape[0])

    '''
    assert len(m.shape) == 2
    assert len(v.shape) == 1
    assert m.shape[1] == v.shape[0]

    return np.dot(m,v)

 
def vector_matrix_multiplication_using_python(m, v):
    ''' 
        this is where we multiply a matrix with a vector
        remember it is important that m.shape[1] == v.shape[0]
        also m is a 2D tensor

        resultant will be a vector of the shape
        (m.shape[0])

    '''
    assert len(m.shape) == 2
    assert len(v.shape) == 1
    assert m.shape[1] == v.shape[0]

    z = np.zeros((m.shape[0])).astype(np.int32)

    for i in range(m.shape[0]):
        z[i] = vector_multiplication_using_python(m[i, :],v)
    
    return z

def matrix_multiplication_using_numpy(m1,m2):
    assert len(m1.shape) == len(m2.shape)
    assert m1.shape[1] == m2.shape[0]

    return np.dot(m1,m2)

def matrix_multiplication_using_python(m1, m2):
    assert len(m1.shape) == len(m2.shape)
    assert m1.shape[1] == m2.shape[0]

    z = np.zeros((m1.shape[0], m2.shape[1]))

    for i in range(m1.shape[0]):
        for j in range(m2.shape[1]):
            z[i,j] = vector_multiplication_using_python(m1[i],m2[:,j])
    
    return z


def execute():
    assert vector_multiplication_using_numpy(np.random.randint(1,2,(6)), \
                np.random.randint(2,3,(6))) == \
            vector_multiplication_using_python(np.random.randint(1,2,(6)), \
                np.random.randint(2,3,(6)))

    m = np.random.randint(2,6, (3,7))
    v = np.random.randint(5,17, (7))

    # print(vector_matrix_multiplication_using_numpy(m,v),\
    #         vector_matrix_multiplication_using_python(m, v))

    m1 = np.random.randint(2,6,(2,6))
    m2 = np.random.randint(6,30,(6, 5))

    print(matrix_multiplication_using_numpy(m1, m2), \
                matrix_multiplication_using_python(m1,m2))

execute()