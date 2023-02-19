import numpy as np

# QUESTION 1
def nevillesMethod(x, fx, interpolatingValue):
    matrix = np.zeros((3,3))

    for index, row in enumerate(matrix):
        row[0] = fx[index]

    nPoints = len(x)

    for i in range(1,nPoints):
            for j in range(1, i + 1):
                mul1 = (interpolatingValue - x[i-j]) * matrix[i][j-1]
                mul2 = (interpolatingValue - x[i]) * matrix[i-1][j-1]
                den = x[i] - x[i-j]
                coefficient = (mul1 - mul2)/den
                matrix[i][j] = coefficient

    print(matrix[nPoints - 1][nPoints - 1])


# QUESTION 2
def newtonsForwardMethod(x, fx):
    length = len(x)
    matrix = np.zeros((length, length))

    for i in range(length):
        matrix[i][0] = fx[i]

    for i in range(1, length):
        for j in range(1, i+1):
            num = matrix[i][j-1] - matrix[i-1][j-1]
            den = x[i] - x[i-j]
            matrix[i][j] = num / den

    return matrix


# QUESTION 3
def approximation(x, matrix, value):
    xSpan = 1
    approx = matrix[0][0]
    for i in range(1, 4):
        xSpan *= value - x[i-1]
        approx += matrix[i][i] * xSpan

    print(approx)

# QUESTION 4
np.set_printoptions(precision=7, suppress=True, linewidth=100)
def apply_div_dif(matrix: np.array):
    size = len(matrix)
    for i in range(2, size):
        for j in range(2, i+2):
            if j >= len(matrix[i]) or matrix[i][j] != 0:
                continue      
            left: float = matrix[i][j - 1]
            diagonal_left: float = matrix[i - 1][j - 1]
            numerator: float = left - diagonal_left
            denominator = matrix[i][0] - matrix[i - j + 1][0]
            operation = numerator / denominator
            matrix[i][j] = operation
    return matrix

def hermite_interpolation(x_points, y_points, slopes):
    num_of_points = len(x_points)
    matrix = np.zeros((2 * num_of_points, 2 * num_of_points))

    for i, x in enumerate(x_points):
        matrix[2 * i][0] = x
        matrix[2 * i + 1][0] = x
    
    for i, y in enumerate(y_points):
        matrix[2 * i][1] = y
        matrix[2 * i + 1][1] = y

    for i, slope in enumerate(slopes):
        matrix[2 * i + 1][2] = slope

    filled_matrix = apply_div_dif(matrix)
    print(filled_matrix)

# QUESTION 5
def cubicSplineInterpolation(x, fx):
    func = lambda val: x[val+1] - x[val]
    length = len(x)
    matrixA = np.zeros((length, length))
    matrixA[0][0] = 1
    matrixA[length-1][length-1] = 1
    for i in range(1, length-1):
        matrixA[i][i-1] = func(i-1)
        matrixA[i][i] = 2*(func(i-1)+func(i))
        matrixA[i][i+1] = func(i)
    print(matrixA)
    print("")

    vectorB = np.zeros((length))
    for i in range(1, length-1):
        vectorB[i] = (3/func(i))*(fx[i+1]-fx[i])-(3/func(i-1))*(fx[i]-fx[i-1])
    print(vectorB)
    print("")

    invMatrix = np.linalg.inv(matrixA)
    vectorX = invMatrix.dot(vectorB)
    print(vectorX)


if __name__ == "__main__": 
    # QUESTION 1
    x = [3.6, 3.8, 3.9]
    fx = [1.675, 1.436, 1.318]
    interpolatingValue = 3.7
    nevillesMethod(x, fx, interpolatingValue)
    print("")

    # QUESTION 2
    x_points = [7.2, 7.4, 7.5, 7.6]
    y_points = [23.5492, 25.3913, 26.8224, 27.4589]
    matrix = newtonsForwardMethod(x_points, y_points)
    result = []
    for i in range(1, 4):
        result.append(matrix[i][i])
    print(result)
    print("")

    # QUESTION 3
    approximation(x_points, matrix, 7.3)
    print("")

    # QUESTION 4
    x_points2 = [3.6, 3.8, 3.9]
    y_points2 = [1.675, 1.436, 1.318]
    slopes = [-1.195, -1.188, -1.182]
    hermite_interpolation(x_points2, y_points2, slopes)
    print("")

    # QUESTION 5
    x_points3 = [2, 5, 8, 10]
    y_points3 = [3, 5, 7, 9]
    cubicSplineInterpolation(x_points3, y_points3)

    
