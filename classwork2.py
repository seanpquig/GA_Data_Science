"""
returns transpose of matrix passed to it
"""


def matrixTranspose(matrix):
     return [ [row[i] for row in matrix] for i in range(len(matrix[0]))]


"""
returns result of vector and matrix multiplication
"""
def vectorMatrixMult(vect, matx):
  if len(vect) != len(matx):
    print 'Vector width does not equal matrix height'
  else:
    # Transpose matrix, so row lengths match vector for easy multiplication
    matx = matrixTranspose(matx)
    output_matx = []
    for j in range(len(matx)):
      row_mult = 0
      for i in range(len(vect)):
        row_mult += vect[i] * matx[j][i]
      output_matx.append(row_mult)
    return output_matx


"""
returns result of matrix-matrix multiplication
simplifies by treating each row of matrix 1 as 
a vector and aggregates results of vector matrix
operations
"""
def matrixMult(matx1, matx2):
  if len(matx1[0]) != len(matx2):
    print 'Width of 1st matrix doesn\'t match height of 2nd'
  else:
    output_matx = []
    for i in range(len(matx1)):
      output_matx.append(vectorMatrixMult(matx1[i], matx2))
    return output_matx

"""
Creates an idntity matrix of dimensions n x n
"""
def makeIdenMatx(n):
  matrix = []
  for i in xrange(n):
    row = []
    for j in xrange(n):
      if i==j:
        row.append(1)
      else:
        row.append(0)
    matrix.append(row)
    
  return matrix


# test cases
vector = [1, 2, 3]
matrix1 = [[1, 2], [3, 4]]
matrix2 = [[1, 3], [2, 4]]
# print vectorMatrixMult(vector, matrix2), '\n'

matx = matrixMult(matrix1, matrix2)
for row in matx:
  print row
print '\n'

idenMatx = makeIdenMatx(6)
for row in idenMatx:
  print row
