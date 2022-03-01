import tinytorch.fast_ops
import tinytorch
from numba import njit

# MAP
print("MAP")
tmap = tinytorch.fast_ops.tensor_map(njit()(tinytorch.operators.id))
out, a = tinytorch.zeros((10,)), tinytorch.zeros((10,))
tmap(*out.tuple(), *a.tuple())
print(tmap.parallel_diagnostics(level=3))

# ZIP
print("ZIP")
out, a, b = tinytorch.zeros((10,)), tinytorch.zeros((10,)), tinytorch.zeros((10,))
tzip = tinytorch.fast_ops.tensor_zip(njit()(tinytorch.operators.eq))

tzip(*out.tuple(), *a.tuple(), *b.tuple())
print(tzip.parallel_diagnostics(level=3))

# REDUCE
print("REDUCE")
out, a = tinytorch.zeros((1,)), tinytorch.zeros((10,))
treduce = tinytorch.fast_ops.tensor_reduce(njit()(tinytorch.operators.add))

treduce(*out.tuple(), *a.tuple(), 0)
print(treduce.parallel_diagnostics(level=3))


# MM
print("MATRIX MULTIPLY")
out, a, b = (
    tinytorch.zeros((10, 10)),
    tinytorch.zeros((10, 20)),
    tinytorch.zeros((20, 10)),
)
tmm = tinytorch.fast_ops.tensor_matrix_multiply

tmm(*out.tuple(), *a.tuple(), *b.tuple())
print(tmm.parallel_diagnostics(level=3))
