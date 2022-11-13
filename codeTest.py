
import torch
def f(x,*y):
    return 4*x+10
x = torch.randint(5,9,(2,3))
print('原来的x是:\n{}'.format(x))
# 原来的x是:
# tensor([[2, 1, 0],
#         [2, 1, 1]])
x.map_(x,f)
print(x)
# tensor([[18, 14, 10],
#         [18, 14, 14]])
