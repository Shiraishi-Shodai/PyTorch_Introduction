import torch

t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)

def model(t_u, w, b):
    return t_u * w + b

def loss_fn(t_p, t_c):
    '''
    平均二乗誤差
    '''
    squared_diffs = (t_p - t_c) **2
    return squared_diffs.mean()

# 予測
w = torch.ones(())
b = torch.zeros(())

t_p = model(t_u, w, b)
# print(t_p)

# 損失の計算
loss = loss_fn(t_p, t_c)
# print(loss)

# delta = 0.1
# plus_change = loss_f(model(t_u, w + delta, b), t_c)
# minus_change = loss_f(model(t_u, w - delta, b), t_c)

# loss_rate_of_change_w = (plus_change - minus_change) / (2.0 * delta)

# # 学習率0.01
# learning_rate = 1e-2
# w = w - learning_rate * loss_rate_of_change_w

# print(plus_change)
# print(minus_change)
# print(loss_rate_of_change_w)
# print(w) 

# delta = 0.1
# plus_change = loss_f(model(t_u, w, b + delta), t_c)
# minus_change = loss_f(model(t_u, w, b - delta), t_c)

# loss_rate_of_change_b = (plus_change - minus_change) / (2.0 * delta)

# #学習率0.01
# learning_rate = 1e-2
# b = b - learning_rate * loss_rate_of_change_b

# print(plus_change)
# print(minus_change)
# print(loss_rate_of_change_b)
# print(b)

def dloss_fn(t_p, t_c):
    dsq_diffs = 2 * (t_p - t_c) / t_p.size(0) # 除算は平均のため

    return dsq_diffs.mean()

def dmodel_dw(t_u, w, b):
    return t_u

def dmodel_db(t_u, w, b):
    return 1.0

def grad_fn(t_u, t_c, t_p, w, b):
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw = dloss_dtp * dmodel_dw(t_u, w, b)
    dloss_db = dloss_dtp * dmodel_db(t_u, w, b)
    return torch.stack([dloss_dw.sum(), dloss_db.sum()])

def training_loop(n_epochs, learning_rate, params, t_u, t_c,
                  print_params=True):
    for epoch in range(1, n_epochs + 1):
        w, b = params

        t_p = model(t_u, w, b)  # <1>
        loss = loss_fn(t_p, t_c)
        grad = grad_fn(t_u, t_c, t_p, w, b)  # <2>

        if epoch in {1, 2, 3, 10, 11, 99, 100, 4000, 5000}:  # <3>
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
            if print_params:
                print('    Params:', params)
                print('    Grad:  ', grad)
        if epoch in {4, 12, 101}:
            print('...')

        if not torch.isfinite(loss).all():
            break  # <3>
            
    return params

training_loop(
    n_epochs = 100, 
    learning_rate = 1e-2, 
    params = torch.tensor([1.0, 0.0]), 
    t_u = t_u, 
    t_c = t_c)