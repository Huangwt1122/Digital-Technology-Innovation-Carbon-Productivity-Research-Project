
gain_dict = model.get_booster().get_score(importance_type='gain')
Gain = np.array([gain_dict.get(f'f{j}', 0) for j in range(m)])

sum_gain = np.sum(Gain)
w = Gain / sum_gain if sum_gain > 0 else np.ones(m) / m

X_norm = np.zeros_like(X)
for j in range(m):
    col = X[:, j]
    xmin, xmax = col.min(), col.max()
    denom = (xmax - xmin)
    if denom > 0:
        X_norm[:, j] = (col - xmin) / denom
    else:
        X_norm[:, j] = 0.0

eps = 1e-12
X_norm = np.clip(X_norm, eps, 1.0)

n = X.shape[0]
k = 1.0 / np.log(n)
e = np.zeros(m)
for j in range(m):
    col = X_norm[:, j]
    p = col / np.sum(col) 
    e[j] = -k * np.sum(p * np.log(p))

DTI = np.zeros(n)
for i in range(n):
    for j in range(m):
        DTI[i] += w[j] * (1.0 - e[j]) * X_norm[i, j]