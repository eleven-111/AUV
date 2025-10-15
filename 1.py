# Create the polynormial basis
basis = np.column_stack([X ** 2, X, np.ones_like(X)])

# Solve the linear regression through Least Square
# Try with ddifferent regularization and verify the performance (e.g., prediction loss)
w = np.linalg.inv(basis.T @ basis) @ basis.T @ Y

# Plot the curve according to the quadratic function: y = w_1 * x * x + w_2 * x + w_3
NX = []
NY = []
for x in np.arange(0, 1, 0.01):
	y_pred = w[0] * x ** 2 + w[1] * x + w[2]
	NX.append(x)
	NY.append(y_pred)

# Display the curve, DO NOT modify
plt.plot(NX, NY)
plt.show()



lambda_ = 0.1  # 设定正则化参数
I = np.eye(basis.shape[1])
w_ridge = np.linalg.inv(basis.T @ basis + lambda_ * I) @ basis.T @ Y

from sklearn.metrics import mean_squared_error
y_pred_ols = basis @ w
y_pred_ridge = basis @ w_ridge
mse_ols = mean_squared_error(Y, y_pred_ols)
mse_ridge = mean_squared_error(Y, y_pred_ridge)
print(f"OLS模型的均方误差: {mse_ols}")
print(f"Ridge正则化模型的均方误差: {mse_ridge}")



new_X = np.random.rand(10)
new_Y = 2 * new_X ** 2 + 3 * new_X + 1 + np.random.randn(10) * 0.5  # 模拟带有噪声的二次函数数据


new_basis = np.column_stack([new_X ** 2, new_X, np.ones_like(new_X)])
new_y_pred_ols = new_basis @ w
new_y_pred_ridge = new_basis @ w_ridge
new_mse_ols = mean_squared_error(new_Y, new_y_pred_ols)
new_mse_ridge = mean_squared_error(new_Y, new_y_pred_ridge)
print(f"新数据上OLS模型的均方误差: {new_mse_ols}")
print(f"新数据上Ridge正则化模型的均方误差: {new_mse_ridge}")



best_lambda = 0
min_mse = float('inf')
lambda_values = np.logspace(-3, 3, 7)  # 尝试不同数量级的lambda值
for lambda_ in lambda_values:
    I = np.eye(new_basis.shape[1])
    w_ridge = np.linalg.inv(new_basis.T @ new_basis + lambda_ * I) @ new_basis.T @ new_Y
    new_y_pred_ridge = new_basis @ w_ridge
    current_mse = mean_squared_error(new_Y, new_y_pred_ridge)
    if current_mse < min_mse:
        min_mse = current_mse
        best_lambda = lambda_
print(f"新数据上最优的正则化参数lambda: {best_lambda}")