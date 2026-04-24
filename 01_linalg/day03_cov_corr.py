import numpy as np

np.set_printoptions(precision=6, suppress=True)

# -------------------------------------------------
# 1. Returns matrix
# Rows = dates
# Columns = assets
# -------------------------------------------------
R = np.array([
    [ 0.01,  0.02, -0.01],
    [ 0.03,  0.01,  0.00],
    [-0.02,  0.00,  0.02],
    [ 0.04,  0.03,  0.01],
    [ 0.00, -0.01,  0.03]
])

T, N = R.shape
print("R =")
print(R)
print("\nShape of R:", R.shape)
print("T (number of dates) =", T)
print("N (number of assets) =", N)

# -------------------------------------------------
# 2. Mean return of each asset
# axis=0 means: take the mean down each column
# -------------------------------------------------
mean_returns = R.mean(axis=0)
print("\nMean returns:")
print(mean_returns)

# -------------------------------------------------
# 3. Demean the returns
# Subtract each asset's mean from each observation
# -------------------------------------------------
X = R - mean_returns
print("\nDemeaned returns X = R - mean_returns:")
print(X)

# -------------------------------------------------
# 4. Sample covariance matrix by hand
# Formula: (X.T @ X) / (T - 1)
# -------------------------------------------------
cov_manual = (X.T @ X) / (T - 1)
print("\nManual covariance matrix:")
print(cov_manual)

# -------------------------------------------------
# 5. Compare to NumPy
# rowvar=False means columns are variables/assets
# ddof=1 gives sample covariance
# -------------------------------------------------
cov_np = np.cov(R, rowvar=False, ddof=1)
print("\nNumPy covariance matrix:")
print(cov_np)

print("\nDo manual and NumPy covariance match?")
print(np.allclose(cov_manual, cov_np))

# -------------------------------------------------
# 6. Standard deviations from covariance matrix
# diagonal entries are variances
# -------------------------------------------------
variances = np.diag(cov_manual)
std_devs = np.sqrt(variances)

print("\nVariances (diagonal of covariance matrix):")
print(variances)

print("\nStandard deviations:")
print(std_devs)

# -------------------------------------------------
# 7. Correlation matrix by hand
# corr_ij = cov_ij / (std_i * std_j)
# np.outer(std_devs, std_devs) builds all pairwise products
# -------------------------------------------------
corr_manual = cov_manual / np.outer(std_devs, std_devs)
print("\nManual correlation matrix:")
print(corr_manual)

# -------------------------------------------------
# 8. Compare to NumPy
# -------------------------------------------------
corr_np = np.corrcoef(R, rowvar=False)
print("\nNumPy correlation matrix:")
print(corr_np)

print("\nDo manual and NumPy correlation match?")
print(np.allclose(corr_manual, corr_np))

print("\nCorrelation between asset 1 and asset 2:")
print(corr_manual[0, 1])