import numpy as np
from sklearn.model_selection import KFold,train_test_split,cross_val_score
from sklearn.linear_model import Lasso
from sklearn.metrics import root_mean_squared_error as rmse
import matplotlib.pyplot as plt
from data_generator import postfix
from q3 import liftDataset, liftLabels


# Number of samples
N = 1000

# Noise variance 
sigma = 0.01


# Feature dimension
d = 40

psfx = postfix(N,d,sigma) 
      

X = np.load("X"+psfx+".npy")
y = np.load("y"+psfx+".npy")

print("Dataset has n=%d samples, each with d=%d features," % X.shape,"as well as %d labels." % y.shape[0])

X = liftDataset(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)

print("Randomly split dataset to %d training and %d test samples" % (X_train.shape[0],X_test.shape[0]))

alphas = np.logspace(-10, 10, num=40, base=2)

rmse_vals = []
rmse_stdevs = []

for alpha in alphas:
    model = Lasso(alpha = alpha)

    cv = KFold(
            n_splits=5, 
            random_state=42,
            shuffle=True
            )



    scores = cross_val_score(
            model, X_train, y_train, cv=cv,scoring="neg_root_mean_squared_error")


    print("Cross-validation RMSE for α=%f : %f ± %f" % (alpha,-np.mean(scores),np.std(scores)) )
    rmse_vals.append(-np.mean(scores))
    rmse_stdevs.append(np.std(scores))

optimal_index = np.argmin(rmse_vals)
optimal_alpha = alphas[optimal_index]
print(f"Optimal alpha is {optimal_alpha}")

model = Lasso(alpha = optimal_alpha)

print("Fitting linear model over entire training set...",end="")
model.fit(X_train, y_train)
print(" done")

labels = liftLabels(d)
print("Model parameters with abs > 10^-3:")
print("\t Intercept: %3.5f" % model.intercept_,end="")
for i,val in enumerate(model.coef_):
    if abs(val) > 10 ** (-3):
        print(f", β%d: %3.5f ({labels[i]})" % (i,val), end="")
        
        
print("\n")    


# Compute RMSE
rmse_train = rmse(y_train,model.predict(X_train))
rmse_test = rmse(y_test,model.predict(X_test))

print("Train RMSE = %f, Test RMSE = %f" % (rmse_train,rmse_test))

rmse_vals = np.array(rmse_vals)
rmse_stdevs = np.array(rmse_stdevs)

plt.plot(alphas, rmse_vals, color="darkorange", lw=2, label="CV Fold RMSE Mean")
plt.fill_between(alphas, rmse_vals+rmse_stdevs, rmse_vals-rmse_stdevs, facecolor='C0', alpha=0.4)
plt.scatter(alphas, rmse_vals, color="darkorange")
plt.xlabel("Lasso Parameter Alpha")
plt.ylabel("RMSE Value")
plt.title(f"Mean RMSE vs Alpha")
plt.legend(loc="lower right")
plt.xscale("log")
plt.show()







