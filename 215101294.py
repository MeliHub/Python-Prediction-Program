import numpy as np
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LinearRegression

# NAME: MELIKAYA MATIWANE
 
 # READING DATA FROM A FILE
data = np.loadtxt("13.csv", delimiter=";", dtype=np.int32)
print(data)

# CALCULATE MEAN AND STANDARD DEVIATION OF THE DATA SET
mean = data.mean(0)
std = data.std(0) 

# SEPERATING THE COLUMNS BY SPLITTING THE DATA SET HORIZONTALLY
mean_split = np.hsplit(mean, 2)
std_split = np.hsplit(std, 2)

#CONVERT THE MEAN AND STANDARD DEVIATION TO SHOW IN DECIMALS
x_mean = mean_split[0]  
y_mean = mean_split[1] 
x_std = std_split[0] 
y_std = std_split[1] 

#PRINTING MEAN OF X, MEAN OF Y, STANDARD DEVIATION OF X, AND STANDARD DEVIATION OF Y
print("Mean of X: ", x_mean)
print("Mean of Y: ", y_mean)
print("Standard Deviation of X: ", x_std) 
print("Standard Deviation of Y: ", y_std)


# CORRELATION CALCULATION AND PRINTING 
split = np.hsplit(data, 2)
x = split[0]
y = split[1]
r = pearsonr(x,y)
print("Correlation", r)

#SLOPE CALCULATION AND PRINTING
b = r[0] * (y_std) / x_std

print("Slope: ", b)

# INTERCEPT CALCULATION AND PRINTING
a = y_mean - (b * x_mean)

print("Intercept: ", a)


# Calculating SSE
x_val = x.reshape(len(x), 1)
reg = LinearRegression()
reg = reg.fit(x_val, y)
y_val = reg.predict(x_val)
sse = ((y - y_val)**2).sum()
print("SSE: ", sse)

# ASSIGN THE MINUTES THE STUDENT HAS SPENT STUDYING TO x, THEN THE REGRESSION FORMULA WILL SHOW THE PREDICTED PERCENTAGE MARK OF THAT PARTICULAR STUDENT
# i.e  318 minutes spent studying
x = 250
Y = b * (x) + a
print("Predicted Passing Mark based on the minutes spent studying: ", Y)


