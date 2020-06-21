import numpy as np
import matplotlib.pyplot as plt
data = np.array([[152, 51], [156, 53], [160, 54], [164, 55],
                 [168, 57], [172, 60], [176, 62], [180, 65],
                 [184, 69], [188, 72]])
print(data.shape)

# Build a linear regression model
x, y = data[:, 0].reshape(-1, 1), data[:, 1]
# train the linear regression model
regr = linear_model.LinearRegression()
regr.fit(x, y)

# Draw trained line
plt.plot(x, regr.predict(x), color='blue')
# Draw Scatter plot of height-weight
plt.scatter(x, y, color='black')
plt.xlabel('height (cm)')
plt.ylabel('weight (kg)')
plt.show()
