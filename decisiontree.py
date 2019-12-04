import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[: , 1:2].values
y =dataset.iloc[:, 2].values




from sklearn.model_selection import train_test_split
x_train ,x_test ,y_train , y_test = train_test_split(x , y ,test_size=0.2, random_state=0)





"""from sklearn.preprocessing import StandardScaler
standardscaler_obj = StandardScaler()
x_train = standardscaler_obj.fit_transform(x_train)
x_test = standardscaler_obj.fit_transform(x_test)"""





from sklearn.tree import DecisionTreeRegressor
decisiontree_obj = DecisionTreeRegressor(random_state =0)
decisiontree_obj.fit(x , y)






y_pred =   decisiontree_obj.predict(x)


#-----------------visualize data---------------


plt.scatter(x , y,color='red')
plt.plot(x , decisiontree_obj.predict(x) , color='blue')
plt.title('Decision tree')
plt.xlabel('years of experience')
plt.ylabel('salary')
plt.show()