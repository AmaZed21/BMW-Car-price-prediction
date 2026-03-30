import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

#Produce DataFrame from csv file
bmw_data = pd.read_csv('data/dataset.csv')

#Converting data from words to 0/1
def binarise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['transmission'] = (df['transmission'] == 'Automatic').astype(int)
    df['fuelType'] = (df['fuelType'] == 'Diesel').astype(int)
    return df

#Final dataframe to fit and transform
final_bmw_data = binarise(bmw_data)
model_column = final_bmw_data['model']
final_bmw_data.drop(['model'], inplace=True, axis = 1)

#Creating training and validating variables
X = final_bmw_data.drop(['price'], axis=1)
y = final_bmw_data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3,random_state=24)

#Initialise model
model = RandomForestRegressor(n_estimators=1000, random_state=1, n_jobs = -1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#Evaluation
mae  = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae: .2f}')

#Create Final df to compare values
outputted_df = pd.DataFrame(
    {
        'originalPrice': y_test,
        'predictedPrice': y_pred.round(0)
    }
)
outputted_df.index = model_column.iloc[X_test.index].values
outputted_df.index.name = 'carModel'

#Save df as new csv file
outputted_df.to_csv('data/model_predictions.csv')
