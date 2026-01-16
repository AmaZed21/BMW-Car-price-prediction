import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

#Produce DataFrame from csv file
bmw_data = pd.read_csv('/Users/aryan/Documents/Coding/VS_CODE/env/csv_files/bmw.csv')
#print(bmw_data[(bmw_data['year'] == 2016) & (bmw_data['mileage'] < 40000)])

#Converting data from words to 0/1
def binary(df: pd.DataFrame) -> pd.DataFrame:
    for item in df.transmission.tolist():
        if item == 'Automatic':
            df['transmission'].replace(item, 1, inplace = True)
        else:
            df['transmission'].replace(item, 0, inplace = True)
    
    for item in df.fuelType.tolist():
        if item == 'Diesel':
            df['fuelType'].replace(item, 1, inplace = True)
        else:
           df['fuelType'].replace(item, 0, inplace = True)
    return df

#Final dataframe to fit and transform
final_bmw_data = binary(bmw_data)
final_bmw_data.drop(['model'], inplace=True, axis = 1)

#Creating training and validating variables
x = final_bmw_data.drop(['price'], axis=1)
y = final_bmw_data['price']
X_train, X_test, y_train, y_test = train_test_split(x, y,random_state=24)

#Initialise model
model = RandomForestRegressor(n_estimators=1000, random_state=1)
model.fit(X_train, y_train)
y_pred = model.predict(x)


#Create Final df to compare values
outputted_df = pd.DataFrame(
    {
        'originalPrice': y,
        'predictedPrice': y_pred
    }
)
outputted_df.index = pd.read_csv('/Users/aryan/Documents/Coding/VS_CODE/env/csv_files/bmw.csv')['model']

#Save df as new csv file
outputted_df.to_csv('prediction-original-prices.csv')