import pandas as pd

df_train=pd.read_csv(r"E:\AI\Machine learning\Machine learning project\House Price Predication\test.csv")
df_test=pd.read_csv(r"E:\AI\Machine learning\Machine learning project\House Price Predication\train.csv")
df=pd.concat([df_train,df_test])

# collection string and numeric
str_col=df.select_dtypes(include=['object']).columns
num_col=df.select_dtypes(include=['int64', 'float64']).columns

# filling numeric with median value
for col in num_col:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())
    
for col in str_col:
    df[col] = df[col].fillna("NA")

#Encoding categorical features
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in str_col:
    df[col] = le.fit_transform(df[col])

#feature and target
length=df_train.shape[0]
X=df[:length].drop(columns=['SalePrice'])
y=df[:length]['SalePrice']

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.8, random_state=42)

from sklearn.ensemble import RandomForestRegressor
lr=RandomForestRegressor(n_estimators=200, random_state=42)
lr.fit(X_train, y_train)
print("Training score:", lr.score(X_train, y_train))

# Predict value
y_pred = lr.predict(X_val)


print("Predicted:", y_pred[:10])   
print("Actual:", y_val[:10].values)  
