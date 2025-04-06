import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('🌐 Machine Learning App')

st.info('This app buildes a machine learning model!')

df0 = pd.read_csv('https://raw.githubusercontent.com/CSBaruh/Datas/refs/heads/main/synthetic_online_retail_data.csv')

#df = df0.dropna(subset=['gender'])
#df = df0[df0['gender'].notna() & (df0['gender'] != 'Nan')]
df = df0.dropna(subset=['gender'])
df = df.drop(['product_id', 'category_id', 'customer_id', 'review_score'], axis=1)
df['order_date'] = pd.to_datetime(df['order_date'])
df['order_date'] = df['order_date'].dt.strftime('%Y-%m')


with st.expander('Data'):
  st.write('**Raw data**')
  df

  
  
  st.write('**X**')
  X_raw = df.drop('category_name', axis=1)
  #X = df.order_date
  X_raw

  st.write('**y**')
  y_raw = df.category_name
  y_raw
  
# order_date,product_id,category_id,category_name,product_name,quantity,price,payment_method,city,review_score,gender,age
with st.expander('Data Visualization'):
  #st.scatter_chart(data=df, x='quantity', y= 'product_name', color= 'category_name')
  st.scatter_chart(df, x="quantity", y="category_name", color="product_name")

# Input features
with st.sidebar:
  st.header('Input features')
  category_name = st.selectbox('Category Name', ('Electronics', 'Sports & Outdoors', 'Books & Stationery', 'Fashion', 'Home & Living'))
  product_name = st.selectbox('Product Name', ('Smartphone', 'Soccer Ball', 'Tent', 'Story Book', 'Skirt', 'Tablet', 'Yoga Mat', 'Pillow', 'Blanket'
                                               , 'Smartwatch', 'Notebook', 'Laptop', 'Pen', 'Shirt', 'Carpet', 'Novel', 'Eraser', 'Vase', 'Dress'
                                               , 'Pants', 'T-shirt', 'Running Shoes'))
  price = st.slider('Price', 18, 480, 230)
  payment_method = st.selectbox('Payment Method', ('Credit Card', 'Bank Transfer', 'Cash on Delivery'))
  # city = st.selectbox('City', ('New Oliviaberg', 'Port Matthew', 'West Sarah', 'Hernandezburgh', 'Jenkinshaven', 'East Tonyaberg', 'North Jessicabury'
  #                            , 'Aliciaberg', 'West Larrymouth', 'Lake Ian', 'Elizabethmouth', 'Melanieberg', 'Allisonland', 'Myershaven', 'South Tonya'
  #                           , 'Port Allisonfort', 'Levyport', 'Fullerland', 'North Anthony', 'North Whitneytown', 'West Cynthiaton', 'East Christopher'
  #                            , 'Port Danielleview', 'East Christopherborough', 'Douglasport', 'North Jamesside', 'North Carrie', 'Port Kenneth'
  #                            , 'East Kylie', 'Mendezburgh', 'Kristenland', 'Lake Sarah', 'Fryeberg', 'Lake Michael', 'North Mary', 'Cynthiaport'
  #                            , 'East Corytown', 'Martinezview', 'North Terrancehaven', 'Lake Rhondatown', 'Williamston', 'Lake Jeffrey', 'New Carolfort'
  #                            , 'Franklinmouth', 'New Terri', 'Jeffreyview', 'Teresaville', 'Mcdonaldmouth', 'West Geraldhaven', 'West Geraldhaven', 'Walkerland'
  #                            , 'Port Patriciashire', 'Jacobburgh', 'New Brittanytown', 'Woodshaven', 'Rachelland', 'Gonzalezshire', 'Blakeshire', 'Whitakerview'
  #                            , 'Port Thomaston', 'New Jeremy', 'Lake Nancy', 'Lake Jasmineport', 'Garrisonberg', 'South Katherineside', 'Mccallhaven', 'Lake Teresafurt'
  #                            , 'Roymouth', 'East Charles', 'West Krista', 'Hobbston', 'Jacobfurt', 'New Felicia', 'Spencerside', 'West Jacob'))
  gender = st.selectbox('Gender', ('M', 'F'))
  age = st.slider('Age', 18, 75, 28)



# Create a Dataframe for the input features
  data = {'category_name': category_name,
        'product_name': product_name,
        'price': price,
        'payment_method': payment_method,
        # 'city': city,
        'gender': gender,
        'age': age}
  input_df = pd.DataFrame(data, index=[0])
  input_sales = pd.concat([input_df, X_raw], axis=0)

with st.expander('Input features'):
  st.write('**Input**')
  input_df
  st.write('**Combined**')
  input_sales


# Data preparation
# Encode X
encode = ['category_name', 'product_name', 'price', 'payment_method', 'gender', 'age']
df_sales = pd.get_dummies(input_sales, prefix=encode)
input_row = df_sales[:1]

# Encode y
target_mapper = {'Electronics' : 0
                 , 'Sports & Outdoors': 1
                 , 'Books & Stationery': 2
                 , 'Fashion':3
                 , 'Home & Living': 4
                }

def target_encode(val):
  return target_mapper[val]

y = y_raw.apply(target_encode)

with st.expander('Data preparation'):
  st.write('**Encoded X (input sales)**')
  input_row
  st.write('**Encoded y**')
  y

#  Model trainig and inference
## Train ML model
clf = RandomForestClassifier()
clf.fit(df_sales,y)

## Apply model to make predictions
predicition = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)
prediction_proba









