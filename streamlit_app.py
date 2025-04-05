import streamlit as st
import pandas as pd

st.title('🌐 Machine Learning App')

st.info('This app buildes a machine learning model!')

with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/CSBaruh/Datas/refs/heads/main/synthetic_online_retail_data.csv')
  df
  
  st.write('**X**')
  X = df.drop('category_name', axis=1)
  X

  st.write('**y**')
  y = df.category_name
  y
  
# order_date,product_id,category_id,category_name,product_name,quantity,price,payment_method,city,review_score,gender,age
with st.expander('Data Visualization'):
  st.scatter_chart(data=df, x='product_name', y= 'order_date', color= 'category_name')

# Data preparations
with st.sidebar:
  st.header('Input features')
  category_name = st.selectbox('Category Name', ('Electronics', 'Sports & Outdoors', 'Books & Stationery', 'Fashion', 'Home & Living'))
  product_name = st.selectbox('Product Name', ('Smartphone', 'Soccer Ball', 'Tent', 'Story Book', 'Skirt', 'Tablet', 'Yoga Mat', 'Pillow', 'Blanket'
                                               , 'Smartwatch', 'Notebook', 'Laptop', 'Pen', 'Shirt', 'Carpet', 'Novel', 'Eraser', 'Vase', 'Dress'
                                               , 'Pants', 'T-shirt', 'Running Shoes'))
  price = st.slider('Price', 18, 480, 230)
  payment_method = st.selectbox('Payment Method', ('Credit Card', 'Bank Transfer', 'Cash on Delivery'))
  city = st.selectbox('City', ('New Oliviaberg', 'Port Matthew', 'West Sarah', 'Hernandezburgh', 'Jenkinshaven', 'East Tonyaberg', 'North Jessicabury'
                              , 'Aliciaberg', 'West Larrymouth', 'Lake Ian', 'Elizabethmouth', 'Melanieberg', 'Allisonland', 'Myershaven', 'South Tonya'
                              , 'Port Allisonfort', 'Levyport', 'Fullerland', 'North Anthony', 'North Whitneytown', 'West Cynthiaton', 'East Christopher'
                              , 'Port Danielleview', 'East Christopherborough', 'Douglasport', 'North Jamesside', 'North Carrie', 'Port Kenneth'
                              , 'East Kylie', 'Mendezburgh', 'Kristenland', 'Lake Sarah', 'Fryeberg', 'Lake Michael', 'North Mary', 'Cynthiaport'
                              , 'East Corytown', 'Martinezview', 'North Terrancehaven', 'Lake Rhondatown', 'Williamston', 'Lake Jeffrey', 'New Carolfort'
                              , 'Franklinmouth', 'New Terri', 'Jeffreyview', 'Teresaville', 'Mcdonaldmouth', 'West Geraldhaven', 'West Geraldhaven', 'Walkerland'
                              , 'Port Patriciashire', 'Jacobburgh', 'New Brittanytown', 'Woodshaven', 'Rachelland', 'Gonzalezshire', 'Blakeshire', 'Whitakerview'
                              , 'Port Thomaston', 'New Jeremy', 'Lake Nancy', 'Lake Jasmineport', 'Garrisonberg', 'South Katherineside', 'Mccallhaven', 'Lake Teresafurt'
                              , 'Roymouth', 'East Charles', 'West Krista', 'Hobbston', 'Jacobfurt', 'New Felicia', 'Spencerside', 'West Jacob'))
  gender = st.selectbox('Gender', ('M', 'F'))
  age = st.slider('Age', 18, 75, 28)



# Create a Dataframe for the input features
  data = {'category_name': category_name,
        'product_name': product_name,
        'price': price,
        'payment_method': payment_method,
        'city': city,
        'gender': gender,
        'age': age}
  input_df = pd.DataFrame(data, index=[0])
  input_sales = pd.concat([input_df, X], axis=0)

with st.expander('Input features'):
  st.write('**Input**')
  input_df
  st.write('**Combined**')
  input_sales

# Encode

encode = ['category_name', 'product_name', 'payment_method', 'city','gender', 'age']
df_sales = pd.get_dummies(input_df, prefix=encode)
df_sales[:1]
