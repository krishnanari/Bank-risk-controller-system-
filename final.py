import pandas as pd 
import numpy as np
import streamlit as st
import plotly.express as px
import pickle
from PIL import Image
import re

# Load your datasets
df = pd.read_csv("Datae.csv")
df1 = pd.read_csv("Data.csv")
df2 = pd.read_csv("Data1.csv")

# Function to safely convert to sqrt
def safe_sqrt(value):
    try:
        return np.sqrt(float(value))  # Convert to float and take sqrt
    except (ValueError, TypeError):
        return np.nan  
# Define occupation types in alphabetical order with corresponding numeric codes
occupation_types = {
    0: 'Accountants',
    1: 'Cleaning staff',
    2: 'Cooking staff',
    3: 'Core staff',
    4: 'Drivers',
    5: 'HR staff',
    6: 'High skill tech staff',
    7: 'IT staff',
    8: 'Laborers',
    9: 'Low-skill Laborers',
    10: 'Managers',
    11: 'Medicine staff',
    12: 'Private service staff',
    13: 'Realty agents',
    14: 'Sales staff',
    15: 'Secretaries',
    16: 'Security staff',
    17: 'Waiters/barmen staff',
}


# Mapping for NAME_EDUCATION_TYPE
education_type_mapping = {'Secondary / secondary special': 0, 'Higher education': 1, 'Incomplete higher': 2, 'Lower secondary': 3, 'Academic degree': 4}

gender_mapping = {'F': 0, 'M': 1, 'XNA': 2}
own_car_mapping = {'N': 0, 'Y': 1,}
# Mapping for NAME_FAMILY_STATUS
family_status_mapping = {'Single / not married': 3, 'Married': 1, 'Civil marriage': 0, 'Widow': 4, 'Separated': 2}

# Main Streamlit code
# -------------------------------------------------- Logo & details on top

st.markdown("# :orange[*Bank*] *Risk* :orange[*Controller*] *System*")
st.markdown("""
<hr style="border: none; height: 5px; background-color: #FFFFFF;" />
""", unsafe_allow_html=True)

# Define tab options
tabs = ["Home", "Data Showcase", "ML Prediction", "ML Recommendation system", "Data Visualization", "About"]
selected_tab = st.selectbox("Select a tab", tabs)

# Home tab content
if selected_tab == "Home":
    st.markdown("### :orange[*OVERVIEW* ]")
    st.markdown("### *The expected outcome of this project is a robust predictive model that can accurately identify customers who are likely to default on their loans. This will enable the financial institution to proactively manage their credit portfolio, implement targeted interventions, and ultimately reduce the risk of loan defaults.*")
    st.markdown("### :orange[*DOMAIN* ] ")
    st.markdown(" ### *Banking* ")
    st.markdown("""
                ### :orange[*TECHNOLOGIES USED*]     
                ### *PYTHON*
                ### *DATA PREPROCESSING*
                ### *EDA*
                ### *PANDAS*
                ### *NUMPY*
                ### *VISUALIZATION*
                ### *MACHINE LEARNING*
                ### *STREAMLIT GUI*
                """)

# Data Showcase tab content
elif selected_tab == "Data Showcase":
    st.header("Data Used")
    st.dataframe(df)

    st.header("Model Performance")
    data = {
        "Algorithm": ["Decision Tree","Random Forest","KNN","XGradientBoost"],
        "Accuracy": [89,88,97,93],
        "Precision": [90,90,96,94],
        "Recall": [89,89,96,94],
        "F1 Score": [89,89,97,94]
    }
    dff = pd.DataFrame(data)
    st.dataframe(dff)
    st.markdown(f"## The Selected Algorithm is :orange[*KNN*] and its Accuracy is   :orange[*97%*]")


elif selected_tab == "ML Prediction":
    st.markdown(f'## :violet[*Predicting Customers Default on Loans*]')
    st.write('<h5 style="color:#FBCEB1;"><i>NOTE: Min & Max given for reference, you can enter any value</i></h5>', unsafe_allow_html=True)

    with st.form("my_form"):
        col1, col2 = st.columns([5, 5])
        
        with col1:
            TOTAL_INCOME = st.text_input("TOTAL INCOME (Min: 25650.0 & Max: 117000000.0)", key='TOTAL_INCOME')
            AMOUNT_CREDIT = st.text_input("CREDIT AMOUNT (Min: 45000.0 & Max: 4050000.0)", key='AMOUNT_CREDIT')
            AMOUNT_ANNUITY = st.text_input("ANNUITY AMOUNT (Min: 1980.0 & Max: 225000.0)", key='AMOUNT_ANNUITY')
            OCCUPATION_TYPE_CODE = st.selectbox("OCCUPATION TYPE (0 to 17)", sorted(occupation_types.items()), format_func=lambda x: x[1], key='OCCUPATION_TYPE_CODE')[0]
            GENDER = st.selectbox("GENDER", list(gender_mapping.keys()), key='GENDER')
        with col2:
            OWN_CAR = st.selectbox("OWN CAR", list(own_car_mapping.keys()), key='OWN_CAR')
            EDUCATION_TYPE = st.selectbox("EDUCATION TYPE", list(education_type_mapping.keys()), key='EDUCATION_TYPE')
            FAMILY_STATUS = st.selectbox("FAMILY STATUS", list(family_status_mapping.keys()), key='FAMILY_STATUS')
            OBS_30_COUNT = st.text_input("OBS_30 COUNT (Min: 0 & Max: 348.0)", key='OBS_30_COUNT')
            DEF_30_COUNT = st.text_input("DEF_30 COUNT (Min: 0 & Max: 34.0)", key='DEF_30_COUNT')

        submit_button = st.form_submit_button(label="PREDICT STATUS")

    st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #FBCEB1;
            color: purple;
            width: 50%;
            display: block;
            margin: auto;
        }
        </style>
    """, unsafe_allow_html=True)

    # Validate input
    flag = 0 
    pattern = r"^(?:\d+|\d*\.\d+)$"

    for i in [TOTAL_INCOME, AMOUNT_CREDIT, AMOUNT_ANNUITY, OBS_30_COUNT, DEF_30_COUNT]:             
        if re.match(pattern, i):
            pass
        else:                    
            flag = 1  
            break

    if submit_button and flag == 1:
        if len(i) == 0:
            st.write("Please enter a valid number, space not allowed")
        else:
            st.write("You have entered an invalid value: ", i)  

    if submit_button and flag == 0:
        try:
            # Convert inputs to appropriate numeric types
            total_income = float(TOTAL_INCOME)
            amount_credit = float(AMOUNT_CREDIT)
            amount_annuity = float(AMOUNT_ANNUITY)
            occupation_type_code = int(OCCUPATION_TYPE_CODE)
            gender_code = gender_mapping[GENDER]
            own_car_code = own_car_mapping[OWN_CAR]
            education_type_code = education_type_mapping[EDUCATION_TYPE]
            family_status_code = family_status_mapping[FAMILY_STATUS]
            obs_30_count = float(OBS_30_COUNT)
            def_30_count = float(DEF_30_COUNT)

            # Construct sample array for prediction
            sample = np.array([
                [
                    safe_sqrt(total_income),
                    safe_sqrt(amount_credit),
                    safe_sqrt(amount_annuity),
                    occupation_type_code,
                    gender_code,
                    own_car_code,
                    education_type_code,
                    family_status_code,
                    safe_sqrt(obs_30_count),
                    safe_sqrt(def_30_count)
                ]
            ])

            # Load the model
            with open(r"knnmodel.pkl", 'rb') as file:
                knn = pickle.load(file)

            # Perform prediction
            pred = knn.predict(sample)

            # Display prediction result
            if pred == 0:
                st.markdown(f' ## :grey[The status is :] :orange[Repay]')
            else:
                st.write(f' ## :orange[The status is ] :grey[Won\'t Repay]')

        except ValueError as e:
            st.error(f"Error processing inputs: {e}")

# ML Recommendation system tab content
elif selected_tab == "ML Recommendation system":
    st.markdown(f'## :violet[*Mobile Recommendation system*]')
    st.write(" ")
    st.write( f'<h5 style="color:#FBCEB1;"><i>NOTE: Min & Max given for reference, you can enter any value</i></h5>', unsafe_allow_html=True )

    
    # Load the mobile dataset from CSV
    data = pd.read_csv("mobile_recommendation_system_dataset.csv")
    
    # Convert the 'price' column to numeric, coercing errors to NaN
    data['price'] = pd.to_numeric(data['price'], errors='coerce')
    
    # Drop rows with NaN prices (if any)
    data.dropna(subset=['price'], inplace=True)
    
    # Extract unique brands and sort them for the dropdown menu
    brands = sorted(data['name'].apply(lambda x: x.split()[0]).unique())
    
    # Add a dropdown to select a brand
    selected_brand = st.selectbox("Select Brand", brands, key='brand_selectbox')
    
    # Select rating range from dropdown with a unique key based on function name
    select_rating = st.selectbox("Select rating", [3, 4, 5], key='rating_selectbox')
    
    # Select price range from dropdown with a unique key based on function name
    select_price = st.selectbox("Select price range", ["< ₹10,000", "₹10,000 - ₹20,000", "₹20,000 - ₹30,000", "₹30,000 - ₹40,000", "> ₹40,000"], key='price_selectbox')
    
    # Function to recommend mobile phones based on rating, price range, and brand
    def recommend(brand, min_rating, max_rating, min_price, max_price):
        # Filter by brand, rating, and price range
        filtered_data = data[(data['name'].str.contains(brand, case=False)) &
                                (data['ratings'] >= min_rating) & (data['ratings'] < max_rating) &
                                (data['price'] >= min_price) & (data['price'] < max_price)]
        return filtered_data
    
    # Determine rating range based on selection
    if select_rating == 3:
        min_rating, max_rating = 3.0, 4.0
    elif select_rating == 4:
        min_rating, max_rating = 4.0, 5.0
    elif select_rating == 5:
        min_rating, max_rating = 5.0, 5.1  # Assuming ratings are <= 5.0
    
    # Determine price range based on selection
    if select_price == "< ₹10,000":
        min_price, max_price = 0, 10000
    elif select_price == "₹10,000 - ₹20,000":
        min_price, max_price = 10000, 20000
    elif select_price == "₹20,000 - ₹30,000":
        min_price, max_price = 20000, 30000
    elif select_price == "₹30,000 - ₹40,000":
        min_price, max_price = 30000, 40000
    elif select_price == "> ₹40,000":
        min_price, max_price = 40000, float('inf')  # Assuming no upper limit for the price

    # Show recommendations when button is clicked
    if st.button("Show Recommendations"):
        filtered_data = recommend(selected_brand, min_rating, max_rating, min_price, max_price)
        
        # Display selected brand, rating, and price range
        st.markdown(f"## Selected Brand: {selected_brand}")
        st.markdown(f"## Selected Rating: {select_rating}")
        st.markdown(f"## Selected Price Range: {select_price}")
        
        # Display recommended mobile phones
        st.markdown("## Recommendations:")
        if not filtered_data.empty:
            for index, row in filtered_data.iterrows():
                st.markdown(f"### Name: {row['name']}")
                st.markdown(f"#### Ratings: {row['ratings']}")
                st.markdown(f"#### Price: ₹{row['price']}")
                st.markdown(f"![Image]({row['imgURL']})")
                st.write("")
        else:
            st.markdown("No recommendations found based on the given criteria.")

# Data Visualization tab content
elif selected_tab == "Data Visualization":
    st.subheader("Insights of Bank Risk Controller System")
             
    # Assuming df is your DataFrame and 'AMT_INCOME_TOTAL' is your column of interest
    fig = px.histogram(df, x='AMT_INCOME_TOTAL_sqrt', nbins=50, marginal='box', histnorm='density')

    # Add KDE line
    fig.update_traces(marker_color='blue', opacity=0.7)
    fig.add_scatter(x=df['AMT_INCOME_TOTAL_sqrt'], y=df['AMT_INCOME_TOTAL_sqrt'].value_counts(normalize=True).sort_index(),
                    mode='lines', name='KDE', line=dict(color='red'))

    # Update layout for better visualization
    fig.update_layout(title='AMT_INCOME_TOTAL Distribution with KDE',
                      xaxis_title='AMT_INCOME_TOTAL',
                      yaxis_title='Density',
                      showlegend=True)

    st.plotly_chart(fig, use_container_width=True)

    #--------------------------------------------------------------4

    # Bar Plot: Top 10 Occupation Types
    occupation_counts = df1['OCCUPATION_TYPE'].value_counts().reset_index()
    occupation_counts.columns = ['OCCUPATION_TYPE', 'COUNT']

    # Create a bar chart
    fig = px.bar(occupation_counts, y='OCCUPATION_TYPE', x='COUNT', color="COUNT", title='Occupation Type Counts', color_continuous_scale='PiYG')
    st.plotly_chart(fig, use_container_width=True)

    #--------------------------------------------------------------5

    INCOME_counts = df1['NAME_INCOME_TYPE'].value_counts().reset_index()
    INCOME_counts.columns = ['NAME_INCOME_TYPE', 'COUNT']

    # Create a line chart
    fig = px.line(INCOME_counts, x='NAME_INCOME_TYPE', y='COUNT', title='Income Type Counts')
    st.plotly_chart(fig, use_container_width=True)

    #--------------------------------------------------------------6

    family = df1['NAME_FAMILY_STATUS'].value_counts().reset_index()
    family.columns = ['NAME_FAMILY_STATUS', 'COUNT']

    # Create a pie chart
    fig = px.pie(family, names='NAME_FAMILY_STATUS', values='COUNT', title='Family Status Distribution')
    st.plotly_chart(fig, use_container_width=True)

    #--------------------------------------------------------------7

    EDUCATION_counts = df1['NAME_EDUCATION_TYPE'].value_counts().reset_index()
    EDUCATION_counts.columns = ['NAME_EDUCATION_TYPE', 'COUNT']

    # Create a bar chart
    fig = px.bar(EDUCATION_counts, x='NAME_EDUCATION_TYPE', y='COUNT', color='COUNT',
                 color_continuous_scale='Viridis', title='Occupation Type Counts')
    fig.update_layout(legend_title_text='Education Type')
    st.plotly_chart(fig, use_container_width=True)

    #--------------------------------------------------------------3

    fig2 = px.pie(df1, names='NAME_CONTRACT_TYPE_x', title='Distribution of Contract Types')
    fig2.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig2, use_container_width=True)

    #--------------------------------------------------------------2

    dff = df2[['AMT_INCOME_TOTAL_sqrt',
               'AMT_CREDIT_x_sqrt', 'AMT_ANNUITY_x_sqrt',
               'OCCUPATION_TYPE_sqrt', 'NAME_EDUCATION_TYPE_sqrt',
               'AMT_GOODS_PRICE_x_sqrt',
               'OBS_30_CNT_SOCIAL_CIRCLE_sqrt', "TARGET"]]

    # Calculate the correlation matrix
    corr = dff.corr().round(2)

    # Plot the heatmap using Plotly Express
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu",
                    title="Correlation Matrix Heatmap")
    st.plotly_chart(fig, use_container_width=True)

    #--------------------------------------------------------------8

    fig1 = px.histogram(df1, x='OCCUPATION_TYPE', color='TARGET', barmode='group')
    fig1.update_layout(title='Countplot of TARGET by OCCUPATION_TYPE', xaxis_title='OCCUPATION_TYPE', yaxis_title='Count')
    st.plotly_chart(fig1, use_container_width=True)

# About tab content
elif selected_tab == "About":
    st.markdown("""
        ## About Bank Risk Controller System
        This application is developed as part of the Bank Risk Controller System project. It aims to provide a predictive model for identifying customers likely to default on their loans, leveraging machine learning and data analysis techniques.
        For more information, contact us at [arikrishnan23121999@gmail.com](mailto:email@domain.com).
    """)

