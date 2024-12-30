import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.title("Bike Sharing Data Dashboard")
st.markdown("### Insights on Bike Sharing Data")

# Load Data
@st.cache_data
def load_data():
    df_hour = pd.read_csv("hour.csv")
    df_day = pd.read_csv("day.csv")
    df_hour['dteday'] = pd.to_datetime(df_hour['dteday'])
    df_day['dteday'] = pd.to_datetime(df_day['dteday'])
    return df_hour, df_day

df_hour, df_day = load_data()

# Sidebar for Navigation
st.sidebar.title("Navigation")
options = [
    "Monthly Trends",
    "Weather Conditions",
    "Hourly Trends",
    "Seasonal Trends",
    "User Types",
    "RFM Analysis",
    "Predictive Analysis"
]
selection = st.sidebar.radio("Go to:", options)

# Monthly Trends
if selection == "Monthly Trends":
    st.header("Monthly Trends")
    st.write("#### Insight: Observing seasonal patterns in bike usage over months.")
    df_hour['year'] = df_hour['dteday'].dt.year
    df_hour['month'] = df_hour['dteday'].dt.month
    monthly_counts = df_hour.groupby(['year', 'month'])['cnt'].sum().reset_index()
    monthly_counts_pivot = monthly_counts.pivot(index='month', columns='year', values='cnt')

    plt.figure(figsize=(12, 6))
    for year in monthly_counts['year'].unique():
        plt.plot(
            monthly_counts_pivot.index,
            monthly_counts_pivot[year],
            marker='o',
            label=str(year)
        )

    plt.title("Monthly Total Count for Two Years", fontsize=16)
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Total Count", fontsize=12)
    plt.xticks(
        ticks=range(1, 13),
        labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        rotation=45
    )
    plt.legend(title="Year")
    plt.grid(visible=True, linestyle='--', alpha=0.7)
    st.pyplot(plt)

# Weather Conditions
elif selection == "Weather Conditions":
    st.header("Weather Conditions")
    st.write("#### Insight: Clear weather tends to have the highest bike usage.")

    # Define weather categories
    weather_categories = {
        1: 'Clear or Partly Cloudy',
        2: 'Mist or Cloudy',
        3: 'Light Rain/Snow',
        4: 'Heavy Rain/Snow'
    }

    # Map weather situation to categories
    df_day['weather_desc'] = df_day['weathersit'].map(weather_categories)

    # Count the occurrences for each category
    weather_counts = df_day['weather_desc'].value_counts()

    # Ensure categories match the data
    categories = list(weather_counts.index)
    counts = list(weather_counts.values)

    # Create bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(categories, counts, color='skyblue', edgecolor='black')
    plt.title('Distribution of Weather Conditions', fontsize=14)
    plt.xlabel('Weather Conditions', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(plt)

# Hourly Trends
elif selection == "Hourly Trends":
    st.header("Hourly Trends")
    st.write("#### Insight: Peak hours for bike usage are typically during commute times.")
    hourly_counts = df_hour.groupby('hr')['cnt'].sum()

    plt.figure(figsize=(8, 6))
    plt.plot(hourly_counts.index, hourly_counts.values, marker='o', color='skyblue')
    plt.title('Hourly Bike Usage', fontsize=14)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Total Count', fontsize=12)
    plt.xticks(range(0, 24, 2))
    plt.grid(visible=True, linestyle='--', alpha=0.7)
    st.pyplot(plt)

# Seasonal Trends
elif selection == "Seasonal Trends":
    st.header("Seasonal Trends")
    st.write("#### Insight: Summer sees the highest usage due to favorable weather conditions.")
    plt.figure(figsize=(8, 6))
    sns.barplot(
        x="season",
        y="cnt",
        data=df_day,
        palette="Blues_d",
        edgecolor="black"
    )
    plt.title('Total Count by Season', fontsize=14)
    plt.xlabel('Season', fontsize=12)
    plt.ylabel('Total Count', fontsize=12)
    st.pyplot(plt)

# User Types
elif selection == "User Types":
    st.header("User Types")
    st.write("#### Insight: Registered users account for the majority of bike usage.")
    casual_counts = df_day['casual'].sum()
    registered_counts = df_day['registered'].sum()
    plt.figure(figsize=(8, 6))
    plt.pie(
        [casual_counts, registered_counts],
        labels=['Casual', 'Registered'],
        autopct='%1.1f%%',
        colors=["#D3D3D3", "#72BCD4"]
    )
    plt.title("Comparison of Casual and Registered Users")
    st.pyplot(plt)

# RFM Analysis
elif selection == "RFM Analysis":
    st.header("RFM Analysis")
    st.write("#### Insight: Identifying loyal users through recency, frequency, and monetary analysis.")
    current_date = df_hour['dteday'].max()
    rfm_df = df_hour.groupby('registered').agg(
        Recency=('dteday', lambda x: (current_date - x.max()).days),
        Frequency=('dteday', 'count'),
        Monetary=('cnt', 'sum')
    ).reset_index()

    rfm_df['R_Segment'] = pd.qcut(rfm_df['Recency'], 4, labels=[4, 3, 2, 1])
    rfm_df['F_Segment'] = pd.qcut(rfm_df['Frequency'], 4, labels=[1, 2, 3, 4])
    rfm_df['M_Segment'] = pd.qcut(rfm_df['Monetary'], 4, labels=[1, 2, 3, 4])

    rfm_df['RFM_Segment'] = rfm_df['R_Segment'].astype(str) + rfm_df['F_Segment'].astype(str) + rfm_df['M_Segment'].astype(str)
    rfm_df['RFM_Score'] = rfm_df[['R_Segment', 'F_Segment', 'M_Segment']].sum(axis=1).astype(int)

    def rfm_category(score):
        if score >= 9:
            return 'Best Customers'
        elif score >= 7:
            return 'Loyal Customers'
        elif score >= 5:
            return 'Potential Loyalists'
        elif score >= 3:
            return 'Needs Attention'
        else:
            return 'At Risk'

    rfm_df['Customer_Category'] = rfm_df['RFM_Score'].apply(rfm_category)

    st.dataframe(rfm_df)
    plt.figure(figsize=(8, 6))
    rfm_df['Customer_Category'].value_counts().plot(kind='bar', color='skyblue')
    plt.title('Customer Segmentation by RFM Categories')
    plt.xlabel('Customer Category')
    plt.ylabel('Number of Customers')
    st.pyplot(plt)

# Predictive Analysis
elif selection == "Predictive Analysis":
    st.header("Predictive Analysis")
    st.write("#### Insight: Predicting bike usage based on weather and seasonal data.")
    features = ['temp', 'hum', 'windspeed', 'season', 'weathersit']
    X = df_hour[features]
    y = df_hour['cnt']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.5, color='skyblue')
    plt.title('Actual vs Predicted Counts')
    plt.xlabel('Actual Counts')
    plt.ylabel('Predicted Counts')
    plt.grid(visible=True, linestyle='--', alpha=0.7)
    st.pyplot(plt)
