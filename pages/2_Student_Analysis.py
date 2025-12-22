import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math
from statistics import NormalDist
from datetime import timedelta

# --- CONFIGURATION ---
st.set_page_config(page_title="Student Progress & Booking Predictor", layout="wide")
CURRENT_DATE = pd.Timestamp.now()

# --- DATA PROCESSING FUNCTIONS ---
def clean_location(loc):
    if pd.isna(loc):
        return "Unknown"
    loc = str(loc)
    parts = loc.split()
    clean_parts = [p for p in parts if not any(char.isdigit() for char in p)]
    if clean_parts:
        return " ".join(clean_parts)
    return loc

@st.cache_data
def load_and_process_data(file):
    # Load Data
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    # 1. Date Conversion
    date_cols = ['Enrolment Date', 'Exam Start Date', 'Prac Start Date', 
                 'Assessment Start Date', 'Last Login']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)

    # 2. Clean Location
    if 'Location' in df.columns:
        df['City'] = df['Location'].apply(clean_location)
    else:
        df['City'] = "Unknown"

    # 3. Filter Outliers (Test accounts)
    outlier_names = ['Student 2986', 'Student 765', 'Student 5307'] 
    df = df[~df['Contact Name'].isin(outlier_names)]

    # 4. Calculate Durations & Stats
    df['Days_Enrol_to_Exam'] = (df['Exam Start Date'] - df['Enrolment Date']).dt.days
    df['Days_Exam_to_Prac'] = (df['Prac Start Date'] - df['Exam Start Date']).dt.days
    df['Days_Prac_to_Assess'] = (df['Assessment Start Date'] - df['Prac Start Date']).dt.days
    
    # 5. Status Logic
    def get_status(row):
        if row['Active'] == 'Inactive':
            return 'Inactive'
        if pd.notnull(row['Assessment Start Date']) and row['Assessment Start Date'] < CURRENT_DATE:
            return 'Inactive'
        return 'Active'
    
    df['Calculated_Status'] = df.apply(get_status, axis=1)
    
    # 6. Zombie Logic
    if 'Last Login' in df.columns:
        df['Days_Since_Login'] = (CURRENT_DATE - df['Last Login']).dt.days
        df['Is_Zombie'] = (df['Calculated_Status'] == 'Active') & (df['Days_Since_Login'] > 90) & (df['Exam Start Date'].isna())
    else:
        df['Is_Zombie'] = False

    return df

# --- CHAINED PREDICTION ENGINE ---
def calculate_waterfall_demand(df, stats, months_ahead=18):
    """
    Simulates the full future journey for every active student.
    Returns aggregated demand per month for Exams, Practicals, and Assessments.
    """
    
    # Define the Timeline Buckets
    future_months = []
    start_date = CURRENT_DATE.replace(day=1)
    for i in range(months_ahead):
        m_start = start_date + pd.DateOffset(months=i)
        m_end = m_start + pd.DateOffset(months=1) - timedelta(seconds=1)
        future_months.append((m_start, m_end))

    # Initialize Demand Grids
    demand = {
        'Exams': {m[0].strftime('%Y-%m'): 0.0 for m in future_months},
        'Practical Training': {m[0].strftime('%Y-%m'): 0.0 for m in future_months},
        'Assessments': {m[0].strftime('%Y-%m'): 0.0 for m in future_months}
    }
    
    # Initialize Raw Data Collection
    raw_predictions = []

    # Helper to distribute probability
    def distribute_prob(date_ts, std_dev, target_dict):
        if pd.isna(date_ts) or pd.isna(std_dev) or std_dev == 0:
            std_dev = 1 * 24 * 3600 # Fallback 1 day
        
        dist = NormalDist(mu=date_ts, sigma=std_dev)
        
        for m_start, m_end in future_months:
            p = dist.cdf(m_end.timestamp()) - dist.cdf(m_start.timestamp())
            target_dict[m_start.strftime('%Y-%m')] += p

    # --- PROCESS STUDENTS ---
    # We iterate through active students and project their REMAINING steps
    
    active_students = df[df['Calculated_Status'] == 'Active'].copy()
    
    for _, row in active_students.iterrows():
        # Current State Tracking
        curr_date = None
        curr_std = 0 # Variance accumulates
        
        # Step 1: EXAMS
        if pd.notnull(row['Exam Start Date']):
            # Already done/booked
            curr_date = row['Exam Start Date']
            curr_std = 0 # Known date, no uncertainty
        else:
            # Predict Exam
            if pd.notnull(row['Enrolment Date']):
                mean_days = stats['Enrol->Exam']['mean']
                std_days = stats['Enrol->Exam']['std']
                
                pred_exam_date = row['Enrolment Date'] + pd.to_timedelta(mean_days, unit='D')
                curr_std = std_days * 24 * 3600
                
                # Add to Demand
                distribute_prob(pred_exam_date.timestamp(), curr_std, demand['Exams'])
                
                # Store for Raw Data
                raw_predictions.append({
                    'Contact Name': row['Contact Name'],
                    'Topic': row['Topic'],
                    'City': row['City'],
                    'Event_Type': 'Exams',
                    'Predicted_Date': pred_exam_date
                })
                
                # Update state for next step
                curr_date = pred_exam_date
            else:
                continue # Cannot predict without enrolment date

        # Step 2: PRACTICALS
        if pd.notnull(row['Prac Start Date']):
            # Already done/booked
            curr_date = row['Prac Start Date']
            curr_std = 0
        elif curr_date is not None:
            # Predict Practical (based on Exam Date - real or predicted)
            mean_days = stats['Exam->Prac']['mean']
            std_days = stats['Exam->Prac']['std']
            
            # Combine uncertainties: Var_total = Var_1 + Var_2 -> Std_total = Sqrt(Std1^2 + Std2^2)
            # Simplified: Add variances
            new_var = (curr_std**2) + ((std_days * 24 * 3600)**2)
            curr_std = math.sqrt(new_var)
            
            pred_prac_date = curr_date + pd.to_timedelta(mean_days, unit='D')
            
            # Add to Demand
            distribute_prob(pred_prac_date.timestamp(), curr_std, demand['Practical Training'])
            
            # Store for Raw Data
            raw_predictions.append({
                'Contact Name': row['Contact Name'],
                'Topic': row['Topic'],
                'City': row['City'],
                'Event_Type': 'Practical Training',
                'Predicted_Date': pred_prac_date
            })
            
            # Update state
            curr_date = pred_prac_date

        # Step 3: ASSESSMENTS
        if pd.notnull(row['Assessment Start Date']):
            # Already done
            pass
        elif curr_date is not None:
            # Predict Assessment (based on Practical Date - real or predicted)
            mean_days = stats['Prac->Assess']['mean']
            std_days = stats['Prac->Assess']['std']
            
            new_var = (curr_std**2) + ((std_days * 24 * 3600)**2)
            curr_std = math.sqrt(new_var)
            
            pred_assess_date = curr_date + pd.to_timedelta(mean_days, unit='D')
            
            # Add to Demand
            distribute_prob(pred_assess_date.timestamp(), curr_std, demand['Assessments'])
            
             # Store for Raw Data
            raw_predictions.append({
                'Contact Name': row['Contact Name'],
                'Topic': row['Topic'],
                'City': row['City'],
                'Event_Type': 'Assessments',
                'Predicted_Date': pred_assess_date
            })

    # Consolidate Results
    final_df = pd.DataFrame()
    for event_type, month_data in demand.items():
        temp = pd.DataFrame(list(month_data.items()), columns=['Month_Year', 'Expected_Students'])
        temp['Event_Type'] = event_type
        final_df = pd.concat([final_df, temp])
        
    final_df['Classes_Needed_Prob'] = final_df['Expected_Students'].apply(lambda x: math.ceil(x / 10))
    
    return final_df, pd.DataFrame(raw_predictions)

# --- DASHBOARD LAYOUT ---
st.title("🎓 Student Progress & Booking Predictor")

uploaded_file = st.file_uploader("Upload Excel or CSV export", type=['xlsx', 'xls', 'csv'])

if uploaded_file:
    df = load_and_process_data(uploaded_file)
    
    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Filters")
    
    topics = list(df['Topic'].unique())
    selected_topic = st.sidebar.multiselect("Select Topic", options=topics, default=topics)
    
    cities = sorted(list(df['City'].unique()))
    selected_city = st.sidebar.multiselect("Select Location", options=cities, default=cities)
    
    # Apply Filters
    df_filtered = df[
        (df['Topic'].isin(selected_topic)) & 
        (df['City'].isin(selected_city))
    ]

    # --- TABS ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔮 Booking Predictor", 
        "📉 Funnel", 
        "⏱️ Speed Analysis", 
        "📚 Course Difficulty", 
        "📝 Homework Impact", 
        "🌍 Locations"
    ])

    # --- TAB 1: BOOKING PREDICTOR (WATER
