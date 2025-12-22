import streamlit as st
import pandas as pd
import plotly.express as px
import math
from statistics import NormalDist
from datetime import timedelta

# --- CONFIGURATION ---
st.set_page_config(page_title="Future Forecasting", layout="wide")
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
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    # Date Conversion
    date_cols = ['Enrolment Date', 'Exam Start Date', 'Prac Start Date', 
                 'Assessment Start Date', 'Last Login']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)

    # Clean Location
    if 'Location' in df.columns:
        df['City'] = df['Location'].apply(clean_location)
    else:
        df['City'] = "Unknown"

    # Filter Outliers
    outlier_names = ['Student 2986', 'Student 765', 'Student 5307'] 
    df = df[~df['Contact Name'].isin(outlier_names)]

    # Calculate Durations
    df['Days_Enrol_to_Exam'] = (df['Exam Start Date'] - df['Enrolment Date']).dt.days
    df['Days_Exam_to_Prac'] = (df['Prac Start Date'] - df['Exam Start Date']).dt.days
    df['Days_Prac_to_Assess'] = (df['Assessment Start Date'] - df['Prac Start Date']).dt.days
    
    # Status Logic
    def get_status(row):
        if row['Active'] == 'Inactive':
            return 'Inactive'
        if pd.notnull(row['Assessment Start Date']) and row['Assessment Start Date'] < CURRENT_DATE:
            return 'Inactive'
        return 'Active'
    
    df['Calculated_Status'] = df.apply(get_status, axis=1)
    
    return df

# --- WATERFALL PREDICTION ENGINE ---
def calculate_waterfall_demand(df, stats, months_ahead=18):
    future_months = []
    start_date = CURRENT_DATE.replace(day=1)
    for i in range(months_ahead):
        m_start = start_date + pd.DateOffset(months=i)
        m_end = m_start + pd.DateOffset(months=1) - timedelta(seconds=1)
        future_months.append((m_start, m_end))

    month_keys = ['Backlog (Overdue)'] + [m[0].strftime('%Y-%m') for m in future_months]
    
    demand = {
        'Exams': {k: 0.0 for k in month_keys},
        'Practical Training': {k: 0.0 for k in month_keys},
        'Assessments': {k: 0.0 for k in month_keys}
    }
    
    raw_predictions = []

    def distribute_prob(date_ts, std_dev, target_dict):
        if pd.isna(date_ts) or pd.isna(std_dev) or std_dev == 0:
            std_dev = 1 * 24 * 3600
        
        dist = NormalDist(mu=date_ts, sigma=std_dev)
        first_future_ts = future_months[0][0].timestamp()
        backlog_p = dist.cdf(first_future_ts)
        target_dict['Backlog (Overdue)'] += backlog_p
        
        for m_start, m_end in future_months:
            p = dist.cdf(m_end.timestamp()) - dist.cdf(m_start.timestamp())
            target_dict[m_start.strftime('%Y-%m')] += p

    active_students = df[df['Calculated_Status'] == 'Active'].copy()
    
    if active_students.empty:
        return pd.DataFrame(), pd.DataFrame()

    for _, row in active_students.iterrows():
        curr_date = None
        curr_std = 0
        
        # 1. EXAMS
        if pd.notnull(row['Exam Start Date']):
            curr_date = row['Exam Start Date']
            curr_std = 0
        else:
            if pd.notnull(row['Enrolment Date']):
                mean_days = stats['Enrol->Exam']['mean']
                std_days = stats['Enrol->Exam']['std']
                pred_exam_date = row['Enrolment Date'] + pd.to_timedelta(mean_days, unit='D')
                curr_std = std_days * 24 * 3600
                distribute_prob(pred_exam_date.timestamp(), curr_std, demand['Exams'])
                raw_predictions.append({
                    'Contact Name': row['Contact Name'], 'Topic': row['Topic'], 'City': row['City'],
                    'Event_Type': 'Exams', 'Predicted_Date': pred_exam_date
                })
                curr_date = pred_exam_date
            else:
                continue

        # 2. PRACTICALS
        if pd.notnull(row['Prac Start Date']):
            curr_date = row['Prac Start Date']
            curr_std = 0
        elif curr_date is not None:
            mean_days = stats['Exam->Prac']['mean']
            std_days = stats['Exam->Prac']['std']
            new_var = (curr_std**2) + ((std_days * 24 * 3600)**2)
            curr_std = math.sqrt(new_var)
            pred_prac_date = curr_date + pd.to_timedelta(mean_days, unit='D')
            distribute_prob(pred_prac_date.timestamp(), curr_std, demand['Practical Training'])
            raw_predictions.append({
                'Contact Name': row['Contact Name'], 'Topic': row['Topic'], 'City': row['City'],
                'Event_Type': 'Practical Training', 'Predicted_Date': pred_prac_date
            })
            curr_date = pred_prac_date

        # 3. ASSESSMENTS
        if pd.notnull(row['Assessment Start Date']):
            pass
        elif curr_date is not None:
            mean_days = stats['Prac->Assess']['mean']
            std_days = stats['Prac->Assess']['std']
            new_var = (curr_std**2) + ((std_days * 24 * 3600)**2)
            curr_std = math.sqrt(new_var)
            pred_assess_date = curr_date + pd.to_timedelta(mean_days, unit='D')
            distribute_prob(pred_assess_date.timestamp(), curr_std, demand['Assessments'])
            raw_predictions.append({
                'Contact Name': row['Contact Name'], 'Topic': row['Topic'], 'City': row['City'],
                'Event_Type': 'Assessments', 'Predicted_Date': pred_assess_date
            })

    final_df = pd.DataFrame()
    for event_type, month_data in demand.items():
        temp = pd.DataFrame(list(month_data.items()), columns=['Month_Year', 'Expected_Students'])
        temp['Event_Type'] = event_type
        final_df = pd.concat([final_df, temp])
        
    final_df['Classes_Needed_Prob'] = final_df['Expected_Students'].apply(lambda x: math.ceil(x / 10))
    return final_df, pd.DataFrame(raw_predictions)

# --- DASHBOARD LAYOUT ---
st.title("🔮 Future Forecasting & Capacity Planner")
st.markdown("Predictive modelling for Exams, Training, and Assessments.")

uploaded_file = st.file_uploader("Upload Excel or CSV export", type=['xlsx', 'xls', 'csv'])

if uploaded_file:
    df = load_and_process_data(uploaded_file)
    
    # --- FILTERS ---
    st.sidebar.header("Filters")
    topics = list(df['Topic'].unique())
    selected_topic = st.sidebar.multiselect("Select Topic", options=topics, default=topics)
    
    cities = sorted(list(df['City'].unique()))
    selected_city = st.sidebar.multiselect("Select Location", options=cities, default=cities)
    
    df_filtered = df[(df['Topic'].isin(selected_topic)) & (df['City'].isin(selected_city))]

    # --- CALCULATE STATS (Training the Model) ---
    stats = {}
    for metric, col in [('Enrol->Exam', 'Days_Enrol_to_Exam'), 
                        ('Exam->Prac', 'Days_Exam_to_Prac'), 
                        ('Prac->Assess', 'Days_Prac_to_Assess')]:
        valid = df[df[col] > 0][col]
        if len(valid) > 1:
            stats[metric] = {'mean': valid.mean(), 'std': valid.std()}
        else:
            stats[metric] = {'mean': valid.mean() if not valid.empty else 30, 'std': 15}

    st.subheader("Model Parameters (Current Pace)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Enrolment ➔ Exam", f"{stats['Enrol->Exam']['mean']:.0f} Days", f"±{stats['Enrol->Exam']['std']:.0f} Days")
    col2.metric("Exam ➔ Practical", f"{stats['Exam->Prac']['mean']:.0f} Days", f"±{stats['Exam->Prac']['std']:.0f} Days")
    col3.metric("Practical ➔ Assessment", f"{stats['Prac->Assess']['mean']:.0f} Days", f"±{stats['Prac->Assess']['std']:.0f} Days")

    # --- RUN PREDICTION ---
    if not df_filtered.empty:
        demand_df, raw_preds_df = calculate_waterfall_demand(df_filtered, stats, months_ahead=18)
        
        if not demand_df.empty:
            event_order = ["Exams", "Practical Training", "Assessments"]
            demand_df['Event_Type'] = pd.Categorical(demand_df['Event_Type'], categories=event_order, ordered=True)
            demand_df['Sort_Key'] = demand_df['Month_Year'].apply(lambda x: '0000' if 'Backlog' in x else x)
            demand_df = demand_df.sort_values(['Event_Type', 'Sort_Key'])

            tab_chart, tab_raw = st.tabs(["📊 Demand Chart & Plan", "📄 Raw Student Predictions"])
            
            with tab_chart:
                col_c1, col_c2 = st.columns([1, 3])
                with col_c1:
                    st.write("#### Resource Plan")
                    st.dataframe(demand_df[['Event_Type', 'Month_Year', 'Expected_Students', 'Classes_Needed_Prob']].style.format({'Expected_Students': '{:.1f}'}), hide_index=True, use_container_width=True)
                with col_c2:
                    st.write("#### Projected Demand Waves")
                    chart_data = demand_df[demand_df['Month_Year'] != 'Backlog (Overdue)']
                    if not chart_data.empty:
                        fig = px.line(chart_data, x='Month_Year', y='Expected_Students', color='Event_Type', markers=True, title="Future Student Flow (Excluding Backlog)")
                        fig.update_traces(fill='tozeroy')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No future demand found (Check Backlog).")
            
            with tab_raw:
                st.dataframe(raw_preds_df.sort_values(['Event_Type', 'Predicted_Date']), hide_index=True, use_container_width=True)
        else:
            st.warning("No predictions generated.")
    else:
        st.warning("No active students found in selection.")
else:
    st.info("👋 Upload data to view Forecasting.")
