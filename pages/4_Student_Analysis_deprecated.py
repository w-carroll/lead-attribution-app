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
    Includes BACKLOG logic for overdue items.
    """
    
    # Define the Timeline Buckets
    future_months = []
    # Start from current month
    start_date = CURRENT_DATE.replace(day=1)
    for i in range(months_ahead):
        m_start = start_date + pd.DateOffset(months=i)
        m_end = m_start + pd.DateOffset(months=1) - timedelta(seconds=1)
        future_months.append((m_start, m_end))

    # Initialize Demand Grids (Keys must handle 'Backlog')
    # We use a list of keys to ensure order: Backlog -> Month 1 -> Month 2...
    month_keys = ['Backlog (Overdue)'] + [m[0].strftime('%Y-%m') for m in future_months]
    
    demand = {
        'Exams': {k: 0.0 for k in month_keys},
        'Practical Training': {k: 0.0 for k in month_keys},
        'Assessments': {k: 0.0 for k in month_keys}
    }
    
    raw_predictions = []

    # Helper to distribute probability
    def distribute_prob(date_ts, std_dev, target_dict):
        if pd.isna(date_ts) or pd.isna(std_dev) or std_dev == 0:
            std_dev = 1 * 24 * 3600 # Fallback 1 day
        
        dist = NormalDist(mu=date_ts, sigma=std_dev)
        
        # 1. Calculate Backlog (Everything before the first future month)
        first_future_ts = future_months[0][0].timestamp()
        backlog_p = dist.cdf(first_future_ts)
        target_dict['Backlog (Overdue)'] += backlog_p
        
        # 2. Calculate Future Months
        for m_start, m_end in future_months:
            p = dist.cdf(m_end.timestamp()) - dist.cdf(m_start.timestamp())
            target_dict[m_start.strftime('%Y-%m')] += p

    # --- PROCESS STUDENTS ---
    active_students = df[df['Calculated_Status'] == 'Active'].copy()
    
    if active_students.empty:
        return pd.DataFrame(), pd.DataFrame()

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
                
                raw_predictions.append({
                    'Contact Name': row['Contact Name'],
                    'Topic': row['Topic'],
                    'City': row['City'],
                    'Event_Type': 'Exams',
                    'Predicted_Date': pred_exam_date
                })
                
                # Update state
                curr_date = pred_exam_date
            else:
                continue

        # Step 2: PRACTICALS
        if pd.notnull(row['Prac Start Date']):
            curr_date = row['Prac Start Date']
            curr_std = 0
        elif curr_date is not None:
            # Predict Practical
            mean_days = stats['Exam->Prac']['mean']
            std_days = stats['Exam->Prac']['std']
            
            # Accumulate variance
            new_var = (curr_std**2) + ((std_days * 24 * 3600)**2)
            curr_std = math.sqrt(new_var)
            
            pred_prac_date = curr_date + pd.to_timedelta(mean_days, unit='D')
            
            # Add to Demand
            distribute_prob(pred_prac_date.timestamp(), curr_std, demand['Practical Training'])
            
            raw_predictions.append({
                'Contact Name': row['Contact Name'],
                'Topic': row['Topic'],
                'City': row['City'],
                'Event_Type': 'Practical Training',
                'Predicted_Date': pred_prac_date
            })
            
            curr_date = pred_prac_date

        # Step 3: ASSESSMENTS
        if pd.notnull(row['Assessment Start Date']):
            pass
        elif curr_date is not None:
            # Predict Assessment
            mean_days = stats['Prac->Assess']['mean']
            std_days = stats['Prac->Assess']['std']
            
            new_var = (curr_std**2) + ((std_days * 24 * 3600)**2)
            curr_std = math.sqrt(new_var)
            
            pred_assess_date = curr_date + pd.to_timedelta(mean_days, unit='D')
            
            # Add to Demand
            distribute_prob(pred_assess_date.timestamp(), curr_std, demand['Assessments'])
            
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

    # --- TAB 1: BOOKING PREDICTOR (WATERFALL) ---
    with tab1:
        st.subheader("📅 Future Demand Forecaster (Waterfall Simulation)")
        st.markdown("""
        **What you are seeing:** This model simulates the **full future journey** of every student. 
        It predicts exams, then practicals, then assessments in a chain. 
        **Backlog (Overdue)** items are shown first to highlight immediate demand.
        """)
        
        # 1. Calculate Historical Averages
        stats = {}
        for metric, col in [('Enrol->Exam', 'Days_Enrol_to_Exam'), 
                            ('Exam->Prac', 'Days_Exam_to_Prac'), 
                            ('Prac->Assess', 'Days_Prac_to_Assess')]:
            valid = df[df[col] > 0][col]
            if len(valid) > 1:
                stats[metric] = {'mean': valid.mean(), 'std': valid.std()}
            else:
                stats[metric] = {'mean': valid.mean() if not valid.empty else 30, 'std': 15}

        # Display Pipeline Speeds
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Enrolment ➔ Exam", f"{stats['Enrol->Exam']['mean']:.0f} Days", f"±{stats['Enrol->Exam']['std']:.0f} Days")
        col_m2.metric("Exam ➔ Practical", f"{stats['Exam->Prac']['mean']:.0f} Days", f"±{stats['Exam->Prac']['std']:.0f} Days")
        col_m3.metric("Practical ➔ Assessment", f"{stats['Prac->Assess']['mean']:.0f} Days", f"±{stats['Prac->Assess']['std']:.0f} Days")

        # 2. Run Waterfall Prediction
        if not df_filtered.empty:
            demand_df, raw_preds_df = calculate_waterfall_demand(df_filtered, stats, months_ahead=18)
            
            if not demand_df.empty:
                # Sort Event Types
                event_order = ["Exams", "Practical Training", "Assessments"]
                demand_df['Event_Type'] = pd.Categorical(demand_df['Event_Type'], categories=event_order, ordered=True)
                
                # Handle Sorting for Chart (Backlog comes first)
                demand_df['Sort_Key'] = demand_df['Month_Year'].apply(lambda x: '0000' if 'Backlog' in x else x)
                demand_df = demand_df.sort_values(['Event_Type', 'Sort_Key'])

                # Sub-Tabs
                subtab_chart, subtab_raw = st.tabs(["📊 Demand Chart", "📄 Raw Prediction Data"])
                
                with subtab_chart:
                    col_pred1, col_pred2 = st.columns([1, 3])
                    with col_pred1:
                        st.write("#### Resource Plan")
                        st.dataframe(
                            demand_df[['Event_Type', 'Month_Year', 'Expected_Students', 'Classes_Needed_Prob']]
                            .style.format({'Expected_Students': '{:.1f}'}),
                            hide_index=True,
                            use_container_width=True
                        )
                    with col_pred2:
                        st.write("#### Projected Demand Waves")
                        # Filter backlog for line chart (it messes up the time axis)
                        chart_data = demand_df[demand_df['Month_Year'] != 'Backlog (Overdue)']
                        
                        if not chart_data.empty:
                            fig_pred = px.line(chart_data, x='Month_Year', y='Expected_Students', 
                                              color='Event_Type', markers=True,
                                              title="Future Student Flow (Excluding Backlog)",
                                              labels={'Expected_Students': 'Expected Students', 'Month_Year': 'Month'})
                            fig_pred.update_traces(fill='tozeroy')
                            st.plotly_chart(fig_pred, use_container_width=True)
                        else:
                            st.info("No future demand found (Check Backlog in the table on the left).")

                with subtab_raw:
                    st.write("#### Raw Predictions")
                    if not raw_preds_df.empty:
                        raw_preds_df['Predicted_Month'] = raw_preds_df['Predicted_Date'].dt.strftime('%Y-%m')
                        st.dataframe(
                            raw_preds_df.sort_values(['Event_Type', 'Predicted_Date']),
                            hide_index=True,
                            use_container_width=True
                        )
                    else:
                        st.info("No predictions generated.")
            else:
                st.warning("No predictions could be generated. (Check if active students exist).")
        else:
            st.warning("No active students found in current selection.")

    # --- TAB 2: FUNNEL ---
    with tab2:
        st.subheader("Student Progression Funnel")
        n_enrolled = len(df_filtered)
        stages = {
            'Enrolled': n_enrolled,
            'Homework Completed': len(df_filtered[df_filtered['Homework Completed'] == 'Yes']),
            'Exam Started': df_filtered['Exam Start Date'].notnull().sum(),
            'Prac Started': df_filtered['Prac Start Date'].notnull().sum(),
            'Assessment Started': df_filtered['Assessment Start Date'].notnull().sum()
        }
        funnel_df = pd.DataFrame(list(stages.items()), columns=['Stage', 'Count'])
        funnel_df['% of Enrolled'] = (funnel_df['Count'] / n_enrolled * 100).round(1)
        
        col_f1, col_f2 = st.columns([1, 2])
        col_f1.dataframe(funnel_df, hide_index=True)
        fig_funnel = px.funnel(funnel_df, x='Count', y='Stage', title="Progression Flow")
        col_f2.plotly_chart(fig_funnel, use_container_width=True)
        
        zombies = df_filtered[df_filtered['Is_Zombie']]
        st.error(f"⚠️ **Zombie Alert:** {len(zombies)} Active students have not logged in for >90 days.")
        with st.expander("View Zombie List"):
            st.dataframe(zombies[['Contact Name', 'Enrolment Date', 'Days_Since_Login', 'Topic', 'City']])

    # --- TAB 3: SPEED ANALYSIS ---
    with tab3:
        st.subheader("Time to Reach Milestones")
        stats_speed = []
        for stage, col in [('Enrolment -> Exam', 'Days_Enrol_to_Exam'), 
                           ('Exam -> Practical', 'Days_Exam_to_Prac'),
                           ('Practical -> Assessment', 'Days_Prac_to_Assess')]:
            valid = df_filtered[df_filtered[col] > 0][col]
            if not valid.empty:
                stats_speed.append({
                    'Stage': stage,
                    'Average Days': valid.mean(),
                    'Min': valid.min(),
                    'Max': valid.max(),
                    'Count': len(valid)
                })
        st.table(pd.DataFrame(stats_speed).style.format({'Average Days': '{:.1f}'}))

    # --- TAB 4: COURSE DIFFICULTY ---
    with tab4:
        st.subheader("Topic Difficulty")
        topic_stats = df_filtered.groupby('Topic').agg(
            Enrolled=('Contact Name', 'count'),
            Reached_Exams=('Exam Start Date', 'count'),
            Avg_Days_Exam=('Days_Enrol_to_Exam', lambda x: x[x>0].mean())
        ).reset_index()
        topic_stats['Exam Rate (%)'] = (topic_stats['Reached_Exams'] / topic_stats['Enrolled'] * 100).round(1)
        
        fig_combo = go.Figure()
        fig_combo.add_trace(go.Bar(x=topic_stats['Topic'], y=topic_stats['Reached_Exams'], name='Students at Exam'))
        fig_combo.add_trace(go.Scatter(x=topic_stats['Topic'], y=topic_stats['Avg_Days_Exam'], name='Avg Days to Reach', yaxis='y2', mode='lines+markers'))
        fig_combo.update_layout(yaxis2=dict(overlaying='y', side='right'))
        st.plotly_chart(fig_combo, use_container_width=True)

    # --- TAB 5: HOMEWORK IMPACT ---
    with tab5:
        st.subheader("Homework Impact on Success")
        hw_stats = df_filtered.groupby('Homework Completed').apply(
            lambda x: pd.Series({
                'Total': len(x),
                'Reached Practical': x['Prac Start Date'].notnull().sum()
            })
        ).reset_index()
        hw_stats['Success Rate (%)'] = (hw_stats['Reached Practical'] / hw_stats['Total'] * 100).round(1)
        
        fig_hw = px.bar(hw_stats, x='Homework Completed', y='Success Rate (%)', 
                        color='Homework Completed', text='Success Rate (%)',
                        title="Likelihood of Reaching Practicals based on Homework")
        st.plotly_chart(fig_hw, use_container_width=True)

    # --- TAB 6: LOCATION ---
    with tab6:
        st.subheader("Geographic Breakdown")
        loc_counts = df_filtered['City'].value_counts().reset_index()
        loc_counts.columns = ['City', 'Count']
        loc_counts = loc_counts[loc_counts['City'] != "Unknown"]
        fig_loc = px.bar(loc_counts.head(20), x='Count', y='City', orientation='h', title="Top Student Locations")
        fig_loc.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_loc, use_container_width=True)

else:
    st.info("👋 Welcome! Please upload your 'Learner Progress Report' CSV or Excel file to begin.")
