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

# --- PROBABILISTIC PREDICTION ENGINE ---
def calculate_probabilistic_demand(df, avg_days, std_days, stage_from, stage_to, event_name, months_ahead=12):
    """
    Calculates expected student count per month using Gaussian Distribution.
    Returns aggregated demand AND the raw student-level predictions.
    """
    # 1. Identify Candidates
    candidates = df[
        (df['Calculated_Status'] == 'Active') & 
        (df[stage_from].notnull()) & 
        (df[stage_to].isnull())
    ].copy()

    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 2. Calculate Predicted Date (Mean) for each student
    candidates['Predicted_Date'] = candidates[stage_from] + pd.to_timedelta(avg_days, unit='D')
    candidates['Event_Type'] = event_name
    
    # 3. Probabilistic Distrubution
    future_months = []
    start_date = CURRENT_DATE.replace(day=1) # Start of current month
    for i in range(months_ahead):
        m_start = start_date + pd.DateOffset(months=i)
        m_end = m_start + pd.DateOffset(months=1) - timedelta(seconds=1)
        future_months.append((m_start, m_end))

    monthly_load = {m[0].strftime('%Y-%m'): 0.0 for m in future_months}
    monthly_load['Backlog'] = 0.0

    for _, row in candidates.iterrows():
        pred_ts = row['Predicted_Date'].timestamp()
        std_dev_seconds = std_days * 24 * 3600
        
        if pd.isna(std_dev_seconds) or std_dev_seconds == 0:
            std_dev_seconds = 1 * 24 * 3600 
            
        dist = NormalDist(mu=pred_ts, sigma=std_dev_seconds)

        for m_start, m_end in future_months:
            p = dist.cdf(m_end.timestamp()) - dist.cdf(m_start.timestamp())
            monthly_load[m_start.strftime('%Y-%m')] += p
        
        backlog_prob = dist.cdf(future_months[0][0].timestamp())
        monthly_load['Backlog'] += backlog_prob

    # Format Aggregated Output
    demand_df = pd.DataFrame(list(monthly_load.items()), columns=['Month_Year', 'Expected_Students'])
    demand_df['Event_Type'] = event_name
    demand_df['Classes_Needed_Prob'] = demand_df['Expected_Students'].apply(lambda x: math.ceil(x / 10))
    
    # Format Raw Data Output (Cleaned for display)
    raw_df = candidates[['Contact Name', 'City', 'Topic', stage_from, 'Predicted_Date', 'Event_Type']].copy()
    raw_df['Predicted_Month'] = raw_df['Predicted_Date'].dt.strftime('%Y-%m')
    
    return demand_df, raw_df

# --- DASHBOARD LAYOUT ---
st.title("🎓 Student Progress & Booking Predictor")

uploaded_file = st.file_uploader("Upload Excel or CSV export", type=['xlsx', 'xls', 'csv'])

if uploaded_file:
    df = load_and_process_data(uploaded_file)
    
    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Filters")
    
    # Topic Filter
    topics = list(df['Topic'].unique())
    selected_topic = st.sidebar.multiselect("Select Topic", options=topics, default=topics)
    
    # Location Filter (New)
    cities = sorted(list(df['City'].unique()))
    # Default to all, but allow filtering
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

    # --- TAB 1: BOOKING PREDICTOR (UPDATED) ---
    with tab1:
        st.subheader("📅 Future Demand Forecaster (Probabilistic)")
        
        # 1. Calculate Historical Averages (Global Stats to keep model robust even if filtered)
        stats = {}
        for metric, col in [('Enrol->Exam', 'Days_Enrol_to_Exam'), 
                            ('Exam->Prac', 'Days_Exam_to_Prac'), 
                            ('Prac->Assess', 'Days_Prac_to_Assess')]:
            valid = df[df[col] > 0][col] # Use full DF for stats
            if len(valid) > 1:
                stats[metric] = {'mean': valid.mean(), 'std': valid.std()}
            else:
                stats[metric] = {'mean': valid.mean() if not valid.empty else 30, 'std': 15}

        # Display Model Parameters (The "X, Y, Z" times)
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Avg Time to Exam", f"{stats['Enrol->Exam']['mean']:.0f} Days", f"±{stats['Enrol->Exam']['std']:.0f} Days")
        col_m2.metric("Avg Time to Practical", f"{stats['Exam->Prac']['mean']:.0f} Days", f"±{stats['Exam->Prac']['std']:.0f} Days")
        col_m3.metric("Avg Time to Assessment", f"{stats['Prac->Assess']['mean']:.0f} Days", f"±{stats['Prac->Assess']['std']:.0f} Days")

        # 2. Generate Predictions (on Filtered Data)
        d1, r1 = calculate_probabilistic_demand(df_filtered, stats['Enrol->Exam']['mean'], stats['Enrol->Exam']['std'], 
                                          'Enrolment Date', 'Exam Start Date', 'Exams')
        
        d2, r2 = calculate_probabilistic_demand(df_filtered, stats['Exam->Prac']['mean'], stats['Exam->Prac']['std'], 
                                          'Exam Start Date', 'Prac Start Date', 'Practical Training')
        
        d3, r3 = calculate_probabilistic_demand(df_filtered, stats['Prac->Assess']['mean'], stats['Prac->Assess']['std'], 
                                           'Prac Start Date', 'Assessment Start Date', 'Assessments')

        all_preds = pd.concat([d1, d2, d3])
        all_raw = pd.concat([r1, r2, r3])

        if not all_preds.empty:
            # Sort for Chart
            all_preds['Sort_Key'] = all_preds['Month_Year'].apply(lambda x: '0000' if 'Backlog' in x else x)
            all_preds = all_preds.sort_values(['Event_Type', 'Sort_Key'])

            # Sub-Tabs for Chart vs Raw Data
            subtab_chart, subtab_raw = st.tabs(["📊 Demand Chart", "📄 Raw Prediction Data"])
            
            with subtab_chart:
                col_pred1, col_pred2 = st.columns([1, 3])
                with col_pred1:
                    st.write("#### Resource Plan")
                    st.dataframe(
                        all_preds[['Event_Type', 'Month_Year', 'Expected_Students', 'Classes_Needed_Prob']]
                        .style.format({'Expected_Students': '{:.1f}'}),
                        hide_index=True,
                        use_container_width=True
                    )
                with col_pred2:
                    st.write("#### Smoothed Demand Curve")
                    chart_data = all_preds[all_preds['Month_Year'] != 'Backlog']
                    if not chart_data.empty:
                        fig_pred = px.bar(chart_data, x='Month_Year', y='Expected_Students', 
                                          color='Event_Type', barmode='group',
                                          title="Expected Student Load (Gaussian Smoothed)",
                                          labels={'Expected_Students': 'Expected Students', 'Month_Year': 'Month'})
                        st.plotly_chart(fig_pred, use_container_width=True)
                    else:
                        st.info("No future demand found.")

            with subtab_raw:
                st.write("#### Raw Student Predictions")
                st.markdown("This list shows every student included in the prediction above, along with their **Predicted Date** based on the average time for their next stage.")
                st.dataframe(
                    all_raw.sort_values(['Event_Type', 'Predicted_Date']),
                    hide_index=True,
                    use_container_width=True
                )
        else:
            st.warning("Not enough active students at these stages to generate predictions.")

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
