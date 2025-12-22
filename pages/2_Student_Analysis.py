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

    # 3. Filter Outliers (Test accounts) - Updated list or keep generic
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
    
    # 3. Probabilistic Distrubution
    # Create a list of Month Start/End dates for the next X months
    future_months = []
    start_date = CURRENT_DATE.replace(day=1) # Start of current month
    for i in range(months_ahead):
        m_start = start_date + pd.DateOffset(months=i)
        m_end = m_start + pd.DateOffset(months=1) - timedelta(seconds=1)
        future_months.append((m_start, m_end))

    # We will accumulate "fractional students" here
    monthly_load = {m[0].strftime('%Y-%m'): 0.0 for m in future_months}
    monthly_load['Backlog'] = 0.0

    # Iterate through each student and distribute their probability
    # Using timestamp conversion for NormalDist (since it works with floats)
    for _, row in candidates.iterrows():
        pred_ts = row['Predicted_Date'].timestamp()
        std_dev_seconds = std_days * 24 * 3600
        
        # If std_dev is 0 or NaN (single data point), treat as spike
        if pd.isna(std_dev_seconds) or std_dev_seconds == 0:
            std_dev_seconds = 1 * 24 * 3600 # Default to 1 day spread if no history
            
        dist = NormalDist(mu=pred_ts, sigma=std_dev_seconds)

        # Calculate prob for each month bucket
        total_prob = 0
        for m_start, m_end in future_months:
            p = dist.cdf(m_end.timestamp()) - dist.cdf(m_start.timestamp())
            monthly_load[m_start.strftime('%Y-%m')] += p
            total_prob += p
        
        # Whatever probability is "left over" on the left side is Backlog
        # (Technically some is far future, but for this window we assume left tail is backlog)
        backlog_prob = dist.cdf(future_months[0][0].timestamp())
        monthly_load['Backlog'] += backlog_prob

    # Format Output
    demand_df = pd.DataFrame(list(monthly_load.items()), columns=['Month_Year', 'Expected_Students'])
    demand_df['Event_Type'] = event_name
    demand_df['Classes_Needed_Prob'] = demand_df['Expected_Students'].apply(lambda x: math.ceil(x / 10))
    
    return demand_df

# --- DASHBOARD LAYOUT ---
st.title("🎓 Student Progress & Booking Predictor")

uploaded_file = st.file_uploader("Upload Excel or CSV export", type=['xlsx', 'xls', 'csv'])

if uploaded_file:
    df = load_and_process_data(uploaded_file)
    
    # Sidebar Filters
    st.sidebar.header("Filters")
    topics = list(df['Topic'].unique())
    selected_topic = st.sidebar.multiselect("Select Topic", options=topics, default=topics)
    df_filtered = df[df['Topic'].isin(selected_topic)]

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
        st.markdown("""
        **How this works:**
        Instead of guessing a single date, we use the **Gaussian Distribution** (Bell Curve) of your historical data.
        If a student is predicted for March 31st but your timing varies by ±2 weeks, this model allocates 
        50% of that student to March and 50% to April. This smoothes out the demand curve.
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

        # 2. Generate Probabilistic Predictions
        pred_exams = calculate_probabilistic_demand(df_filtered, stats['Enrol->Exam']['mean'], stats['Enrol->Exam']['std'], 
                                          'Enrolment Date', 'Exam Start Date', 'Exams')
        
        pred_pracs = calculate_probabilistic_demand(df_filtered, stats['Exam->Prac']['mean'], stats['Exam->Prac']['std'], 
                                          'Exam Start Date', 'Prac Start Date', 'Practical Training')
        
        pred_assess = calculate_probabilistic_demand(df_filtered, stats['Prac->Assess']['mean'], stats['Prac->Assess']['std'], 
                                           'Prac Start Date', 'Assessment Start Date', 'Assessments')

        all_preds = pd.concat([pred_exams, pred_pracs, pred_assess])

        if not all_preds.empty:
            # Sort for Chart
            all_preds['Sort_Key'] = all_preds['Month_Year'].apply(lambda x: '0000' if 'Backlog' in x else x)
            all_preds = all_preds.sort_values(['Event_Type', 'Sort_Key'])

            # Display Data
            col_pred1, col_pred2 = st.columns([1, 2])
            
            with col_pred1:
                st.write("#### Resource Requirements")
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
            
            st.info(f"**Model Parameters (Gaussian):**\n"
                    f"- Exams: Mean {stats['Enrol->Exam']['mean']:.0f} days, StdDev ±{stats['Enrol->Exam']['std']:.0f} days\n"
                    f"- Practicals: Mean {stats['Exam->Prac']['mean']:.0f} days, StdDev ±{stats['Exam->Prac']['std']:.0f} days\n"
                    f"- Assessments: Mean {stats['Prac->Assess']['mean']:.0f} days, StdDev ±{stats['Prac->Assess']['std']:.0f} days")
        else:
            st.warning("Not enough data to generate predictions.")

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
