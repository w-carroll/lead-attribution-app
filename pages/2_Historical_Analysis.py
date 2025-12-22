import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURATION ---
st.set_page_config(page_title="Historical Analysis", layout="wide")
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
    
    # Zombie Logic
    if 'Last Login' in df.columns:
        df['Days_Since_Login'] = (CURRENT_DATE - df['Last Login']).dt.days
        df['Is_Zombie'] = (df['Calculated_Status'] == 'Active') & (df['Days_Since_Login'] > 90) & (df['Exam Start Date'].isna())
    else:
        df['Is_Zombie'] = False

    return df

# --- DASHBOARD LAYOUT ---
st.title("📊 Historical Analysis Dashboard")
st.markdown("Analysis of past student performance, drop-offs, and engagement.")

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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📉 Funnel & Drop-offs", 
        "⏱️ Speed Analysis", 
        "📚 Course Difficulty", 
        "📝 Homework Impact", 
        "🌍 Locations"
    ])

    # --- TAB 1: FUNNEL ---
    with tab1:
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

    # --- TAB 2: SPEED ANALYSIS ---
    with tab2:
        st.subheader("Time to Reach Milestones (Historical Actuals)")
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

    # --- TAB 3: COURSE DIFFICULTY ---
    with tab3:
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

    # --- TAB 4: HOMEWORK IMPACT ---
    with tab4:
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

    # --- TAB 5: LOCATION ---
    with tab5:
        st.subheader("Geographic Breakdown")
        loc_counts = df_filtered['City'].value_counts().reset_index()
        loc_counts.columns = ['City', 'Count']
        loc_counts = loc_counts[loc_counts['City'] != "Unknown"]
        fig_loc = px.bar(loc_counts.head(20), x='Count', y='City', orientation='h', title="Top Student Locations")
        fig_loc.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_loc, use_container_width=True)

else:
    st.info("👋 Upload data to view Historical Analysis.")
