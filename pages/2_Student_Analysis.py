import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Student Progress Dashboard", layout="wide")
CURRENT_DATE = pd.Timestamp('2025-12-12') # Update this to your 'Report Date'

# --- DATA PROCESSING FUNCTIONS ---
def clean_location(loc):
    """
    Attempts to extract City from "City Postcode" format.
    E.g., "Manchester M17EW" -> "Manchester"
    """
    if pd.isna(loc):
        return "Unknown"
    loc = str(loc)
    # Strategy: Keep words that don't contain digits (removes postcodes)
    parts = loc.split()
    clean_parts = [p for p in parts if not any(char.isdigit() for char in p)]
    if clean_parts:
        return " ".join(clean_parts)
    return loc # Return original if regex failed (e.g. "M1 5WD")

@st.cache_data
def load_and_process_data(file):
    # Determine file type
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    # 1. Anonymize Names (Handling duplicates)
    # We create a map of Name -> ID based on unique names found
    unique_names = df['Contact Name'].unique()
    name_map = {name: f'Student {i+1}' for i, name in enumerate(unique_names)}
    df['Student_ID'] = df['Contact Name'].map(name_map)
    
    # 2. Date Conversion
    date_cols = ['Enrolment Date', 'Exam Start Date', 'Prac Start Date', 
                 'Assessment Start Date', 'Last Login']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)

    # 3. Clean Location
    if 'Location' in df.columns:
        df['City'] = df['Location'].apply(clean_location)
    else:
        df['City'] = "Unknown"

    # 4. Filter Outliers (Test accounts)
    # You can add specific IDs to exclude here if they persist
    outlier_ids = ['Student 2986', 'Student 765', 'Student 5307'] 
    df = df[~df['Student_ID'].isin(outlier_ids)]

    # 5. Calculate Durations
    df['Days_Enrol_to_Exam'] = (df['Exam Start Date'] - df['Enrolment Date']).dt.days
    df['Days_Enrol_to_Prac'] = (df['Prac Start Date'] - df['Enrolment Date']).dt.days
    
    # 6. Status Logic (Active vs Inactive)
    # Active if Status is Active AND Assessment hasn't passed
    def get_status(row):
        if row['Active'] == 'Inactive':
            return 'Inactive'
        if pd.notnull(row['Assessment Start Date']) and row['Assessment Start Date'] < CURRENT_DATE:
            return 'Inactive'
        return 'Active'
    
    df['Calculated_Status'] = df.apply(get_status, axis=1)
    
    # 7. Engagement (Zombie) Logic
    if 'Last Login' in df.columns:
        df['Days_Since_Login'] = (CURRENT_DATE - df['Last Login']).dt.days
        df['Is_Zombie'] = (df['Calculated_Status'] == 'Active') & (df['Days_Since_Login'] > 90) & (df['Exam Start Date'].isna())
    else:
        df['Is_Zombie'] = False

    return df

# --- DASHBOARD LAYOUT ---
st.title("🎓 Student Progress Analysis")
st.markdown("Upload your learner export to generate the full analysis report.")

uploaded_file = st.file_uploader("Upload Excel or CSV", type=['xlsx', 'xls', 'csv'])

if uploaded_file:
    df = load_and_process_data(uploaded_file)
    
    # Sidebar Filters
    st.sidebar.header("Filters")
    selected_topic = st.sidebar.multiselect("Select Topic", options=df['Topic'].unique(), default=df['Topic'].unique())
    df_filtered = df[df['Topic'].isin(selected_topic)]

    # --- TOP METRICS ---
    col1, col2, col3, col4 = st.columns(4)
    n_enrolled = len(df_filtered)
    n_active = len(df_filtered[df_filtered['Calculated_Status'] == 'Active'])
    n_zombies = df_filtered['Is_Zombie'].sum()
    
    col1.metric("Total Enrolled", n_enrolled)
    col2.metric("Active Learners", n_active)
    col3.metric("Inactive Learners", n_enrolled - n_active)
    col4.metric("Zombie Learners (>90 days inactive)", n_zombies, delta_color="inverse")

    # --- TABS FOR ANALYSIS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📉 Funnel & Drop-off", "⏱️ Time Analysis", "📚 Course Difficulty", "📝 Homework Impact", "🌍 Location"])

    with tab1:
        st.subheader("Student Progression Funnel")
        # Funnel Logic
        stages = {
            'Enrolled': len(df_filtered),
            'Homework Completed': len(df_filtered[df_filtered['Homework Completed'] == 'Yes']),
            'Exam Started': df_filtered['Exam Start Date'].notnull().sum(),
            'Prac Started': df_filtered['Prac Start Date'].notnull().sum(),
            'Assessment Started': df_filtered['Assessment Start Date'].notnull().sum()
        }
        
        funnel_df = pd.DataFrame(list(stages.items()), columns=['Stage', 'Count'])
        funnel_df['% of Enrolled'] = (funnel_df['Count'] / n_enrolled * 100).round(1)
        
        col_a, col_b = st.columns([1, 2])
        col_a.dataframe(funnel_df, hide_index=True)
        
        fig_funnel = px.funnel(funnel_df, x='Count', y='Stage', title="Progression Flow")
        col_b.plotly_chart(fig_funnel, use_container_width=True)
        
        st.info("💡 **Zombie List:** Below are Active students with no exam booked who haven't logged in for >90 days.")
        zombies = df_filtered[df_filtered['Is_Zombie']][['Student_ID', 'Enrolment Date', 'Days_Since_Login', 'Topic', 'City']]
        st.dataframe(zombies)

    with tab2:
        st.subheader("Time to Reach Milestones (Days)")
        
        # Calculate Stats
        stats = []
        for stage, col in [('Exam', 'Days_Enrol_to_Exam'), ('Practical', 'Days_Enrol_to_Prac')]:
            valid = df_filtered[df_filtered[col] > 0][col]
            if not valid.empty:
                stats.append({
                    'Stage': stage,
                    'Average Days': valid.mean(),
                    'Min Days': valid.min(),
                    'Max Days': valid.max(),
                    'Count': len(valid)
                })
        
        stats_df = pd.DataFrame(stats)
        st.table(stats_df.style.format({'Average Days': '{:.1f}'}))
        
        # Histogram
        fig_hist = px.histogram(df_filtered[df_filtered['Days_Enrol_to_Exam']>0], x="Days_Enrol_to_Exam", color="Topic", nbins=50, title="Distribution of Time to Exam")
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab3:
        st.subheader("Topic Difficulty Analysis")
        # Aggregation by Topic
        topic_stats = df_filtered.groupby('Topic').agg(
            Enrolled=('Student_ID', 'count'),
            Reached_Exams=('Exam Start Date', 'count'),
            Avg_Days_Exam=('Days_Enrol_to_Exam', lambda x: x[x>0].mean())
        ).reset_index()
        
        topic_stats['Exam Rate (%)'] = (topic_stats['Reached_Exams'] / topic_stats['Enrolled'] * 100).round(1)
        
        # Combo Chart equivalent in Plotly
        fig_combo = go.Figure()
        fig_combo.add_trace(go.Bar(x=topic_stats['Topic'], y=topic_stats['Reached_Exams'], name='Students at Exam'))
        fig_combo.add_trace(go.Scatter(x=topic_stats['Topic'], y=topic_stats['Avg_Days_Exam'], name='Avg Days to Reach', yaxis='y2', mode='lines+markers'))
        
        fig_combo.update_layout(
            title="Volume vs. Speed per Topic",
            yaxis=dict(title="Number of Students"),
            yaxis2=dict(title="Days to Reach Exam", overlaying='y', side='right')
        )
        st.plotly_chart(fig_combo, use_container_width=True)

    with tab4:
        st.subheader("Homework Correlation")
        st.write("Does completing homework affect the chance of reaching Practical Training?")
        
        hw_stats = df_filtered.groupby('Homework Completed').apply(
            lambda x: pd.Series({
                'Total': len(x),
                'Reached Practical': x['Prac Start Date'].notnull().sum()
            })
        ).reset_index()
        
        hw_stats['Success Rate (%)'] = (hw_stats['Reached Practical'] / hw_stats['Total'] * 100).round(1)
        
        col_hw1, col_hw2 = st.columns([1, 2])
        col_hw1.dataframe(hw_stats, hide_index=True)
        
        fig_hw = px.bar(hw_stats, x='Homework Completed', y='Success Rate (%)', 
                        color='Homework Completed', title="Likelihood of Reaching Practicals",
                        text='Success Rate (%)')
        col_hw2.plotly_chart(fig_hw, use_container_width=True)

    with tab5:
        st.subheader("Geographic Analysis")
        st.write("Top Student Locations (by City)")
        
        # Count by City
        loc_counts = df_filtered['City'].value_counts().reset_index()
        loc_counts.columns = ['City', 'Count']
        loc_counts = loc_counts[loc_counts['City'] != "Unknown"]
        
        # Bar Chart
        fig_loc = px.bar(loc_counts.head(15), x='Count', y='City', orientation='h', title="Top 15 Locations")
        fig_loc.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_loc, use_container_width=True)
        
        st.dataframe(loc_counts)

else:
    st.info("Waiting for file upload...")
