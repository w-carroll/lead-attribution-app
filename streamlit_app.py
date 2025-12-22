import streamlit as st
import pandas as pd
import plotly.express as px
import re

# --- Page Config ---
st.set_page_config(page_title="Lead Attribution App", layout="wide")

# --- Password Protection ---
def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Please enter the app password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password incorrect. Please try again", type="password", on_change=password_entered, key="password")
        return False
    else:
        return True

if not check_password():
    st.stop()

# --- Helper Functions ---
def clean_phone_get_last6(phone):
    if pd.isna(phone): return None
    phone_str = str(phone)
    numeric_phone = re.sub(r'\D', '', phone_str)
    if len(numeric_phone) >= 6: return numeric_phone[-6:]
    return None

def clean_currency(val):
    if pd.isna(val): return 0.0
    val_str = str(val)
    clean_str = re.sub(r'[£, ]', '', val_str)
    try: return float(clean_str)
    except ValueError: return 0.0

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# --- Main App Interface ---
st.title("📊 Lead Attribution Analysis")
st.markdown("Upload your exports to see which marketing channels are driving revenue.")

# File Uploader Section
with st.expander("📂 Upload Files", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1: google_file = st.file_uploader("Google Leads (CSV)", type="csv")
    with col2: meta_file = st.file_uploader("Meta Leads (CSV)", type="csv")
    with col3: keap_file = st.file_uploader("Keap Orders (CSV)", type="csv")

if google_file and meta_file and keap_file:
    st.divider()
    with st.spinner('Crunching the numbers...'):
        try:
            # --- Load & Process ---
            google_leads = pd.read_csv(google_file)
            meta_leads = pd.read_csv(meta_file)
            keap_orders = pd.read_csv(keap_file)

            # IDs & Currency
            google_leads['match_id'] = google_leads['Phone number'].apply(clean_phone_get_last6)
            meta_leads['match_id'] = meta_leads['Phone'].apply(clean_phone_get_last6)
            keap_orders['match_id'] = keap_orders['Phone 1'].apply(clean_phone_get_last6)
            keap_orders['OrderValue'] = keap_orders['OrderTotal'].apply(clean_currency)

            # Dates
            if 'Submit time GMT' in google_leads.columns:
                google_leads['LeadDate'] = pd.to_datetime(google_leads['Submit time GMT'], errors='coerce')
            if 'Created' in meta_leads.columns:
                meta_leads['LeadDate'] = pd.to_datetime(meta_leads['Created'], errors='coerce')
            if 'OrderDate' in keap_orders.columns:
                keap_orders['OrderDateParsed'] = pd.to_datetime(keap_orders['OrderDate'], dayfirst=True, errors='coerce')
            else:
                keap_orders['OrderDateParsed'] = pd.NaT

            # --- Merging ---
            # Google
            g_cols = ['match_id', 'Phone number', 'Email', 'First name', 'Last name', 'Lead Stage', 
                      'Which course are you interested in?', 'Campaign name', 'Ad Group name', 'LeadDate']
            g_cols = [c for c in g_cols if c in google_leads.columns]
            
            k_cols = ['match_id', 'OrderId', 'OrderTotal', 'OrderValue', 'Phone 1', 'First Name', 'Last Name', 
                      'Order: Payment Type', 'Order: Centre where training is to be undertaken', 'OrderDateParsed']
            k_cols = [c for c in k_cols if c in keap_orders.columns]
            
            google_merged = pd.merge(google_leads[g_cols], keap_orders[k_cols], on='match_id', how='inner', suffixes=('_Lead', '_Order'))
            google_merged['Source'] = 'Google Ads'
            
            # Meta
            m_cols = ['match_id', 'Phone', 'Email address', 'Name', 'Form', 'LeadDate']
            m_cols = [c for c in m_cols if c in meta_leads.columns]
            
            meta_merged = pd.merge(meta_leads[m_cols], keap_orders[k_cols], on='match_id', how='inner')
            meta_merged['Source'] = 'Meta'
            
            # Standardize
            if 'Name' in meta_merged.columns: meta_merged.rename(columns={'Name': 'Full Name_Lead'}, inplace=True)
            if 'Phone' in meta_merged.columns: meta_merged.rename(columns={'Phone': 'Phone number'}, inplace=True)
            
            # Combine
            all_matches = pd.concat([google_merged, meta_merged], ignore_index=True)
            
            # Days to Convert
            if not all_matches.empty:
                all_matches['DaysToConvert'] = (all_matches['OrderDateParsed'] - all_matches['LeadDate']).dt.days
                all_matches['DaysToConvert'] = all_matches['DaysToConvert'].fillna(-1).astype(int)

            # --- KEY METRICS ROW ---
            # Calculate totals for the KPI cards
            total_rev_google = google_merged.drop_duplicates('OrderId')['OrderValue'].sum()
            total_rev_meta = meta_merged.drop_duplicates('OrderId')['OrderValue'].sum()
            total_sales_google = google_merged['OrderId'].nunique()
            total_sales_meta = meta_merged['OrderId'].nunique()

            st.subheader("🚀 Performance Snapshot")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Attributed Revenue", f"£{total_rev_google + total_rev_meta:,.0f}")
            m2.metric("Total Sales Count", total_sales_google + total_sales_meta)
            m3.metric("Google Revenue", f"£{total_rev_google:,.0f}")
            m4.metric("Meta Revenue", f"£{total_rev_meta:,.0f}")

            # --- TABS FOR ANALYSIS ---
            tab_viz, tab_data, tab_details = st.tabs(["📈 Visualizations", "📄 Summary Data", "🔍 Deep Dive Tables"])

            with tab_viz:
                # --- NEW: TREND ANALYSIS ---
                st.markdown("#### Revenue Trend over Time")
                
                if not all_matches.empty and 'OrderDateParsed' in all_matches.columns:
                    # Filter out any rows with missing dates for the chart
                    trend_df = all_matches.dropna(subset=['OrderDateParsed']).copy()
                    
                    if not trend_df.empty:
                        # Control for Time Frequency
                        col_trend1, col_trend2 = st.columns([1, 4])
                        with col_trend1:
                            time_freq = st.selectbox("Group By:", ["Weekly", "Monthly", "Daily"], index=0)
                        
                        # Resample logic
                        freq_map = {"Weekly": "W", "Monthly": "M", "Daily": "D"}
                        rule = freq_map[time_freq]
                        
                        # Group by Source and Time
                        trend_grouped = (trend_df.set_index('OrderDateParsed')
                                         .groupby([pd.Grouper(freq=rule), 'Source'])['OrderValue']
                                         .sum()
                                         .reset_index())
                        
                        # Plot Line Chart
                        fig_trend = px.line(trend_grouped, x='OrderDateParsed', y='OrderValue', color='Source',
                                            markers=True, title=f"Revenue Trend ({time_freq})",
                                            labels={'OrderDateParsed': 'Date', 'OrderValue': 'Revenue (£)'},
                                            color_discrete_map={'Google Ads': '#4285F4', 'Meta': '#1877F2'})
                        st.plotly_chart(fig_trend, use_container_width=True)
                    else:
                        st.info("No order dates found in the data to plot trends.")
                else:
                    st.info("Orders must have 'OrderDate' to show trends.")

                st.divider()

                col_chart1, col_chart2 = st.columns(2)
                
                # Chart 1: Revenue by Source
                with col_chart1:
                    st.markdown("#### Revenue by Source")
                    df_rev = pd.DataFrame({
                        'Source': ['Google Ads', 'Meta'],
                        'Revenue': [total_rev_google, total_rev_meta]
                    })
                    fig_rev = px.bar(df_rev, x='Source', y='Revenue', color='Source', text_auto='.2s', 
                                     color_discrete_map={'Google Ads': '#4285F4', 'Meta': '#1877F2'})
                    st.plotly_chart(fig_rev, use_container_width=True)

                # Chart 2: Sales Count
                with col_chart2:
                    st.markdown("#### Sales Count by Source")
                    df_sales = pd.DataFrame({
                        'Source': ['Google Ads', 'Meta'],
                        'Sales': [total_sales_google, total_sales_meta]
                    })
                    fig_sales = px.bar(df_sales, x='Source', y='Sales', color='Source', text_auto=True,
                                       color_discrete_map={'Google Ads': '#4285F4', 'Meta': '#1877F2'})
                    st.plotly_chart(fig_sales, use_container_width=True)
                
                # Chart 3: Google Campaigns
                if 'Campaign name' in google_merged.columns:
                    st.markdown("#### Top Google Campaigns (Revenue)")
                    camp_perf = google_merged.drop_duplicates('OrderId').groupby('Campaign name')['OrderValue'].sum().reset_index()
                    if not camp_perf.empty:
                        fig_camp = px.bar(camp_perf, y='Campaign name', x='OrderValue', orientation='h', 
                                          title="Revenue by Campaign", text_auto='.2s')
                        st.plotly_chart(fig_camp, use_container_width=True)

                # Chart 4: Time to Convert
                if not all_matches.empty and 'DaysToConvert' in all_matches.columns:
                    valid_days = all_matches[all_matches['DaysToConvert'] >= 0]
                    if not valid_days.empty:
                        st.markdown("#### Speed to Sale (Days from Lead to Order)")
                        fig_hist = px.histogram(valid_days, x="DaysToConvert", color="Source", nbins=20,
                                                title="Distribution of Conversion Times",
                                                labels={'DaysToConvert': 'Days'},
                                                color_discrete_map={'Google Ads': '#4285F4', 'Meta': '#1877F2'})
                        st.plotly_chart(fig_hist, use_container_width=True)

            with tab_data:
                st.subheader("Attribution Summary")
                summary_stats = []
                for source_name, original_df, merged_df in [('Google Ads', google_leads, google_merged), ('Meta', meta_leads, meta_merged)]:
                    total_leads = len(original_df)
                    unique_sales_df = merged_df.drop_duplicates(subset=['OrderId'])
                    sales_count = len(unique_sales_df)
                    total_revenue = unique_sales_df['OrderValue'].sum()
                    conv_str = f"1 in {(total_leads/sales_count):.1f}" if sales_count > 0 else "No Sales"

                    summary_stats.append({
                        "Source": source_name,
                        "Total Leads": total_leads,
                        "Sales Count": sales_count,
                        "Conversion": conv_str,
                        "Revenue": f"£{total_revenue:,.2f}"
                    })
                st.dataframe(pd.DataFrame(summary_stats), hide_index=True)

                # Duplicate Checker
                st.subheader("Duplicate Detector")
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    if 'OrderId' in google_merged.columns:
                        g_dupes = google_merged[google_merged.duplicated(subset=['OrderId'], keep=False)]
                        if not g_dupes.empty:
                            st.warning(f"Google: {len(g_dupes)} duplicate matches found.")
                            st.dataframe(g_dupes[['OrderId', 'First name']].sort_values('OrderId'), hide_index=True)
                        else:
                            st.success("Google: Clean (No duplicates)")
                with col_d2:
                    if 'OrderId' in meta_merged.columns:
                        m_dupes = meta_merged[meta_merged.duplicated(subset=['OrderId'], keep=False)]
                        if not m_dupes.empty:
                            st.warning(f"Meta: {len(m_dupes)} duplicate matches found.")
                            st.dataframe(m_dupes[['OrderId', 'Full Name_Lead']].sort_values('OrderId'), hide_index=True)
                        else:
                            st.success("Meta: Clean (No duplicates)")

            with tab_details:
                st.markdown("### Detailed Breakdowns")
                
                def display_breakdown(original_df, merged_df, group_col, title):
                    if group_col not in original_df.columns: return
                    st.markdown(f"**{title}**")
                    
                    total = original_df[group_col].value_counts().reset_index()
                    total.columns = [group_col, 'Total']
                    
                    sold = merged_df.groupby(group_col)['OrderId'].nunique().reset_index()
                    sold.columns = [group_col, 'Sold']
                    
                    rev = merged_df.drop_duplicates(subset=['OrderId']).groupby(group_col)['OrderValue'].sum().reset_index()
                    rev.columns = [group_col, 'Revenue']
                    
                    stats = pd.merge(total, sold, on=group_col, how='left').fillna(0)
                    stats = pd.merge(stats, rev, on=group_col, how='left').fillna(0)
                    stats['Sold'] = stats['Sold'].astype(int)
                    stats['Conv %'] = ((stats['Sold'] / stats['Total']) * 100).map('{:.1f}%'.format)
                    stats['Revenue'] = stats['Revenue'].map('£{:,.2f}'.format)
                    st.dataframe(stats, hide_index=True, use_container_width=True)

                c1, c2 = st.columns(2)
                with c1:
                    display_breakdown(google_leads, google_merged, 'Lead Stage', "Google Lead Stage")
                    display_breakdown(google_leads, google_merged, 'Campaign name', "Google Campaign")
                with c2:
                    display_breakdown(google_leads, google_merged, 'Which course are you interested in?', "Google Course Interest")
                    display_breakdown(meta_leads, meta_merged, 'Form', "Meta Forms")

            # --- DOWNLOAD ---
            if not all_matches.empty:
                st.divider()
                st.subheader("📥 Download Data")
                save_cols = ['Source', 'match_id', 'OrderId', 'OrderTotal', 'Phone number', 
                             'Phone 1', 'LeadDate', 'OrderDate', 'DaysToConvert']
                if 'First name' in all_matches.columns: save_cols.insert(4, 'First name')
                save_cols = [c for c in save_cols if c in all_matches.columns]
                
                csv = convert_df(all_matches[save_cols])
                st.download_button("Download Verification CSV", data=csv, file_name='matched_sales_verification.csv', mime='text/csv')

        except Exception as e:
            st.error(f"An error occurred: {e}")
