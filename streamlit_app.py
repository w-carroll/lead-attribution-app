import streamlit as st
import pandas as pd
import re
import io

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
        # First run, show input for password.
        st.text_input(
            "Please enter the app password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input again.
        st.text_input(
            "Password incorrect. Please try again", type="password", on_change=password_entered, key="password"
        )
        return False
    else:
        # Password correct.
        return True

if not check_password():
    st.stop()  # Stop execution if password not correct

# --- Helper Functions ---
def clean_phone_get_last6(phone):
    if pd.isna(phone):
        return None
    phone_str = str(phone)
    numeric_phone = re.sub(r'\D', '', phone_str)
    if len(numeric_phone) >= 6:
        return numeric_phone[-6:]
    return None

def clean_currency(val):
    if pd.isna(val):
        return 0.0
    val_str = str(val)
    clean_str = re.sub(r'[£, ]', '', val_str)
    try:
        return float(clean_str)
    except ValueError:
        return 0.0

@st.cache_data
def convert_df(df):
    """Converts dataframe to CSV for download button"""
    return df.to_csv(index=False).encode('utf-8')

# --- Main App Interface ---
st.title("📊 Lead Attribution Analysis")
st.markdown("Upload your exports below to match leads to sales.")

col1, col2, col3 = st.columns(3)
with col1:
    google_file = st.file_uploader("Upload Google Leads (CSV)", type="csv")
with col2:
    meta_file = st.file_uploader("Upload Meta Leads (CSV)", type="csv")
with col3:
    keap_file = st.file_uploader("Upload Keap Orders (CSV)", type="csv")

if google_file and meta_file and keap_file:
    st.divider()
    with st.spinner('Processing data...'):
        try:
            # Load Data
            google_leads = pd.read_csv(google_file)
            meta_leads = pd.read_csv(meta_file)
            keap_orders = pd.read_csv(keap_file)

            # --- PRE-PROCESSING ---
            # Generate IDs
            google_leads['match_id'] = google_leads['Phone number'].apply(clean_phone_get_last6)
            meta_leads['match_id'] = meta_leads['Phone'].apply(clean_phone_get_last6)
            keap_orders['match_id'] = keap_orders['Phone 1'].apply(clean_phone_get_last6)
            
            # Clean Currency
            keap_orders['OrderValue'] = keap_orders['OrderTotal'].apply(clean_currency)

            # Parse Dates
            if 'Submit time GMT' in google_leads.columns:
                google_leads['LeadDate'] = pd.to_datetime(google_leads['Submit time GMT'], errors='coerce')
            if 'Created' in meta_leads.columns:
                meta_leads['LeadDate'] = pd.to_datetime(meta_leads['Created'], errors='coerce')
            if 'OrderDate' in keap_orders.columns:
                keap_orders['OrderDateParsed'] = pd.to_datetime(keap_orders['OrderDate'], dayfirst=True, errors='coerce')
            else:
                keap_orders['OrderDateParsed'] = pd.NaT

            # --- MERGING ---
            # Google
            g_cols = ['match_id', 'Phone number', 'Email', 'First name', 'Last name', 
                      'Lead Stage', 'Which course are you interested in?', 
                      'Campaign name', 'Ad Group name', 'When can you start the classes?', 'LeadDate']
            g_cols = [c for c in g_cols if c in google_leads.columns]
            
            # Keap
            k_cols = ['match_id', 'OrderId', 'OrderTotal', 'OrderValue', 'Phone 1', 
                      'First Name', 'Last Name', 'Order: Payment Type', 
                      'Order: Centre where training is to be undertaken', 'OrderDate', 'OrderDateParsed']
            k_cols = [c for c in k_cols if c in keap_orders.columns]
            
            google_merged = pd.merge(google_leads[g_cols], keap_orders[k_cols], on='match_id', how='inner', suffixes=('_Lead', '_Order'))
            google_merged['Source'] = 'Google Ads'
            
            # Meta
            m_cols = ['match_id', 'Phone', 'Email address', 'Name', 'Form', 'LeadDate']
            m_cols = [c for c in m_cols if c in meta_leads.columns]
            
            meta_merged = pd.merge(meta_leads[m_cols], keap_orders[k_cols], on='match_id', how='inner')
            meta_merged['Source'] = 'Meta'
            
            # Standardize Meta Columns
            if 'Name' in meta_merged.columns: meta_merged.rename(columns={'Name': 'Full Name_Lead'}, inplace=True)
            if 'Phone' in meta_merged.columns: meta_merged.rename(columns={'Phone': 'Phone number'}, inplace=True)
            if 'Email address' in meta_merged.columns: meta_merged.rename(columns={'Email address': 'Email'}, inplace=True)

            # Combine
            all_matches = pd.concat([google_merged, meta_merged], ignore_index=True)

            # Calculate Days to Convert
            if not all_matches.empty and 'LeadDate' in all_matches.columns and 'OrderDateParsed' in all_matches.columns:
                all_matches['DaysToConvert'] = (all_matches['OrderDateParsed'] - all_matches['LeadDate']).dt.days
                all_matches['DaysToConvert'] = all_matches['DaysToConvert'].fillna(-1).astype(int)

            # --- OUTPUT 1: ATTRIBUTION SUMMARY ---
            st.subheader("1. Final Attribution Summary")
            
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
            
            st.table(pd.DataFrame(summary_stats))

            # --- OUTPUT 2: DUPLICATE DETECTOR ---
            st.subheader("2. Duplicate / Discrepancy Detector")
            
            col_d1, col_d2 = st.columns(2)
            
            # Google Duplicates
            with col_d1:
                st.markdown("**Google Ads Duplicates**")
                if 'OrderId' in google_merged.columns:
                    g_dupes = google_merged[google_merged.duplicated(subset=['OrderId'], keep=False)]
                    if not g_dupes.empty:
                        st.warning("Orders matched to multiple leads:")
                        disp_cols = [c for c in ['OrderId', 'First name', 'Lead Stage'] if c in g_dupes.columns]
                        st.dataframe(g_dupes[disp_cols].sort_values('OrderId'), hide_index=True)
                    else:
                        st.success("No duplicates found.")

            # Meta Duplicates
            with col_d2:
                st.markdown("**Meta Duplicates**")
                if 'OrderId' in meta_merged.columns:
                    m_dupes = meta_merged[meta_merged.duplicated(subset=['OrderId'], keep=False)]
                    if not m_dupes.empty:
                        st.warning("Orders matched to multiple leads:")
                        disp_cols = [c for c in ['OrderId', 'Full Name_Lead', 'Form'] if c in m_dupes.columns]
                        st.dataframe(m_dupes[disp_cols].sort_values('OrderId'), hide_index=True)
                    else:
                        st.success("No duplicates found.")

            # --- OUTPUT 3: ADVANCED METRICS ---
            st.subheader("3. Advanced Performance Metrics")
            
            tab1, tab2, tab3, tab4 = st.tabs(["Lead Stage", "Campaigns/Forms", "Speed to Sale", "Misc"])

            def display_metric_table(original_df, merged_df, group_col):
                if group_col not in original_df.columns:
                    st.write(f"Column '{group_col}' not found.")
                    return
                
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

            with tab1:
                st.markdown("### Google Lead Stage")
                display_metric_table(google_leads, google_merged, 'Lead Stage')
                st.markdown("### Google Urgency")
                display_metric_table(google_leads, google_merged, 'When can you start the classes?')

            with tab2:
                st.markdown("### Google Campaigns")
                display_metric_table(google_leads, google_merged, 'Campaign name')
                st.markdown("### Google Course Interest")
                display_metric_table(google_leads, google_merged, 'Which course are you interested in?')
                st.markdown("### Meta Forms")
                display_metric_table(meta_leads, meta_merged, 'Form')

            with tab3:
                st.markdown("### Time to Convert (Days)")
                if not all_matches.empty and 'DaysToConvert' in all_matches.columns:
                    valid_sales = all_matches[all_matches['DaysToConvert'] >= 0]
                    if not valid_sales.empty:
                        time_stats = valid_sales.groupby('Source')['DaysToConvert'].agg(['count', 'mean', 'min', 'max']).reset_index()
                        time_stats.columns = ['Source', 'Count', 'Avg Days', 'Fastest', 'Slowest']
                        time_stats['Avg Days'] = time_stats['Avg Days'].map('{:.1f}'.format)
                        st.dataframe(time_stats, hide_index=True)
                    else:
                        st.info("No valid dates found to calculate conversion time.")
                else:
                    st.info("No matches found.")

            with tab4:
                 st.markdown("### Payment Types (Matched Orders)")
                 if 'Order: Payment Type' in google_merged.columns:
                     pay_grp = google_merged.drop_duplicates('OrderId').groupby('Order: Payment Type').size().reset_index(name='Sales')
                     st.dataframe(pay_grp, hide_index=True)

            # --- OUTPUT 4: DOWNLOAD ---
            st.divider()
            st.subheader("4. Download Verification Data")
            
            if not all_matches.empty:
                save_cols = ['Source', 'match_id', 'OrderId', 'OrderTotal', 'Phone number', 
                             'Phone 1', 'LeadDate', 'OrderDate', 'DaysToConvert']
                if 'First name' in all_matches.columns: save_cols.insert(4, 'First name')
                save_cols = [c for c in save_cols if c in all_matches.columns]
                
                csv = convert_df(all_matches[save_cols])
                
                st.download_button(
                    label="📥 Download Detailed Verification CSV",
                    data=csv,
                    file_name='matched_sales_verification.csv',
                    mime='text/csv',
                )
            else:
                st.warning("No matches found to download.")

        except Exception as e:
            st.error(f"An error occurred: {e}")