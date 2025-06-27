#!/usr/bin/env python3
"""
Fleet Management Dashboard - FIXED TIME CALCULATION VERSION
Dashboard with proper time parsing for hh:mm format
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import subprocess
import os
from dotenv import load_dotenv
import sys
from datetime import datetime
import json
import base64
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Dashboard tổ xe UMC",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }

    .header-container {
        text-align: center;
        display: flex;
        flex-direction: row;
        align-items: center;
        justify-content: center;
        margin-bottom: 2rem;
    }

    .header-text {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# COLUMN MAPPING - Vietnamese to English
COLUMN_MAPPING = {
    # Drop these columns (set to None to ignore)
    'Timestamp': None,
    'Email Address': None,
    'Ghi chú': None,
    'Chỉ số đồng hồ sau khi kết thúc chuyến xe': None,
    'Ghi nhận chi tiết chuyến xe': None,
    
    # Core time fields
    'Thời gian bắt đầu': 'start_time',
    'Thời gian kết thúc': 'end_time', 
    'Thời gian': 'duration_hours',  # Duration in hours (hh:mm format)
    
    # Location and classification
    'Điểm đến': 'destination',
    'Phân loại công tác': 'work_category',
    'Nội thành/ngoại thành': 'area_type',
    
    # Date and numeric metrics
    'Ngày ghi nhận': 'record_date',
    'Quãng đường': 'distance_km',
    'Đổ nhiên liệu': 'fuel_liters',
    
    # Revenue (ambulance only)
    'Doanh thu': 'revenue_vnd',
    'Chi tiết chuyến xe': 'trip_details',
    
    # Vehicle and driver info
    'Mã xe': 'vehicle_id',
    'Tên tài xế': 'driver_name',
    'Loại xe': 'vehicle_type'
}

def get_github_token():
    """Get GitHub token for private repo access"""
    try:
        if hasattr(st, 'secrets') and 'GITHUB_TOKEN' in st.secrets:
            return st.secrets['GITHUB_TOKEN']
    except:
        pass
    
    token = os.getenv('GITHUB_TOKEN')
    if token and len(token) > 10:
        return token
    
    if os.path.exists("github_token.txt"):
        try:
            with open("github_token.txt", 'r') as f:
                token = f.read().strip()
            if token and token != "YOUR_TOKEN_HERE" and len(token) > 10:
                return token
        except:
            pass
    
    return None

def parse_duration_to_hours(duration_str):
    """
    🔧 ROBUST TIME PARSER - Converts hh:mm format to decimal hours
    Handles all edge cases including None, NaN, empty strings, and invalid formats
    
    Args:
        duration_str: Time in hh:mm format (can be str, float, int, None, NaN)
    
    Returns:
        float: Hours as decimal (e.g., 2:30 -> 2.5)
    """
    # Step 1: Handle None and NaN values
    if duration_str is None:
        return 0.0
    
    if pd.isna(duration_str):
        return 0.0
    
    if isinstance(duration_str, (float, np.floating)) and np.isnan(duration_str):
        return 0.0
    
    # Step 2: Handle empty strings and whitespace
    if str(duration_str).strip() == "":
        return 0.0
    
    # Step 3: Handle already numeric values (edge case)
    if isinstance(duration_str, (int, float, np.number)):
        try:
            val = float(duration_str)
            return max(0.0, val) if val >= 0 else 0.0
        except:
            return 0.0
    
    # Step 4: Process string values
    try:
        duration_str = str(duration_str).strip().upper()
        
        # Handle special cases
        if duration_str in ['NA', 'NAN', 'NONE', 'NULL', '']:
            return 0.0
        
        # Remove AM/PM if present
        if "AM" in duration_str or "PM" in duration_str:
            duration_str = duration_str.split()[0]
        
        # Step 5: Parse time format
        if ":" in duration_str:
            parts = duration_str.split(":")
            
            if len(parts) == 2:  # hh:mm
                try:
                    hours = int(float(parts[0]))
                    minutes = int(float(parts[1]))
                    
                    # Validate ranges
                    if hours < 0 or minutes < 0 or minutes >= 60:
                        return 0.0
                    
                    return hours + minutes / 60.0
                    
                except (ValueError, TypeError):
                    return 0.0
                    
            elif len(parts) == 3:  # hh:mm:ss
                try:
                    hours = int(float(parts[0]))
                    minutes = int(float(parts[1]))
                    seconds = int(float(parts[2]))
                    
                    # Validate ranges
                    if hours < 0 or minutes < 0 or seconds < 0 or minutes >= 60 or seconds >= 60:
                        return 0.0
                    
                    return hours + minutes / 60.0 + seconds / 3600.0
                    
                except (ValueError, TypeError):
                    return 0.0
            else:
                return 0.0
        else:
            # Try to parse as decimal hours
            try:
                val = float(duration_str)
                return max(0.0, val) if val >= 0 else 0.0
            except:
                return 0.0
                
    except Exception:
        return 0.0

def validate_duration_column(df):
    """
    🔧 DURATION VALIDATOR - Ensures duration column is properly processed
    """
    if df is None or df.empty or 'duration_hours' not in df.columns:
        return df
    
    original_count = len(df)
    
    # Apply robust parsing
    df['duration_hours'] = df['duration_hours'].apply(parse_duration_to_hours)
    
    # Convert to numeric and fill NaN with 0
    df['duration_hours'] = pd.to_numeric(df['duration_hours'], errors='coerce').fillna(0.0)
    
    # Filter out unrealistic values (more than 24 hours per trip)
    df.loc[df['duration_hours'] > 24, 'duration_hours'] = 0.0
    df.loc[df['duration_hours'] < 0, 'duration_hours'] = 0.0
    
    # Log parsing results
    total_hours = df['duration_hours'].sum()
    valid_records = (df['duration_hours'] > 0).sum()
    
    st.sidebar.success(f"⏱️ Duration: {total_hours:.1f}h from {valid_records}/{original_count} records")
    
    return df

def parse_distance(distance_str):
    """Parse distance string and handle negative values"""
    if pd.isna(distance_str):
        return 0.0
    
    try:
        distance = float(distance_str)
        return abs(distance) if distance < 0 else distance
    except:
        return 0.0

def parse_revenue(revenue_str):
    """Parse revenue string and handle both formats: 600000 and 600,000"""
    if pd.isna(revenue_str) or revenue_str == '':
        return 0.0
    
    try:
        revenue_str = str(revenue_str).strip()
        revenue_str = revenue_str.replace(',', '')
        revenue_str = revenue_str.replace('VNĐ', '').replace('đ', '').replace('VND', '')
        revenue_str = revenue_str.strip()
        
        revenue = float(revenue_str)
        return abs(revenue) if revenue < 0 else revenue
        
    except (ValueError, TypeError):
        return 0.0

@st.cache_data(ttl=60)
def load_data_from_github():
    """Load data from GitHub repository"""
    github_token = get_github_token()
    
    if not github_token:
        st.sidebar.error("❌ Cần GitHub token để truy cập private repo")
        return pd.DataFrame()
    
    headers = {
        'Authorization': f'token {github_token}',
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'Fleet-Dashboard-App'
    }
    
    api_url = "https://api.github.com/repos/corner-25/vehicle-storage/contents/data/latest/fleet_data_latest.json"
    
    try:
        response = requests.get(api_url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            api_response = response.json()
            
            if api_response.get('size', 0) > 1000000:
                return load_large_file_via_git_api(headers)
            
            content = base64.b64decode(api_response['content']).decode('utf-8')
            
            if not content.strip():
                return load_large_file_via_git_api(headers)
            
            data = json.loads(content)
            df = pd.DataFrame(data)
            return process_dataframe(df)
        else:
            return load_large_file_via_git_api(headers)
            
    except Exception:
        return load_large_file_via_git_api(headers)

def load_large_file_via_git_api(headers):
    """Load large file using Git API"""
    try:
        commits_url = "https://api.github.com/repos/corner-25/vehicle-storage/commits/main"
        commits_response = requests.get(commits_url, headers=headers, timeout=30)
        
        if commits_response.status_code != 200:
            return pd.DataFrame()
        
        latest_commit = commits_response.json()
        tree_sha = latest_commit['commit']['tree']['sha']
        
        tree_url = f"https://api.github.com/repos/corner-25/vehicle-storage/git/trees/{tree_sha}"
        tree_response = requests.get(tree_url, headers=headers, timeout=30)
        
        if tree_response.status_code != 200:
            return pd.DataFrame()
        
        tree_data = tree_response.json()
        data_folder = None
        for item in tree_data.get('tree', []):
            if item['path'] == 'data' and item['type'] == 'tree':
                data_folder = item['sha']
                break
        
        if not data_folder:
            return pd.DataFrame()
        
        data_tree_url = f"https://api.github.com/repos/corner-25/vehicle-storage/git/trees/{data_folder}"
        data_tree_response = requests.get(data_tree_url, headers=headers, timeout=30)
        
        if data_tree_response.status_code != 200:
            return pd.DataFrame()
        
        data_tree_data = data_tree_response.json()
        latest_folder = None
        for item in data_tree_data.get('tree', []):
            if item['path'] == 'latest' and item['type'] == 'tree':
                latest_folder = item['sha']
                break
        
        if not latest_folder:
            return pd.DataFrame()
        
        latest_tree_url = f"https://api.github.com/repos/corner-25/vehicle-storage/git/trees/{latest_folder}"
        latest_tree_response = requests.get(latest_tree_url, headers=headers, timeout=30)
        
        if latest_tree_response.status_code != 200:
            return pd.DataFrame()
        
        latest_tree_data = latest_tree_response.json()
        file_blob = None
        for item in latest_tree_data.get('tree', []):
            if item['path'] == 'fleet_data_latest.json' and item['type'] == 'blob':
                file_blob = item['sha']
                break
        
        if not file_blob:
            return pd.DataFrame()
        
        blob_url = f"https://api.github.com/repos/corner-25/vehicle-storage/git/blobs/{file_blob}"
        blob_response = requests.get(blob_url, headers=headers, timeout=60)
        
        if blob_response.status_code != 200:
            return pd.DataFrame()
        
        blob_data = blob_response.json()
        content = base64.b64decode(blob_data['content']).decode('utf-8')
        
        if not content.strip():
            return pd.DataFrame()
        
        data = json.loads(content)
        df = pd.DataFrame(data)
        return process_dataframe(df)
        
    except Exception:
        return pd.DataFrame()

def process_dataframe(df):
    """🔧 MAIN DATA PROCESSOR - Apply column mapping and clean data"""
    if df.empty:
        return df
    
    try:
        st.sidebar.info(f"📥 Raw data: {len(df)} records, {len(df.columns)} columns")
        
        # Step 1: Apply column mapping
        reverse_mapping = {}
        for viet_col, eng_col in COLUMN_MAPPING.items():
            if eng_col is not None:
                for col in df.columns:
                    if viet_col in col:
                        reverse_mapping[col] = eng_col
                        break
        
        df = df.rename(columns=reverse_mapping)
        
        # Step 2: Drop unnecessary columns
        drop_columns = []
        for viet_col in COLUMN_MAPPING.keys():
            if COLUMN_MAPPING[viet_col] is None:
                for col in df.columns:
                    if viet_col in col:
                        drop_columns.append(col)
        
        df = df.drop(columns=drop_columns, errors='ignore')
        df = df.loc[:, ~df.columns.duplicated()]
        
        # 🔧 Step 3: CRITICAL - Process duration with enhanced validation
        if 'duration_hours' in df.columns:
            st.sidebar.info("🔧 Processing duration column...")
            df = validate_duration_column(df)
        
        # Step 4: Process other columns
        if 'distance_km' in df.columns:
            df['distance_km'] = df['distance_km'].apply(parse_distance)
        
        if 'revenue_vnd' in df.columns:
            df['revenue_vnd'] = df['revenue_vnd'].apply(parse_revenue)
        
        if 'fuel_liters' in df.columns:
            df['fuel_liters'] = pd.to_numeric(df['fuel_liters'], errors='coerce').fillna(0)
        
        if 'record_date' in df.columns:
            df['record_date'] = pd.to_datetime(df['record_date'], errors='coerce')
            df['date'] = df['record_date'].dt.date
            df['month'] = df['record_date'].dt.to_period('M').astype(str)
        
        st.sidebar.success(f"✅ Processed: {len(df)} records, {len(df.columns)} clean columns")
        return df
        
    except Exception as e:
        st.sidebar.error(f"❌ Error processing data: {e}")
        return df

def run_sync_script():
    """Execute sync script"""
    try:
        if not os.path.exists("manual_fleet_sync.py"):
            st.error("❌ Không tìm thấy file manual_fleet_sync.py")
            return False
        
        token = get_github_token()
        if not token:
            st.error("❌ Không tìm thấy GitHub token!")
            return False
        
        with st.spinner("🔄 Đang chạy sync script..."):
            try:
                if 'manual_fleet_sync' in sys.modules:
                    del sys.modules['manual_fleet_sync']
                
                import manual_fleet_sync
                sync_engine = manual_fleet_sync.ManualFleetSync()
                
                if sync_engine.config['github']['token'] == "YOUR_TOKEN_HERE":
                    st.error("❌ GitHub token chưa được load!")
                    return False
                
                success = sync_engine.sync_now()
                
                if success:
                    st.success("✅ Sync hoàn thành!")
                    st.session_state.last_sync = datetime.now()
                    return True
                else:
                    st.error("❌ Sync thất bại!")
                    return False
                    
            except Exception:
                result = subprocess.run([
                    sys.executable, "manual_fleet_sync.py", "--sync-only"
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    st.success("✅ Sync hoàn thành!")
                    st.session_state.last_sync = datetime.now()
                    return True
                else:
                    st.error(f"❌ Sync thất bại: {result.stderr}")
                    return False
                    
    except Exception as e:
        st.error(f"❌ Lỗi chạy sync: {e}")
        return False

def filter_data_by_date_range(df, start_date, end_date):
    """Filter dataframe by date range"""
    if df.empty or 'record_date' not in df.columns:
        return df
    
    try:
        df['record_date'] = pd.to_datetime(df['record_date'], format='%m/%d/%Y', errors='coerce')
        
        invalid_count = df['record_date'].isna().sum()
        if invalid_count > 0:
            st.sidebar.warning(f"⚠️ Found {invalid_count} records with invalid dates")
        
        valid_mask = (df['record_date'].notna()) & (df['record_date'].dt.date >= start_date) & (df['record_date'].dt.date <= end_date)
        invalid_mask = df['record_date'].isna()
        
        combined_mask = valid_mask | invalid_mask
        filtered_df = df[combined_mask].copy()
        
        return filtered_df
        
    except Exception as e:
        st.sidebar.error(f"❌ Lỗi lọc dữ liệu: {e}")
        return df

def get_date_range_from_data(df):
    """Get min and max dates from data"""
    if df.empty or 'record_date' not in df.columns:
        return datetime.now().date(), datetime.now().date()
    
    try:
        df['record_date'] = pd.to_datetime(df['record_date'], format='%m/%d/%Y', errors='coerce')
        valid_dates = df[df['record_date'].notna()]
        
        if valid_dates.empty:
            return datetime.now().date(), datetime.now().date()
        
        min_date = valid_dates['record_date'].min().date()
        max_date = valid_dates['record_date'].max().date()
        
        return min_date, max_date
        
    except Exception:
        return datetime.now().date(), datetime.now().date()

def create_date_filter_sidebar(df):
    """Create date range filter in sidebar"""
    st.sidebar.markdown("### 📅 Bộ lọc thời gian")
    
    min_date, max_date = get_date_range_from_data(df)
    
    st.sidebar.info(f"📊 Dữ liệu có: {min_date.strftime('%d/%m/%Y')} - {max_date.strftime('%d/%m/%Y')}")
    
    reset_needed = False
    if 'date_filter_start' in st.session_state:
        if st.session_state.date_filter_start < min_date or st.session_state.date_filter_start > max_date:
            reset_needed = True
    if 'date_filter_end' in st.session_state:
        if st.session_state.date_filter_end < min_date or st.session_state.date_filter_end > max_date:
            reset_needed = True
    
    if reset_needed:
        st.sidebar.warning("⚠️ Đã reset bộ lọc ngày do dữ liệu thay đổi")
        if 'date_filter_start' in st.session_state:
            del st.session_state.date_filter_start
        if 'date_filter_end' in st.session_state:
            del st.session_state.date_filter_end
    
    if 'date_filter_start' not in st.session_state:
        st.session_state.date_filter_start = min_date
    if 'date_filter_end' not in st.session_state:
        st.session_state.date_filter_end = max_date
    
    if st.session_state.date_filter_start < min_date:
        st.session_state.date_filter_start = min_date
    if st.session_state.date_filter_start > max_date:
        st.session_state.date_filter_start = max_date
    if st.session_state.date_filter_end < min_date:
        st.session_state.date_filter_end = min_date
    if st.session_state.date_filter_end > max_date:
        st.session_state.date_filter_end = max_date
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Từ ngày:",
            value=st.session_state.date_filter_start,
            min_value=min_date,
            max_value=max_date,
            key="start_date_input"
        )
    
    with col2:
        end_date = st.date_input(
            "Đến ngày:",
            value=st.session_state.date_filter_end,
            min_value=min_date,
            max_value=max_date,
            key="end_date_input"
        )
    
    if start_date != st.session_state.date_filter_start:
        st.session_state.date_filter_start = start_date
    if end_date != st.session_state.date_filter_end:
        st.session_state.date_filter_end = end_date
    
    if start_date > end_date:
        st.sidebar.error("❌ Ngày bắt đầu phải nhỏ hơn ngày kết thúc!")
        return df, min_date, max_date
    
    st.sidebar.markdown("**🚀 Bộ lọc nhanh:**")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("📅 7 ngày gần nhất", use_container_width=True, key="btn_7_days"):
            st.session_state.date_filter_start = max_date - pd.Timedelta(days=6)
            st.session_state.date_filter_end = max_date
            st.rerun()
        
        if st.button("📅 Tháng này", use_container_width=True, key="btn_this_month"):
            today = datetime.now().date()
            st.session_state.date_filter_start = today.replace(day=1)
            st.session_state.date_filter_end = min(today, max_date)
            st.rerun()
    
    with col2:
        if st.button("📅 30 ngày gần nhất", use_container_width=True, key="btn_30_days"):
            st.session_state.date_filter_start = max_date - pd.Timedelta(days=29)
            st.session_state.date_filter_end = max_date
            st.rerun()
        
        if st.button("📅 Tất cả", use_container_width=True, key="btn_all_data"):
            st.session_state.date_filter_start = min_date
            st.session_state.date_filter_end = max_date
            st.rerun()
    
    filter_start = st.session_state.date_filter_start
    filter_end = st.session_state.date_filter_end
    
    filtered_df = filter_data_by_date_range(df, filter_start, filter_end)
    
    if not filtered_df.empty:
        days_selected = (filter_end - filter_start).days + 1
        active_days = filtered_df['record_date'].dt.date.nunique() if 'record_date' in filtered_df.columns else 0
        
        st.sidebar.success(f"✅ Đã chọn: {days_selected} ngày")
        st.sidebar.info(f"📊 {len(filtered_df):,} chuyến từ {active_days} ngày hoạt động")
        
        if len(filtered_df) == 0:
            st.sidebar.warning("⚠️ Không có dữ liệu trong khoảng thời gian này")
    
    return filtered_df, filter_start, filter_end

def create_vehicle_filter_sidebar(df):
    """Create vehicle and driver filters in sidebar"""
    st.sidebar.markdown("### 🚗 Bộ lọc xe và tài xế")
    
    if df.empty:
        return df
    
    if 'vehicle_type' in df.columns:
        vehicle_types = ['Tất cả'] + list(df['vehicle_type'].unique())
        selected_type = st.sidebar.selectbox(
            "Loại xe:",
            options=vehicle_types,
            index=0
        )
        
        if selected_type != 'Tất cả':
            df = df[df['vehicle_type'] == selected_type]
    
    if 'vehicle_id' in df.columns:
        vehicle_ids = list(df['vehicle_id'].unique())
        selected_vehicles = st.sidebar.multiselect(
            "Chọn xe (để trống = tất cả):",
            options=vehicle_ids,
            default=[]
        )
        
        if selected_vehicles:
            df = df[df['vehicle_id'].isin(selected_vehicles)]
    
    if 'driver_name' in df.columns:
        drivers = list(df['driver_name'].unique())
        selected_drivers = st.sidebar.multiselect(
            "Chọn tài xế (để trống = tất cả):",
            options=drivers,
            default=[]
        )
        
        if selected_drivers:
            df = df[df['driver_name'].isin(selected_drivers)]
    
    if 'work_category' in df.columns:
        work_categories = ['Tất cả'] + list(df['work_category'].dropna().unique())
        selected_category = st.sidebar.selectbox(
            "Phân loại công tác:",
            options=work_categories,
            index=0
        )
        
        if selected_category != 'Tất cả':
            df = df[df['work_category'] == selected_category]
    
    if 'area_type' in df.columns:
        area_types = ['Tất cả'] + list(df['area_type'].dropna().unique())
        selected_area = st.sidebar.selectbox(
            "Khu vực:",
            options=area_types,
            index=0
        )
        
        if selected_area != 'Tất cả':
            df = df[df['area_type'] == selected_area]
    
    return df

def create_metrics_overview(df):
    """🔧 ENHANCED METRICS - Create overview metrics with robust time calculation"""
    if df.empty:
        st.warning("⚠️ Không có dữ liệu để hiển thị")
        return
    
    st.markdown("## 📊 Tổng quan hoạt động")
    
    # Ensure duration is validated again before metrics calculation
    if 'duration_hours' in df.columns:
        df = validate_duration_column(df)
    
    total_trips = len(df)
    total_vehicles = df['vehicle_id'].nunique() if 'vehicle_id' in df.columns else 0
    total_drivers = df['driver_name'].nunique() if 'driver_name' in df.columns else 0
    
    # Revenue calculation
    if 'revenue_vnd' in df.columns:
        df['revenue_vnd'] = pd.to_numeric(df['revenue_vnd'], errors='coerce').fillna(0)
        total_revenue = df['revenue_vnd'].sum()
        revenue_records = df[df['revenue_vnd'] > 0]
        avg_revenue_per_trip = revenue_records['revenue_vnd'].mean() if len(revenue_records) > 0 else 0
    else:
        total_revenue = 0
        avg_revenue_per_trip = 0
    
    # 🔧 ENHANCED TIME CALCULATION
    if 'duration_hours' in df.columns:
        # More strict validation for metric calculation
        valid_time_data = df[
            df['duration_hours'].notna() & 
            (df['duration_hours'] > 0) & 
            (df['duration_hours'] <= 24)
        ]
        total_hours = valid_time_data['duration_hours'].sum()
        avg_hours_per_trip = valid_time_data['duration_hours'].mean() if len(valid_time_data) > 0 else 0
        
        # Log time calculation for debugging
        st.sidebar.info(f"⏱️ Valid time records: {len(valid_time_data)}/{len(df)}")
    else:
        total_hours = 0
        avg_hours_per_trip = 0
    
    # Distance calculation
    if 'distance_km' in df.columns:
        df['distance_km'] = df['distance_km'].apply(parse_distance)
        valid_distance_data = df[df['distance_km'].notna() & (df['distance_km'] >= 0)]
        total_distance = valid_distance_data['distance_km'].sum()
        avg_distance = valid_distance_data['distance_km'].mean() if len(valid_distance_data) > 0 else 0
    else:
        total_distance = 0
        avg_distance = 0
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🚗 Tổng chuyến",
            value=f"{total_trips:,}",
            help="Tổng số chuyến đã thực hiện"
        )
    
    with col2:
        st.metric(
            label="🏥 Số xe hoạt động", 
            value=f"{total_vehicles}",
            help="Số xe đang hoạt động"
        )
    
    with col3:
        st.metric(
            label="👨‍💼 Số tài xế",
            value=f"{total_drivers}",
            help="Số tài xế đang làm việc"
        )
    
    with col4:
        st.metric(
            label="💰 Tổng doanh thu",
            value=f"{total_revenue:,.0f} VNĐ",
            help="Tổng doanh thu từ xe cứu thương"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric(
            label="⏱️ Tổng giờ chạy",
            value=f"{total_hours:,.1f} giờ",
            help="Tổng thời gian vận hành"
        )
    
    with col6:
        st.metric(
            label="🛣️ Tổng quãng đường",
            value=f"{total_distance:,.1f} km",
            help="Tổng quãng đường đã di chuyển"
        )
    
    with col7:
        st.metric(
            label="💵 TB doanh thu/chuyến",
            value=f"{avg_revenue_per_trip:,.0f} VNĐ",
            help="Doanh thu trung bình mỗi chuyến (xe cứu thương)"
        )
    
    with col8:
        st.metric(
            label="⏰ TB giờ/chuyến", 
            value=f"{avg_hours_per_trip:.1f} giờ",
            help="Thời gian trung bình mỗi chuyến"
        )

def create_frequency_metrics(df):
    """Create frequency and activity metrics"""
    st.markdown("## 🎯 Chỉ số tần suất hoạt động")
    
    if df.empty or 'record_date' not in df.columns:
        st.warning("⚠️ Không có dữ liệu thời gian")
        return
    
    try:
        df['record_date'] = pd.to_datetime(df['record_date'], format='%m/%d/%Y', errors='coerce')
        df['date'] = df['record_date'].dt.date
        
        valid_dates = df[df['record_date'].notna()]
        invalid_count = df['record_date'].isna().sum()
        
        if invalid_count > 0:
            st.sidebar.info(f"ℹ️ {invalid_count} records có ngày không hợp lệ")
        
        if valid_dates.empty:
            st.warning("⚠️ Không có dữ liệu ngày hợp lệ")
            return
        
        active_days = valid_dates['date'].nunique()
        total_date_range = (valid_dates['record_date'].max() - valid_dates['record_date'].min()).days + 1
        
        daily_trips = valid_dates.groupby('date')['vehicle_id'].count()
        
        total_vehicles = df['vehicle_id'].nunique() if 'vehicle_id' in df.columns else 1
        daily_active_vehicles = valid_dates.groupby('date')['vehicle_id'].nunique()
        
        st.sidebar.markdown("### 📅 Date Analysis")
        st.sidebar.info(f"📊 Từ: {valid_dates['record_date'].min().strftime('%d/%m/%Y')}")
        st.sidebar.info(f"📊 Đến: {valid_dates['record_date'].max().strftime('%d/%m/%Y')}")
        st.sidebar.info(f"📊 Tổng khoảng: {total_date_range} ngày")
        st.sidebar.info(f"📊 Ngày có hoạt động: {active_days} ngày")
        
    except Exception as e:
        st.error(f"❌ Lỗi xử lý ngày tháng: {e}")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_trips_per_day = len(valid_dates) / active_days if active_days > 0 else 0
        st.metric(
            label="📈 Chuyến TB/ngày",
            value=f"{avg_trips_per_day:.1f}",
            help=f"Số chuyến trung bình mỗi ngày hoạt động ({active_days} ngày có chuyến)"
        )
    
    with col2:
        avg_utilization = (daily_active_vehicles.mean() / total_vehicles * 100) if total_vehicles > 0 else 0
        st.metric(
            label="🚗 Tỷ lệ sử dụng xe TB",
            value=f"{avg_utilization:.1f}%",
            help=f"Tỷ lệ xe hoạt động trung bình ({total_vehicles} xe tổng)"
        )
    
    with col3:
        peak_day_trips = daily_trips.max() if not daily_trips.empty else 0
        peak_date = daily_trips.idxmax() if not daily_trips.empty else None
        st.metric(
            label="⬆️ Ngày cao điểm",
            value=f"{peak_day_trips} chuyến",
            help=f"Ngày có nhiều chuyến nhất: {peak_date}" if peak_date else "Ngày có nhiều chuyến nhất"
        )
    
    with col4:
        low_day_trips = daily_trips.min() if not daily_trips.empty else 0
        low_date = daily_trips.idxmin() if not daily_trips.empty else None
        st.metric(
            label="⬇️ Ngày thấp điểm",
            value=f"{low_day_trips} chuyến",
            help=f"Ngày có ít chuyến nhất: {low_date}" if low_date else "Ngày có ít chuyến nhất"
        )

def create_vehicle_performance_table(df):
    """🔧 ENHANCED VEHICLE TABLE - Create detailed vehicle performance table"""
    st.markdown("## 📋 Hiệu suất chi tiết từng xe")
    
    if df.empty or 'vehicle_id' not in df.columns:
        st.warning("⚠️ Không có dữ liệu xe")
        return
    
    # Ensure duration is validated
    if 'duration_hours' in df.columns:
        df = validate_duration_column(df)
    
    try:
        if 'record_date' in df.columns:
            df['record_date'] = pd.to_datetime(df['record_date'], format='%m/%d/%Y', errors='coerce')
            df['date'] = df['record_date'].dt.date
            
            valid_dates = df[df['record_date'].notna()]
            if not valid_dates.empty:
                total_days = (valid_dates['record_date'].max() - valid_dates['record_date'].min()).days + 1
            else:
                total_days = 30
        else:
            total_days = 30
    except:
        total_days = 30
    
    # Ensure numeric columns
    if 'revenue_vnd' in df.columns:
        df['revenue_vnd'] = pd.to_numeric(df['revenue_vnd'], errors='coerce').fillna(0)
    else:
        df['revenue_vnd'] = 0
        
    if 'duration_hours' not in df.columns:
        df['duration_hours'] = 0
        
    if 'distance_km' in df.columns:
        df['distance_km'] = df['distance_km'].apply(parse_distance)
    else:
        df['distance_km'] = 0
        
    if 'fuel_liters' in df.columns:
        df['fuel_liters'] = pd.to_numeric(df['fuel_liters'], errors='coerce').fillna(0)
    else:
        df['fuel_liters'] = 0
    
    # Calculate metrics per vehicle
    vehicles = df['vehicle_id'].unique()
    results = []
    
    for vehicle in vehicles:
        vehicle_data = df[df['vehicle_id'] == vehicle]
        
        total_trips = len(vehicle_data)
        total_revenue = float(vehicle_data['revenue_vnd'].sum())
        avg_revenue = float(vehicle_data['revenue_vnd'].mean()) if total_trips > 0 else 0.0
        
        # 🔧 ENHANCED DURATION CALCULATION
        valid_duration_data = vehicle_data[
            vehicle_data['duration_hours'].notna() & 
            (vehicle_data['duration_hours'] > 0) & 
            (vehicle_data['duration_hours'] <= 24)
        ]
        total_hours = float(valid_duration_data['duration_hours'].sum())
        
        total_distance = float(vehicle_data['distance_km'].sum())
        total_fuel = float(vehicle_data['fuel_liters'].sum())
        
        if 'date' in vehicle_data.columns:
            active_days = vehicle_data['date'].nunique()
        else:
            active_days = total_days
        
        # Derived metrics
        fuel_per_100km = (total_fuel / total_distance * 100.0) if total_distance > 0 else 0.0
        trips_per_day = (float(total_trips) / float(active_days)) if active_days > 0 else 0.0
        utilization = (float(active_days) / float(total_days) * 100.0) if total_days > 0 else 0.0
        
        # Performance rating
        if trips_per_day >= 2 and utilization >= 70:
            performance = 'Cao'
        elif trips_per_day >= 1 and utilization >= 50:
            performance = 'Trung bình'
        else:
            performance = 'Thấp'
        
        results.append({
            'Mã xe': vehicle,
            'Tổng chuyến': total_trips,
            'Tổng doanh thu': round(total_revenue, 0),
            'Doanh thu TB/chuyến': round(avg_revenue, 0),
            'Tổng giờ chạy': round(total_hours, 1),
            'Số ngày hoạt động': active_days,
            'Tổng quãng đường': round(total_distance, 1),
            'Nhiên liệu tiêu thụ': round(total_fuel, 1),
            'Nhiên liệu/100km': round(fuel_per_100km, 2),
            'Chuyến/ngày': round(trips_per_day, 1),
            'Tỷ lệ sử dụng (%)': round(utilization, 1),
            'Hiệu suất': performance
        })
    
    vehicle_display = pd.DataFrame(results)
    vehicle_display = vehicle_display.set_index('Mã xe').sort_values('Tổng doanh thu', ascending=False)
    
    st.dataframe(
        vehicle_display.style.format({
            'Tổng doanh thu': '{:,.0f}',
            'Doanh thu TB/chuyến': '{:,.0f}',
            'Tổng giờ chạy': '{:.1f}',
            'Tổng quãng đường': '{:.1f}',
            'Nhiên liệu tiêu thụ': '{:.1f}',
            'Nhiên liệu/100km': '{:.2f}',
            'Chuyến/ngày': '{:.1f}',
            'Tỷ lệ sử dụng (%)': '{:.1f}'
        }),
        use_container_width=True,
        height=400
    )

def create_driver_performance_table(df):
    """🔧 ENHANCED DRIVER TABLE - Create driver performance table"""
    st.markdown("## 👨‍💼 Hiệu suất tài xế")
    
    if df.empty or 'driver_name' not in df.columns:
        st.warning("⚠️ Không có dữ liệu tài xế")
        return
    
    # Ensure duration is validated
    if 'duration_hours' in df.columns:
        df = validate_duration_column(df)
    
    try:
        if 'record_date' in df.columns:
            df['record_date'] = pd.to_datetime(df['record_date'], format='%m/%d/%Y', errors='coerce')
            df['date'] = df['record_date'].dt.date
    except:
        pass
    
    if 'revenue_vnd' in df.columns:
        df['revenue_vnd'] = pd.to_numeric(df['revenue_vnd'], errors='coerce').fillna(0)
    else:
        df['revenue_vnd'] = 0
    
    # Calculate metrics per driver
    drivers = df['driver_name'].unique()
    results = []
    
    for driver in drivers:
        driver_data = df[df['driver_name'] == driver]
        
        total_trips = len(driver_data)
        total_revenue = float(driver_data['revenue_vnd'].sum())
        
        # 🔧 ENHANCED DURATION CALCULATION FOR DRIVERS
        valid_duration_data = driver_data[
            driver_data['duration_hours'].notna() & 
            (driver_data['duration_hours'] > 0) & 
            (driver_data['duration_hours'] <= 24)
        ]
        total_hours = float(valid_duration_data['duration_hours'].sum())
        
        if 'date' in driver_data.columns:
            active_days = driver_data['date'].nunique()
        else:
            active_days = 30
        
        trips_per_day = (float(total_trips) / float(active_days)) if active_days > 0 else 0.0
        hours_per_day = (total_hours / float(active_days)) if active_days > 0 else 0.0
        
        results.append({
            'Tên': driver,
            'Số chuyến': total_trips,
            'Tổng doanh thu': round(total_revenue, 0),
            'Tổng giờ lái': round(total_hours, 1),
            'Số ngày làm việc': active_days,
            'Chuyến/ngày': round(trips_per_day, 1),
            'Giờ lái/ngày': round(hours_per_day, 1)
        })
    
    driver_display = pd.DataFrame(results)
    driver_display = driver_display.set_index('Tên').sort_values('Tổng doanh thu', ascending=False)
    
    st.dataframe(
        driver_display.style.format({
            'Tổng doanh thu': '{:,.0f}',
            'Tổng giờ lái': '{:.1f}',
            'Chuyến/ngày': '{:.1f}',
            'Giờ lái/ngày': '{:.1f}'
        }),
        use_container_width=True,
        height=400
    )

def create_revenue_analysis_tab(df):
    """Tab 1: Phân tích doanh thu"""
    st.markdown("### 💰 Phân tích doanh thu chi tiết")
    
    if df.empty or 'revenue_vnd' not in df.columns:
        st.warning("⚠️ Không có dữ liệu doanh thu")
        return
    
    df['revenue_vnd'] = pd.to_numeric(df['revenue_vnd'], errors='coerce').fillna(0)
    revenue_data = df[df['revenue_vnd'] > 0].copy()
    
    if revenue_data.empty:
        st.warning("⚠️ Không có chuyến xe có doanh thu")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Doanh thu theo xe")
        vehicle_revenue = revenue_data.groupby('vehicle_id')['revenue_vnd'].agg(['sum', 'count', 'mean']).reset_index()
        vehicle_revenue.columns = ['vehicle_id', 'total_revenue', 'trip_count', 'avg_revenue']
        vehicle_revenue = vehicle_revenue.sort_values('total_revenue', ascending=False)
        
        fig_vehicle = px.bar(
            vehicle_revenue.head(10),
            x='vehicle_id',
            y='total_revenue',
            title="Top 10 xe có doanh thu cao nhất",
            labels={'total_revenue': 'Doanh thu (VNĐ)', 'vehicle_id': 'Mã xe'},
            color='total_revenue',
            color_continuous_scale='Blues'
        )
        fig_vehicle.update_layout(height=400)
        st.plotly_chart(fig_vehicle, use_container_width=True)
    
    with col2:
        st.markdown("#### 📈 Doanh thu theo thời gian")
        if 'record_date' in revenue_data.columns:
            daily_revenue = revenue_data.groupby('date')['revenue_vnd'].sum().reset_index()
            daily_revenue = daily_revenue.sort_values('date')
            
            fig_time = px.line(
                daily_revenue,
                x='date',
                y='revenue_vnd',
                title="Xu hướng doanh thu theo ngày",
                labels={'revenue_vnd': 'Doanh thu (VNĐ)', 'date': 'Ngày'}
            )
            fig_time.update_layout(height=400)
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("Không có dữ liệu thời gian để hiển thị xu hướng")

def create_vehicle_efficiency_tab(df):
    """Tab 2: Hiệu suất xe"""
    st.markdown("### 🚗 Phân tích hiệu suất xe")
    
    if df.empty or 'vehicle_id' not in df.columns:
        st.warning("⚠️ Không có dữ liệu xe")
        return
    
    # Ensure duration is validated for efficiency calculation
    if 'duration_hours' in df.columns:
        df = validate_duration_column(df)
    
    vehicle_stats = []
    for vehicle in df['vehicle_id'].unique():
        vehicle_data = df[df['vehicle_id'] == vehicle]
        
        total_trips = len(vehicle_data)
        total_hours = vehicle_data['duration_hours'].sum() if 'duration_hours' in vehicle_data.columns else 0
        total_distance = vehicle_data['distance_km'].sum() if 'distance_km' in vehicle_data.columns else 0
        total_revenue = vehicle_data['revenue_vnd'].sum() if 'revenue_vnd' in vehicle_data.columns else 0
        
        active_days = vehicle_data['date'].nunique() if 'date' in vehicle_data.columns else 1
        
        trips_per_day = total_trips / active_days if active_days > 0 else 0
        hours_per_trip = total_hours / total_trips if total_trips > 0 else 0
        distance_per_trip = total_distance / total_trips if total_trips > 0 else 0
        revenue_per_hour = total_revenue / total_hours if total_hours > 0 else 0
        
        vehicle_stats.append({
            'vehicle_id': vehicle,
            'total_trips': total_trips,
            'active_days': active_days,
            'trips_per_day': trips_per_day,
            'hours_per_trip': hours_per_trip,
            'distance_per_trip': distance_per_trip,
            'revenue_per_hour': revenue_per_hour,
            'total_hours': total_hours,
            'total_distance': total_distance,
            'total_revenue': total_revenue
        })
    
    efficiency_df = pd.DataFrame(vehicle_stats)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Chuyến/ngày theo xe")
        fig_trips = px.bar(
            efficiency_df.sort_values('trips_per_day', ascending=False).head(15),
            x='vehicle_id',
            y='trips_per_day',
            title="Số chuyến trung bình mỗi ngày",
            labels={'trips_per_day': 'Chuyến/ngày', 'vehicle_id': 'Mã xe'},
            color='trips_per_day',
            color_continuous_scale='Greens'
        )
        fig_trips.update_layout(height=400)
        st.plotly_chart(fig_trips, use_container_width=True)
    
    with col2:
        st.markdown("#### ⏱️ Thời gian trung bình mỗi chuyến")
        fig_hours = px.bar(
            efficiency_df.sort_values('hours_per_trip', ascending=False).head(15),
            x='vehicle_id',
            y='hours_per_trip',
            title="Giờ trung bình mỗi chuyến",
            labels={'hours_per_trip': 'Giờ/chuyến', 'vehicle_id': 'Mã xe'},
            color='hours_per_trip',
            color_continuous_scale='Oranges'
        )
        fig_hours.update_layout(height=400)
        st.plotly_chart(fig_hours, use_container_width=True)

def create_detailed_analysis_section(df):
    """Create detailed analysis section with tabs"""
    st.markdown("---")
    st.markdown("## 📈 Phân tích chi tiết và Biểu đồ trực quan")
    
    if df.empty:
        st.warning("⚠️ Không có dữ liệu để phân tích")
        return
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        st.error("❌ Cần cài đặt plotly: pip install plotly")
        return
    
    tab1, tab2 = st.tabs([
        "💰 Doanh thu", 
        "🚗 Hiệu suất xe"
    ])
    
    with tab1:
        create_revenue_analysis_tab(df)
    
    with tab2:
        create_vehicle_efficiency_tab(df)

def main():
    """🔧 ENHANCED MAIN FUNCTION - Complete dashboard with fixed time calculation"""
    
    # 🔧 CRITICAL: Clear cache and force fresh start
    st.cache_data.clear()
    
    # Header
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logo_base64 = ""
        for p in [
            os.path.join(script_dir, "logo.png"),
            os.path.join(script_dir, "assets", "logo.png")
        ]:
            if os.path.exists(p):
                with open(p, "rb") as f:
                    logo_base64 = base64.b64encode(f.read()).decode()
                break
    except Exception:
        logo_base64 = ""

    if logo_base64:
        logo_html = f"<img src='data:image/png;base64,{logo_base64}' style='height:150px; width:auto;' />"
    else:
        logo_html = "<div style='font-size:2.5rem; margin-right:12px;'>🏥</div>"

    header_html = f"""
    <div style='
        width:100%;
        display:flex;
        align-items:center;
        justify-content:center;
        gap:12px;
        padding:30px 0;
        background:#ffffff;
        border-radius:15px;
        margin-bottom:30px;
    '>
        {logo_html}
        <h1 style='
            color:#1f77b4;
            margin:0;
            font-size:2.7rem;
            font-weight:bold;
            font-family:"Segoe UI", Arial, sans-serif;
            text-shadow:2px 2px 4px rgba(0,0,0,0.1);
            letter-spacing:1px;
            text-align:center;
        '>Dashboard Quản lý Phương tiện vận chuyển tại Bệnh viện Đại học Y Dược TP. Hồ Chí Minh</h1>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("📊 Đang tải dữ liệu từ GitHub..."):
        df_raw = load_data_from_github()
    
    if df_raw.empty:
        st.warning("⚠️ Không có dữ liệu từ GitHub repository")
        st.info("💡 Click 'Sync dữ liệu mới' để lấy dữ liệu từ Google Sheets")
        return
    
    # Sidebar controls
    st.sidebar.markdown("## 🔧 Điều khiển Dashboard")
    
    # Debug toggle
    debug_mode = st.sidebar.checkbox("🔧 Debug Mode", help="Hiển thị thông tin debug")
    
    if debug_mode:
        st.sidebar.markdown("### 🔍 Debug Info")
        st.sidebar.info(f"Python: {sys.version_info.major}.{sys.version_info.minor}")
        st.sidebar.info(f"Pandas: {pd.__version__}")
        st.sidebar.info(f"Numpy: {np.__version__}")
        
        if 'duration_hours' in df_raw.columns:
            sample_duration = df_raw['duration_hours'].head(3).tolist()
            st.sidebar.write(f"Sample durations: {sample_duration}")
    
    # Column mapping info
    with st.sidebar.expander("📋 Column Mapping Guide"):
        st.write("**Vietnamese → English:**")
        for viet, eng in COLUMN_MAPPING.items():
            if eng is not None:
                st.write(f"• {viet} → `{eng}`")
    
    # Sync button
    if st.sidebar.button("🔄 Sync dữ liệu mới", type="primary", use_container_width=True):
        success = run_sync_script()
        if success:
            st.cache_data.clear()
            st.rerun()
    
    # Last sync info
    if 'last_sync' in st.session_state:
        st.sidebar.success(f"🕐 Sync cuối: {st.session_state.last_sync.strftime('%H:%M:%S %d/%m/%Y')}")
    
    # Manual refresh
    if st.sidebar.button("🔄 Làm mới Dashboard", help="Reload dữ liệu từ GitHub"):
        if 'date_filter_start' in st.session_state:
            del st.session_state.date_filter_start
        if 'date_filter_end' in st.session_state:
            del st.session_state.date_filter_end
        st.cache_data.clear()
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Apply filters
    df_filtered, start_date, end_date = create_date_filter_sidebar(df_raw)
    
    st.sidebar.markdown("---")
    
    df_final = create_vehicle_filter_sidebar(df_filtered)
    
    # Show filter results
    st.sidebar.markdown("### 📊 Kết quả lọc")
    if not df_final.empty:
        vehicles_count = df_final['vehicle_id'].nunique() if 'vehicle_id' in df_final.columns else 0
        drivers_count = df_final['driver_name'].nunique() if 'driver_name' in df_final.columns else 0
        
        st.sidebar.metric("📈 Tổng chuyến", f"{len(df_final):,}")
        st.sidebar.metric("🚗 Số xe", f"{vehicles_count}")
        st.sidebar.metric("👨‍💼 Số tài xế", f"{drivers_count}")
        
        percentage = (len(df_final) / len(df_raw) * 100) if len(df_raw) > 0 else 0
        st.sidebar.info(f"📊 {percentage:.1f}% tổng dữ liệu")
    else:
        st.sidebar.error("❌ Không có dữ liệu sau khi lọc")
        st.warning("⚠️ Không có dữ liệu phù hợp với bộ lọc hiện tại")
        return
    
    # Show available columns after filtering
    with st.sidebar.expander("📋 Mapped Columns"):
        for col in df_final.columns:
            non_null_count = df_final[col].notna().sum()
            st.write(f"• `{col}`: {non_null_count}/{len(df_final)}")
    
    # Reset filters button
    if st.sidebar.button("🔄 Reset tất cả bộ lọc"):
        if 'date_filter_start' in st.session_state:
            del st.session_state.date_filter_start
        if 'date_filter_end' in st.session_state:
            del st.session_state.date_filter_end
        st.sidebar.success("✅ Đã reset bộ lọc ngày!")
        st.rerun()
    
    # Dashboard sections
    st.markdown(f"## 📊 Báo cáo từ {start_date.strftime('%d/%m/%Y')} đến {end_date.strftime('%d/%m/%Y')}")
    
    create_metrics_overview(df_final)
    
    st.markdown("---")
    
    create_frequency_metrics(df_final)
    
    st.markdown("---")
    
    create_vehicle_performance_table(df_final)
    
    st.markdown("---")
    
    create_driver_performance_table(df_final)
    
    create_detailed_analysis_section(df_final)
    
    # Debug section
    if debug_mode:
        with st.expander("🔍 Advanced Debug Info"):
            st.write("**Sample Filtered Data:**")
            if not df_final.empty:
                st.dataframe(df_final.head(3))
            
            st.write("**Column Data Types:**")
            for col in df_final.columns:
                st.write(f"• `{col}`: {df_final[col].dtype}")
            
            if 'duration_hours' in df_final.columns:
                st.write("**Duration Column Analysis:**")
                st.write(f"• Total records: {len(df_final)}")
                st.write(f"• Non-null duration: {df_final['duration_hours'].notna().sum()}")
                st.write(f"• Zero duration: {(df_final['duration_hours'] == 0).sum()}")
                st.write(f"• Positive duration: {(df_final['duration_hours'] > 0).sum()}")
                st.write(f"• Total hours: {df_final['duration_hours'].sum():.1f}")
                st.write(f"• Sample values: {df_final['duration_hours'].head(5).tolist()}")

if __name__ == "__main__":
    main()
