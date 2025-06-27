#!/usr/bin/env python3
"""
Manual Fleet Data Sync Engine - IMPROVED VERSION
Fixed: Better error handling, debug logging, và không bỏ sót xe
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
from dotenv import load_dotenv
import base64
from typing import Dict, List, Optional
import requests
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import time

# Load environment variables
load_dotenv()

# Setup logging với level DEBUG để theo dõi chi tiết
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fleet_sync_detailed.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ImprovedFleetSync:
    """
    Improved Fleet Data Sync Engine
    - Better error handling
    - Detailed logging
    - Retry mechanism
    - No data loss
    """
    
    def __init__(self):
        """Khởi tạo sync engine"""
        self.sheets_service = None
        
        # Config cố định
        self.config = {
            "google_sheets": {
                "credentials_file": "ivory-haven-463209-b8-09944271707f.json",
                "spreadsheet_id": "1sYzuvnv-lzQcv-IZjT672LTpfUrqdWCesx4pW8mIuqM"
            },
            "github": {
                "username": "corner-25",
                "repository": "vehicle-storage",
                "token": self.get_github_token(),
                "branch": "main"
            }
        }
        
        # Vehicle classifications
        self.admin_vehicles = ["51B-330.67", "50A-012.59", "50A-007.20", "51A-1212", "50A-004.55"]
        self.ambulance_vehicles = ["50A-007.39", "50M-004.37", "50A-009.44", "50A-010.67", 
                                 "50M-002.19", "51B-509.51", "50A-019.90", "50A-018.35"]
        
        # Driver mapping
        self.driver_names = {
            "ngochai191974@gmail.com": "Ngọc Hải",
            "phongthai230177@gmail.com": "Thái Phong", 
            "dunglamlong@gmail.com": "Long Dũng",
            "trananhtuan461970@gmail.com": "Anh Tuấn",
            "thanhdungvo29@gmail.com": "Thanh Dũng",
            "duck79884@gmail.com": "Đức",
            "ngohoangxuyen@gmail.com": "Hoàng Xuyên",
            "hodinhxuyen@gmail.com": "Đình Xuyên",
            "nvhung1981970@gmail.com": "Văn Hùng",
            "thanggptk21@gmail.com": "Văn Thảo",
            "nguyenhung091281@gmail.com": "Nguyễn Hùng",
            "nguyemthanhtrung12345@gmail.com": "Thành Trung",
            "nguyenhungumc@gmail.com": "Nguyễn Hùng",
            "dvo567947@gmail.com": "Tài xế khác",
            "traannhtuan461970@gmail.com": "Anh Tuấn",
            "hoanganhsie1983@gmail.com": "Hoàng Anh",
            "hoanganhsieumc@gmail.com": "Hoàng Anh",
            "thaonguyenvan860@gmail.com": "Văn Thảo"
        }
        
        # Stats với tracking chi tiết
        self.sync_stats = {
            'total_syncs': 0,
            'successful_syncs': 0,
            'last_sync': None,
            'last_error': None,
            'sheets_processed': 0,
            'sheets_skipped': 0,
            'sheets_failed': 0,
            'total_records': 0
        }
    
    def get_github_token(self) -> str:
        token = os.getenv('GITHUB_TOKEN')
        if token and len(token) > 10:
            return token
        
        token_file = "github_token.txt"
        if os.path.exists(token_file):
            try:
                with open(token_file, 'r') as f:
                    token = f.read().strip()
                if token and token != "YOUR_TOKEN_HERE":
                    return token
            except:
                pass
        
        if __name__ == "__main__":
            print("🔑 GITHUB TOKEN SETUP")
            print("=" * 40)
            print("Nhập GitHub token:")
            token = input("Token: ").strip()
            if token:
                return token
        
        return "YOUR_TOKEN_HERE"
    
    def get_google_credentials(self):
        """Get Google credentials from Streamlit secrets or file"""
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and 'google_credentials' in st.secrets:
                import tempfile
                import json
                
                creds_dict = {
                    "type": st.secrets.google_credentials.type,
                    "project_id": st.secrets.google_credentials.project_id,
                    "private_key_id": st.secrets.google_credentials.private_key_id,
                    "private_key": st.secrets.google_credentials.private_key,
                    "client_email": st.secrets.google_credentials.client_email,
                    "client_id": st.secrets.google_credentials.client_id,
                    "auth_uri": st.secrets.google_credentials.auth_uri,
                    "token_uri": st.secrets.google_credentials.token_uri,
                    "auth_provider_x509_cert_url": st.secrets.google_credentials.auth_provider_x509_cert_url,
                    "client_x509_cert_url": st.secrets.google_credentials.client_x509_cert_url,
                    "universe_domain": st.secrets.google_credentials.universe_domain
                }
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(creds_dict, f)
                    return f.name
        except Exception as e:
            logger.error(f"Error getting Streamlit secrets: {e}")
        
        credentials_file = self.config['google_sheets']['credentials_file']
        if os.path.exists(credentials_file):
            return credentials_file
        
        return None
    
    def authenticate_google_sheets(self) -> bool:
        """Xác thực Google Sheets với retry"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"🔐 Google Sheets authentication attempt {attempt + 1}/{max_retries}")
                
                credentials_file = self.get_google_credentials()
                if not credentials_file:
                    logger.error("❌ Không tìm thấy Google credentials")
                    return False
                
                with open(credentials_file, 'r', encoding='utf-8') as f:
                    creds_data = json.load(f)
                
                scopes = [
                    'https://www.googleapis.com/auth/spreadsheets.readonly',
                    'https://www.googleapis.com/auth/drive.readonly'
                ]
                
                credentials = service_account.Credentials.from_service_account_info(
                    creds_data, scopes=scopes
                )
                
                self.sheets_service = build('sheets', 'v4', credentials=credentials)
                
                # Test connection
                spreadsheet_id = self.config['google_sheets']['spreadsheet_id']
                test_result = self.sheets_service.spreadsheets().get(
                    spreadsheetId=spreadsheet_id
                ).execute()
                
                logger.info("✅ Google Sheets connected successfully")
                return True
                
            except Exception as e:
                logger.error(f"❌ Google Sheets error attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"⏳ Waiting 5 seconds before retry...")
                    time.sleep(5)
                continue
        
        logger.error("❌ Failed to authenticate after all retries")
        return False
    
    def read_single_sheet_with_retry(self, spreadsheet_id: str, sheet_name: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """Đọc một sheet với retry mechanism"""
        for attempt in range(max_retries):
            try:
                logger.debug(f"📖 Reading sheet '{sheet_name}' - attempt {attempt + 1}/{max_retries}")
                
                # Add delay between attempts to avoid rate limiting
                if attempt > 0:
                    delay = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds
                    logger.debug(f"⏳ Waiting {delay}s before retry...")
                    time.sleep(delay)
                
                # Read sheet data with proper range formatting
                range_name = f"'{sheet_name}'"  # Wrap in quotes for special characters
                result = self.sheets_service.spreadsheets().values().get(
                    spreadsheetId=spreadsheet_id,
                    range=range_name,
                    valueRenderOption='UNFORMATTED_VALUE',  # Get raw values
                    dateTimeRenderOption='FORMATTED_STRING'  # Get formatted dates
                ).execute()
                
                values = result.get('values', [])
                logger.debug(f"📊 Sheet '{sheet_name}': Got {len(values)} rows")
                
                # CHANGED: Don't skip sheets with < 2 rows, log instead
                if len(values) == 0:
                    logger.warning(f"⚠️ Sheet '{sheet_name}': EMPTY - but not skipping")
                    return pd.DataFrame()  # Return empty DF instead of None
                
                if len(values) == 1:
                    logger.warning(f"⚠️ Sheet '{sheet_name}': Only headers, no data rows")
                    return pd.DataFrame()  # Return empty DF instead of None
                
                # Process data
                headers = values[0]
                data_rows = values[1:]
                
                logger.debug(f"📋 Sheet '{sheet_name}': Headers = {headers}")
                logger.debug(f"📊 Sheet '{sheet_name}': {len(data_rows)} data rows")
                
                # Clean data - handle uneven rows
                max_cols = len(headers)
                cleaned_data = []
                
                for i, row in enumerate(data_rows):
                    # Extend short rows
                    while len(row) < max_cols:
                        row.append(None)
                    
                    # Truncate long rows
                    if len(row) > max_cols:
                        logger.debug(f"📏 Sheet '{sheet_name}' row {i}: Truncating from {len(row)} to {max_cols} columns")
                        row = row[:max_cols]
                    
                    cleaned_data.append(row)
                
                # Create DataFrame
                df = pd.DataFrame(cleaned_data, columns=headers)
                
                # Add metadata
                df['Mã xe'] = sheet_name
                df['Tên tài xế'] = df['Email Address'].map(self.driver_names).fillna(df['Email Address']) if 'Email Address' in df.columns else 'Unknown'
                
                # Set vehicle type and handle missing columns
                if sheet_name in self.admin_vehicles:
                    df['Loại xe'] = 'Hành chính'
                    df['Chi tiết chuyến xe'] = None
                    df['Doanh thu'] = None
                else:
                    df['Loại xe'] = 'Cứu thương'
                
                logger.info(f"✅ Sheet '{sheet_name}': Successfully processed {len(df)} trips")
                return df
                
            except HttpError as e:
                if e.resp.status == 429:  # Rate limit
                    logger.warning(f"⏳ Rate limited on sheet '{sheet_name}' - attempt {attempt + 1}")
                    time.sleep(10 * (attempt + 1))  # Longer delay for rate limits
                    continue
                else:
                    logger.error(f"❌ HTTP Error reading sheet '{sheet_name}': {e}")
                    if attempt == max_retries - 1:
                        return None
                    continue
                    
            except Exception as e:
                logger.error(f"❌ Error reading sheet '{sheet_name}' attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"💥 FAILED to read sheet '{sheet_name}' after {max_retries} attempts")
                    return None
                continue
        
        return None
    
    def read_all_sheets(self) -> Optional[pd.DataFrame]:
        """Đọc tất cả sheets với improved error handling"""
        try:
            spreadsheet_id = self.config['google_sheets']['spreadsheet_id']
            logger.info(f"📚 Starting to read all sheets from spreadsheet: {spreadsheet_id}")
            
            # Get sheet metadata with retry
            max_retries = 3
            sheet_metadata = None
            
            for attempt in range(max_retries):
                try:
                    sheet_metadata = self.sheets_service.spreadsheets().get(
                        spreadsheetId=spreadsheet_id
                    ).execute()
                    break
                except Exception as e:
                    logger.error(f"❌ Error getting sheet metadata attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(3)
                        continue
                    else:
                        logger.error("💥 Failed to get sheet metadata after all retries")
                        return None
            
            if not sheet_metadata:
                logger.error("❌ No sheet metadata available")
                return None
            
            sheets = sheet_metadata.get('sheets', [])
            logger.info(f"📋 Found {len(sheets)} sheets in spreadsheet")
            
            all_data = []
            successful_sheets = []
            failed_sheets = []
            empty_sheets = []
            
            # Process each sheet
            for i, sheet in enumerate(sheets):
                sheet_name = sheet['properties']['title']
                sheet_id = sheet['properties']['sheetId']
                is_hidden = sheet['properties'].get('hidden', False)
                
                logger.info(f"🔄 Processing sheet {i+1}/{len(sheets)}: '{sheet_name}' (ID: {sheet_id}, Hidden: {is_hidden})")
                
                # Skip hidden sheets but log them
                if is_hidden:
                    logger.warning(f"⚠️ Skipping hidden sheet: '{sheet_name}'")
                    self.sync_stats['sheets_skipped'] += 1
                    continue
                
                # Try to read the sheet
                df = self.read_single_sheet_with_retry(spreadsheet_id, sheet_name)
                
                if df is None:
                    logger.error(f"💥 FAILED to read sheet: '{sheet_name}'")
                    failed_sheets.append(sheet_name)
                    self.sync_stats['sheets_failed'] += 1
                    continue
                
                if df.empty:
                    logger.warning(f"📭 Empty sheet: '{sheet_name}'")
                    empty_sheets.append(sheet_name)
                    # Don't count empty sheets as failed - they might be templates
                    continue
                
                # Successfully read sheet with data
                all_data.append(df)
                successful_sheets.append(sheet_name)
                self.sync_stats['sheets_processed'] += 1
                self.sync_stats['total_records'] += len(df)
                
                logger.info(f"✅ Successfully processed sheet '{sheet_name}': {len(df)} records")
            
            # Log summary
            logger.info("=" * 60)
            logger.info("📊 SHEET PROCESSING SUMMARY:")
            logger.info(f"✅ Successfully processed: {len(successful_sheets)} sheets")
            logger.info(f"📭 Empty sheets: {len(empty_sheets)} sheets")
            logger.info(f"❌ Failed sheets: {len(failed_sheets)} sheets")
            logger.info(f"⚠️ Hidden/Skipped sheets: {self.sync_stats['sheets_skipped']} sheets")
            logger.info(f"📊 Total records: {self.sync_stats['total_records']}")
            
            if successful_sheets:
                logger.info("✅ Successful sheets:")
                for sheet in successful_sheets:
                    logger.info(f"   • {sheet}")
            
            if empty_sheets:
                logger.warning("📭 Empty sheets:")
                for sheet in empty_sheets:
                    logger.warning(f"   • {sheet}")
            
            if failed_sheets:
                logger.error("❌ Failed sheets:")
                for sheet in failed_sheets:
                    logger.error(f"   • {sheet}")
            
            logger.info("=" * 60)
            
            # Combine all successful data
            if not all_data:
                logger.error("💥 NO DATA COLLECTED from any sheets!")
                return None
            
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"🎉 TOTAL COMBINED: {len(combined_df)} trips from {combined_df['Mã xe'].nunique()} vehicles")
            
            return combined_df
            
        except Exception as e:
            logger.error(f"💥 Critical error in read_all_sheets: {e}")
            return None
    
    def save_to_github(self, data: pd.DataFrame) -> bool:
        """Lưu dữ liệu lên GitHub (FIXED VERSION - NO ensure_ascii)"""
        try:
            github_config = self.config['github']
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Check if repo exists first
            check_url = f"https://api.github.com/repos/{github_config['username']}/{github_config['repository']}"
            headers = {
                'Authorization': f"token {github_config['token']}",
                'Accept': 'application/vnd.github.v3+json'
            }
            
            response = requests.get(check_url, headers=headers)
            
            if response.status_code == 404:
                logger.error("❌ Repository không tồn tại!")
                return False
            elif response.status_code != 200:
                logger.error(f"❌ Không thể truy cập repository: {response.text}")
                return False
            
            logger.info("✅ Repository found")
            
            # Convert to JSON without ensure_ascii parameter
            combined_json = data.to_json(orient='records', indent=2)
            
            logger.info(f"📄 JSON content length: {len(combined_json)} characters")
            
            if not combined_json or combined_json.strip() == "":
                logger.error("❌ CRITICAL: JSON content is empty!")
                return False
            
            # Save latest data
            latest_filename = "data/latest/fleet_data_latest.json"
            logger.info(f"🔄 Uploading main data file: {latest_filename}")
            
            upload_success = self.upload_file_to_github(
                combined_json,
                latest_filename,
                f"Update latest data - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            if not upload_success:
                logger.error("❌ CRITICAL: Failed to upload main data file!")
                return False
            else:
                logger.info("✅ Main data file uploaded successfully")
            
            # Save summary
            logger.info("🔄 Uploading summary file...")
            summary = self.generate_summary(data)
            summary_json = json.dumps(summary, indent=2, ensure_ascii=False)
            summary_filename = "data/summary/summary_latest.json"
            
            summary_success = self.upload_file_to_github(
                summary_json,
                summary_filename,
                f"Update summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            if summary_success:
                logger.info("✅ Summary file uploaded successfully")
            else:
                logger.warning("⚠️ Summary upload failed, but main data is OK")
            
            logger.info("✅ Data saved to GitHub successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ GitHub save error: {e}")
            return False
    
    def upload_file_to_github(self, content: str, filename: str, commit_message: str) -> bool:
        """Upload single file to GitHub"""
        try:
            github_config = self.config['github']
            
            url = f"https://api.github.com/repos/{github_config['username']}/{github_config['repository']}/contents/{filename}"
            headers = {
                'Authorization': f"token {github_config['token']}",
                'Accept': 'application/vnd.github.v3+json'
            }
            
            content_encoded = base64.b64encode(content.encode('utf-8')).decode('utf-8')
            
            data = {
                "message": commit_message,
                "content": content_encoded,
                "branch": github_config['branch']
            }
            
            # Check if file exists
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data["sha"] = response.json()["sha"]
                logger.info(f"📝 Updating existing file: {filename}")
            else:
                logger.info(f"📝 Creating new file: {filename}")
            
            # Upload file
            response = requests.put(url, headers=headers, json=data)
            
            if response.status_code in [200, 201]:
                logger.info(f"✅ Successfully uploaded: {filename}")
                return True
            else:
                logger.error(f"❌ Upload error {filename}")
                logger.error(f"Status: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Upload file error: {e}")
            return False
    
    def generate_summary(self, data: pd.DataFrame) -> Dict:
        """Tạo summary stats với thông tin chi tiết về sync"""
        try:
            admin_data = data[data['Loại xe'] == 'Hành chính']
            ambulance_data = data[data['Loại xe'] == 'Cứu thương']
            
            summary = {
                'timestamp': datetime.now().isoformat(),
                'total_trips': len(data),
                'total_vehicles': data['Mã xe'].nunique(),
                'admin_vehicles': len(admin_data['Mã xe'].unique()),
                'ambulance_vehicles': len(ambulance_data['Mã xe'].unique()),
                'admin_trips': len(admin_data),
                'ambulance_trips': len(ambulance_data),
                'top_vehicles': data['Mã xe'].value_counts().head(5).to_dict(),
                'top_drivers': data['Tên tài xế'].value_counts().head(5).to_dict(),
                'sync_stats': self.sync_stats,
                'vehicle_list': sorted(data['Mã xe'].unique().tolist())  # List all vehicles processed
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Summary error: {e}")
            return {'error': str(e), 'sync_stats': self.sync_stats}
    
    def sync_now(self) -> bool:
        """Thực hiện sync ngay với improved tracking"""
        logger.info("🚀 Starting IMPROVED manual sync...")
        logger.info("=" * 60)
        
        # Reset stats for this sync
        self.sync_stats['total_syncs'] += 1
        self.sync_stats['sheets_processed'] = 0
        self.sync_stats['sheets_skipped'] = 0
        self.sync_stats['sheets_failed'] = 0
        self.sync_stats['total_records'] = 0
        
        try:
            # 1. Authenticate Google Sheets
            logger.info("🔐 Step 1: Authenticating Google Sheets...")
            if not self.authenticate_google_sheets():
                raise Exception("Google Sheets authentication failed")
            
            # 2. Read all data
            logger.info("📚 Step 2: Reading all sheets data...")
            combined_data = self.read_all_sheets()
            if combined_data is None or len(combined_data) == 0:
                raise Exception("No data from Google Sheets")
            
            # 3. Save to GitHub
            logger.info("🐙 Step 3: Saving to GitHub...")
            if not self.save_to_github(combined_data):
                raise Exception("GitHub save failed")
            
            # 4. Update stats
            self.sync_stats['successful_syncs'] += 1
            self.sync_stats['last_sync'] = datetime.now().isoformat()
            
            logger.info("=" * 60)
            logger.info("🎉 SYNC COMPLETED SUCCESSFULLY!")
            logger.info(f"📊 Final stats:")
            logger.info(f"   • Total trips synced: {len(combined_data)}")
            logger.info(f"   • Vehicles found: {combined_data['Mã xe'].nunique()}")
            logger.info(f"   • Sheets processed: {self.sync_stats['sheets_processed']}")
            logger.info(f"   • Sheets failed: {self.sync_stats['sheets_failed']}")
            logger.info(f"   • Sheets skipped: {self.sync_stats['sheets_skipped']}")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            self.sync_stats['last_error'] = str(e)
            logger.error("=" * 60)
            logger.error(f"💥 SYNC FAILED: {e}")
            logger.error(f"📊 Partial stats:")
            logger.error(f"   • Sheets processed: {self.sync_stats['sheets_processed']}")
            logger.error(f"   • Sheets failed: {self.sync_stats['sheets_failed']}")
            logger.error(f"   • Sheets skipped: {self.sync_stats['sheets_skipped']}")
            logger.error("=" * 60)
            return False
    
    def test_connections(self) -> Dict[str, bool]:
        """Test connections với detailed info"""
        results = {
            'google_sheets': False,
            'github': False
        }
        
        try:
            # Test Google Sheets
            logger.info("🧪 Testing Google Sheets connection...")
            if self.authenticate_google_sheets():
                results['google_sheets'] = True
                logger.info("✅ Google Sheets connection: OK")
            else:
                logger.error("❌ Google Sheets connection: FAILED")
            
            # Test GitHub
            logger.info("🧪 Testing GitHub connection...")
            github_config = self.config['github']
            if github_config['token'] != "YOUR_TOKEN_HERE":
                headers = {
                    'Authorization': f"token {github_config['token']}",
                    'Accept': 'application/vnd.github.v3+json'
                }
                response = requests.get('https://api.github.com/user', headers=headers)
                if response.status_code == 200:
                    results['github'] = True
                    user_info = response.json()
                    logger.info(f"✅ GitHub user: {user_info.get('login')}")
                else:
                    logger.error(f"❌ GitHub connection failed: {response.status_code}")
            else:
                logger.error("❌ GitHub token not configured")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Test error: {e}")
            return results

    def diagnose_missing_sheets(self) -> Dict:
        """Chẩn đoán các sheet bị thiếu hoặc lỗi"""
        try:
            logger.info("🔍 DIAGNOSING MISSING SHEETS...")
            
            if not self.authenticate_google_sheets():
                return {"error": "Cannot authenticate Google Sheets"}
            
            spreadsheet_id = self.config['google_sheets']['spreadsheet_id']
            
            # Get all sheets metadata
            sheet_metadata = self.sheets_service.spreadsheets().get(
                spreadsheetId=spreadsheet_id
            ).execute()
            
            sheets = sheet_metadata.get('sheets', [])
            diagnosis = {
                'total_sheets': len(sheets),
                'visible_sheets': [],
                'hidden_sheets': [],
                'empty_sheets': [],
                'error_sheets': [],
                'successful_sheets': [],
                'sheet_details': {}
            }
            
            logger.info(f"📋 Found {len(sheets)} total sheets")
            
            for sheet in sheets:
                sheet_name = sheet['properties']['title']
                sheet_id = sheet['properties']['sheetId']
                is_hidden = sheet['properties'].get('hidden', False)
                
                sheet_info = {
                    'id': sheet_id,
                    'hidden': is_hidden,
                    'status': 'unknown',
                    'row_count': 0,
                    'has_data': False,
                    'error': None
                }
                
                if is_hidden:
                    diagnosis['hidden_sheets'].append(sheet_name)
                    sheet_info['status'] = 'hidden'
                else:
                    diagnosis['visible_sheets'].append(sheet_name)
                    
                    # Try to read the sheet
                    try:
                        result = self.sheets_service.spreadsheets().values().get(
                            spreadsheetId=spreadsheet_id,
                            range=f"'{sheet_name}'"
                        ).execute()
                        
                        values = result.get('values', [])
                        sheet_info['row_count'] = len(values)
                        
                        if len(values) == 0:
                            diagnosis['empty_sheets'].append(sheet_name)
                            sheet_info['status'] = 'empty'
                        elif len(values) == 1:
                            sheet_info['status'] = 'headers_only'
                            sheet_info['has_data'] = False
                        else:
                            diagnosis['successful_sheets'].append(sheet_name)
                            sheet_info['status'] = 'success'
                            sheet_info['has_data'] = True
                        
                        logger.info(f"📊 {sheet_name}: {len(values)} rows, Status: {sheet_info['status']}")
                        
                    except Exception as e:
                        diagnosis['error_sheets'].append(sheet_name)
                        sheet_info['status'] = 'error'
                        sheet_info['error'] = str(e)
                        logger.error(f"❌ {sheet_name}: Error - {e}")
                
                diagnosis['sheet_details'][sheet_name] = sheet_info
            
            # Print diagnosis summary
            logger.info("=" * 60)
            logger.info("🔍 DIAGNOSIS SUMMARY:")
            logger.info(f"📊 Total sheets: {diagnosis['total_sheets']}")
            logger.info(f"👁️ Visible sheets: {len(diagnosis['visible_sheets'])}")
            logger.info(f"🙈 Hidden sheets: {len(diagnosis['hidden_sheets'])}")
            logger.info(f"✅ Successful sheets: {len(diagnosis['successful_sheets'])}")
            logger.info(f"📭 Empty sheets: {len(diagnosis['empty_sheets'])}")
            logger.info(f"❌ Error sheets: {len(diagnosis['error_sheets'])}")
            
            if diagnosis['successful_sheets']:
                logger.info("✅ Sheets with data:")
                for sheet in diagnosis['successful_sheets']:
                    row_count = diagnosis['sheet_details'][sheet]['row_count']
                    logger.info(f"   • {sheet}: {row_count} rows")
            
            if diagnosis['hidden_sheets']:
                logger.warning("🙈 Hidden sheets:")
                for sheet in diagnosis['hidden_sheets']:
                    logger.warning(f"   • {sheet}")
            
            if diagnosis['empty_sheets']:
                logger.warning("📭 Empty sheets:")
                for sheet in diagnosis['empty_sheets']:
                    logger.warning(f"   • {sheet}")
            
            if diagnosis['error_sheets']:
                logger.error("❌ Error sheets:")
                for sheet in diagnosis['error_sheets']:
                    error = diagnosis['sheet_details'][sheet]['error']
                    logger.error(f"   • {sheet}: {error}")
            
            logger.info("=" * 60)
            
            return diagnosis
            
        except Exception as e:
            logger.error(f"💥 Diagnosis failed: {e}")
            return {"error": str(e)}


def main():
    """Main function với enhanced menu"""
    print("🚀 IMPROVED FLEET DATA SYNC - NO MORE MISSING VEHICLES!")
    print("=" * 60)
    print("📊 Google Sheets → GitHub with Better Error Handling")
    print("=" * 60)
    
    sync_engine = ImprovedFleetSync()
    
    # Check GitHub token
    if sync_engine.config['github']['token'] == "YOUR_TOKEN_HERE":
        print("❌ GitHub token chưa được setup!")
        return
    
    while True:
        print("\n📋 ENHANCED MENU:")
        print("1. 🧪 Test connections")
        print("2. 🔄 Sync ngay (improved)")
        print("3. 📊 Xem stats")
        print("4. 🔍 Chẩn đoán sheet bị thiếu")  # NEW
        print("5. 🌐 Open GitHub repo")
        print("6. 📋 View detailed logs")  # NEW
        print("7. 🚪 Exit")
        
        choice = input("\nChọn (1-7): ").strip()
        
        if choice == '1':
            print("\n🧪 Testing connections...")
            results = sync_engine.test_connections()
            print(f"📊 Google Sheets: {'✅' if results['google_sheets'] else '❌'}")
            print(f"🐙 GitHub: {'✅' if results['github'] else '❌'}")
        
        elif choice == '2':
            print("\n🔄 Starting IMPROVED sync...")
            success = sync_engine.sync_now()
            if success:
                print("🎉 Sync completed successfully!")
                print("💡 Check the detailed logs above for any issues")
            else:
                print("💥 Sync failed! Check logs for details")
        
        elif choice == '3':
            print("\n📊 SYNC STATS:")
            stats = sync_engine.sync_stats
            for key, value in stats.items():
                print(f"  {key}: {value}")
        
        elif choice == '4':
            print("\n🔍 Diagnosing missing sheets...")
            diagnosis = sync_engine.diagnose_missing_sheets()
            if 'error' in diagnosis:
                print(f"❌ Diagnosis failed: {diagnosis['error']}")
            else:
                print("✅ Diagnosis completed - check logs above for details")
        
        elif choice == '5':
            repo_url = f"https://github.com/{sync_engine.config['github']['username']}/{sync_engine.config['github']['repository']}"
            print(f"\n🌐 GitHub Repository:")
            print(f"   {repo_url}")
        
        elif choice == '6':
            print("\n📋 Log files:")
            print("   • fleet_sync_detailed.log - Current session")
            print("   • fleet_sync.log - Previous sessions") 
            if os.path.exists("fleet_sync_detailed.log"):
                print("\n📖 Last 10 lines of detailed log:")
                try:
                    with open("fleet_sync_detailed.log", 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for line in lines[-10:]:
                            print(f"   {line.strip()}")
                except Exception as e:
                    print(f"❌ Error reading log: {e}")
        
        elif choice == '7':
            print("👋 Bye!")
            break
        
        else:
            print("❌ Invalid choice!")


if __name__ == "__main__":
    main()
