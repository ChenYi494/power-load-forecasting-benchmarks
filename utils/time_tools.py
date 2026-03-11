from datetime import datetime
import pytz

def format_time(ts):
    tz = pytz.timezone('Asia/Taipei')
    return datetime.fromtimestamp(ts, tz).strftime('%Y-%m-%d %H:%M:%S')