import platform
import psutil

def get_hardware_profile():
    return {
        "cpu": platform.processor(),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "gpu": False
    }