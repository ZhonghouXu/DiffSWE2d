import numpy as np
import torch

def load_asc_dem(filepath, device="cpu"):
    """Reads an ESRI ASCII DEM into a correctly shaped PyTorch tensor."""
    with open(filepath, 'r') as f:
        header = {}
        for _ in range(6):
            key, val = f.readline().split()
            header[key.lower()] = float(val)
            
    dem_array = np.loadtxt(filepath, skiprows=6)
    
    nodata = header.get('nodata_value', -9999)
    dem_array[dem_array == nodata] = 1000.0  
    
    dem_tensor = torch.tensor(dem_array, dtype=torch.float32, device=device)
    dem_tensor = dem_tensor.unsqueeze(0).unsqueeze(0)
    
    return dem_tensor, header

def load_rainfall_txt(filepath):
    """Reads rainfall txt (Col 1: Time, Col 2: Rain in mm) and converts to SI units."""
    data = np.loadtxt(filepath)
    times = data[:, 0]
    rain_meters = data[:, 1] / 1000.0 
    
    return times, rain_meters