import nibabel as nib
from pathlib import Path

# Define a color palette for different file types
FILE_TYPE_COLORS = {
    'trk': {'bg': '#b4ede3', 'text': 'white'},
    'tck': {'bg': '#fbe5ca', 'text': 'white'},
    'xlsx': {'bg': '#c5f2c7', 'text': 'white'},
    'txt': {'bg': '#c5f2c7', 'text': 'white'},
    'csv': {'bg': '#fac1d6', 'text': 'white'},
    'png': {'bg': '#ecb9f9', 'text': 'white'},
    'jpg': {'bg': '#c5dcfa', 'text': 'white'},
    'default': {'bg': '#c5dcfa', 'text': 'white'}
}


VIEWS = {
    "axial": "Axial View",
    "coronal": "Coronal View",
    "sagittal": "Sagittal View"
}


RGBA_TO_CMAP = {
    "255,87,51,1": "spring",  # Bright Orange
    "255,195,0,1": "YlOrBr",  # Sunflower Yellow
    "255,255,102,1": "YlGn",  # Lemon Yellow
    "139,87,42,1": "tab20b",  # Mocha Brown
    "39,174,96,1": "summer",  # Emerald Green
    "20,90,50,1": "PiYG",  # Forest Green
    "232,67,147,1": "PuRd",  # Hot Pink
    "108,52,131,1": "Purples",  # Deep Purple
    "52,152,219,1": "Blues",  # Sky Blue
    "26,188,156,1": "GnBu",  # Turquoise
    "184,233,134,1": "YlGnBu",  # Pale Lime
    "44,62,80,1": "cool",  # Deep Navy
    "255,159,243,1": "gist_rainbow",  # Plum
    "253,106,0,1": "autumn",  # Pumpkin Orange
    "196,48,92,1": "coolwarm",  # Rose Red
    "0,0,0,0": "viridis",  # Transparent / Auto
}

# Get min and max of each view
nii_path = Path(__file__).parents[1] / "assets" / "brain_image.nii.gz"
img = nib.load(nii_path)
data = img.get_fdata()
shape = data.shape

# Min and max slices in each orientation
slice_ranges = {
    "sagittal": {"min": 0, "max" : shape[0] - 1},
    "coronal": {"min":0, "max":shape[1] - 1},
    "axial": {"min":0, "max":shape[2] - 1}
}
