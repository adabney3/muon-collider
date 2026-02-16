import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_twiss_parameters(filename):
    with open(filename, 'r') as f:
            lines = f.readlines()
    
    header_line = None
    data_start = None
    for i, line in enumerate(lines):
            if line.startswith('*'):
                header_line = line
            elif line.startswith('$'):
                data_start = i + 1
                break
    
    col_names = header_line.strip().replace('*', '').split()

    data = pd.read_csv(filename, 
                          delim_whitespace=True,
                          skiprows=data_start,
                          names=col_names,
                          header=None)
    
    for col in data.columns:
            if data[col].dtype == 'object':
                data[col] = data[col].astype(str).str.strip('"')
    
    numeric_cols = [col for col in data.columns if col not in ['NAME', 'KEYWORD']]
    for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    return data

df = get_twiss_parameters('twiss_IR_v09.outx')

if df is not None:

    #calculate gamma
    df['GAMMAX_CALC'] = (1 + df['ALFX']**2) / df['BETX']
    df['GAMMAY_CALC'] = (1 + df['ALFY']**2) / df['BETY']

beta = df['BETX'] # semi-major axis
gamma = df['GAMMAX_CALC']  # semi-minor axis
x = df['X']  # center x
y = df['Y']  # center y
epsilon = 0.00000000052830000  #EX (emmitance) from twiss file header
alpha = df['ALFX']  #alpha corresponds to tilt

phi = (1/2) * np.arctan2(2 * df['ALFX'], df['BETX'] - df['GAMMAX_CALC'])
theta = np.linspace(0, 2 * np.pi, 100)

x = np.sqrt(epsilon * beta) * np.cos(theta)
x_prime = -np.sqrt(epsilon / beta) * (alpha * np.cos(theta) + np.sin(theta))

plt.figure(figsize=(8, 6))
plt.plot(x, x_prime, 'b-', linewidth=2)
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ellipse')
plt.show()