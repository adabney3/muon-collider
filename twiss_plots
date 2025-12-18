# import pandas as pd

# with open('twiss_IR_v09.outx', 'r') as f:
#     lines = f.readlines()
#     print(lines)

import pandas as pd
import matplotlib.pyplot as plt

def read_outx_file(filename):
    """
    Read MAD-X Twiss .outx file and return as pandas DataFrame
    Extracts column names from the line starting with '*'
    """
    try:
        # First, read the file to find column names from the '*' line
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Find the line with column names (starts with '*')
        header_line = None
        data_start = None
        for i, line in enumerate(lines):
            if line.startswith('*'):
                header_line = line
            elif line.startswith('$'):
                # Data starts after the '$' line
                data_start = i + 1
                break
        
        if header_line is None or data_start is None:
            raise ValueError("Could not find header line or data start")
        
        # Extract column names from header line
        # Remove the '*' and split by whitespace
        col_names = header_line.strip().replace('*', '').split()
        
        # Read the data starting from the data line
        data = pd.read_csv(filename, 
                          delim_whitespace=True,
                          skiprows=data_start,
                          names=col_names,
                          header=None)
        
        # Clean up data (remove quotes)
        for col in data.columns:
            if data[col].dtype == 'object':
                data[col] = data[col].astype(str).str.strip('"')
        
        # Convert numeric columns to float
        numeric_cols = [col for col in data.columns if col not in ['NAME', 'KEYWORD']]
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        print(f"Successfully read {filename}")
        print(f"Shape: {data.shape[0]} rows, {data.shape[1]} columns")
        return data
    except Exception as e:
        print(f"Error reading file: {e}")
        import traceback
        traceback.print_exc()
        return None

df = read_outx_file('twiss_IR_v09.outx')

def plot_single_column(df, column_name, title=None):
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df[column_name], marker='o', markersize=3)
    plt.xlabel('Index')
    plt.ylabel(column_name)
    plt.title(title if title else f'{column_name} vs Index')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_two_columns(df, x_col, y_col, title=None):
    plt.figure(figsize=(10, 6))
    plt.plot(df[x_col], df[y_col], marker='o', markersize=3)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title if title else f'{y_col} vs {x_col}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_multiple_columns(df, x_col, y_cols, title=None):
    plt.figure(figsize=(12, 6))
    for y_col in y_cols:
        plt.plot(df[x_col], df[y_col], marker='o', markersize=3, label=y_col)
    plt.xlabel(x_col)
    plt.ylabel('Values')
    plt.title(title if title else 'Multiple Columns')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# Plot Beta functions vs S position
plot_two_columns(df, 'S', 'BETX', 'Beta X vs S Position')
plot_two_columns(df, 'S', 'BETY', 'Beta Y vs S Position')

# Plot both beta functions together
plot_multiple_columns(df, 'S', ['BETX', 'BETY'], 'Beta Functions vs S')

# Plot dispersion
plot_two_columns(df, 'S', 'DX', 'Horizontal Dispersion vs S')

# Plot alpha functions
plot_multiple_columns(df, 'S', ['ALFX', 'ALFY'], 'Alpha Functions vs S')


# Filter by element type (e.g., only quadrupoles)
quadrupoles = df[df['KEYWORD'] == 'QUADRUPOLE']
print(f"Number of quadrupoles: {len(quadrupoles)}")

# Sort by S position (if not already sorted)
df_sorted = df.sort_values('S')

# Find element with maximum beta
max_betx_idx = df['BETX'].idxmax()
max_betx_element = df.loc[max_betx_idx]
print(f"Max BETX at: {max_betx_element['NAME']}, S = {max_betx_element['S']}")

# Get elements in a specific S range
ip_region = df[(df['S'] > 180) & (df['S'] < 185)]
print(f"Elements near IP: {len(ip_region)}")


