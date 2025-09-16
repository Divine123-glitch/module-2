import pandas as pd

# Read the Excel file
excel_file = 'sales_data_cleaned.xlsx'  # Replace with your Excel file path
df = pd.read_excel(excel_file)

# Convert to CSV
csv_file = 'sales_data.csv'  # Replace with your desired CSV file path
df.to_csv(csv_file, index=False)

print(f"Excel file {excel_file} has been converted to {csv_file}")