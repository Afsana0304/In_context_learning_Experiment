import pandas as pd


file_path = "/hpi/fs00/home/afsana.mimi/llama_project/synthetic_data/summary_output_json.json"

# Load JSON data
with open(file_path, 'r') as file:
    data = pd.read_json(file)

# Convert JSON to DataFrame
df = pd.DataFrame(data)



# Save the DataFrame to a file (CSV, Excel, or JSON)
output_path_csv = "/hpi/fs00/home/afsana.mimi/llama_project/synthetic_data/data1.csv"
#print(df)

# Save as CSV
df.to_csv(output_path_csv, index=False)



print(f"Data saved as CSV, Excel, and JSON at specified locations!")
