import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the Excel file
# Replace 'your_file.xlsx' with the actual filename
df = pd.read_excel(r'D:\MLME project\Mlme\pythonProject\CNN_model_Health_forest1\simple\results.xlsx')

# Create a pivot table for the heatmap
pivot_table = df.pivot(index='Batch Size', columns='Learning Rate', values='Final Validation Loss')

# Create the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, cmap='YlOrRd', fmt='.3f')

plt.title('Validation Loss Heatmap')
plt.xlabel('Learning Rate')
plt.ylabel('Batch Size')
plt.tight_layout()
plt.show()