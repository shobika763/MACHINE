import pandas as pd

# Define the data
data = {
    "S.No": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "Ozone": [41, 36, 12, 18, 27, 28, 23, 19, 8, 24, 7, 16, 11, 14, 18, 14, 34, 6, 30, 11],
    "Solar.R": [190, 118, 149, 313, 192, 193, 299, 99, 19, 194, 152, 256, 290, 274, 65, 334, 307, 78, 322, 44],
    "Wind": [7.4, 8.0, 12.6, 11.5, 14.3, 14.9, 8.6, 13.8, 20.1, 8.6, 6.9, 9.7, 9.2, 10.9, 13.2, 11.5, 12.0, 18.4, 11.5, 9.7],
    "Temp": [67, 72, 74, 62, 56, 66, 65, 59, 61, 69, 74, 69, 66, 68, 58, 64, 66, 57, 68, 62],
    "Month": [5] * 20,
    "Day": list(range(1, 21))
}

# Create a DataFrame
df = pd.DataFrame(data)

# Summarize the data
summary = df.describe()

# Display summary
print("Summary of Air Quality Data:")
print(summary)