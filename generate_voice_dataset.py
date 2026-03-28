import numpy as np
import pandas as pd

rows = []

emotions = ["Happy", "Sad", "Angry", "Neutral"]

# Generate dummy MFCC-like data
for emotion in emotions:
    for _ in range(25):  # 25 samples per emotion (total 100 rows)
        mfcc = np.random.normal(loc=0, scale=1, size=20)

        # Add slight variation per emotion (important)
        if emotion == "Happy":
            mfcc += 2
        elif emotion == "Sad":
            mfcc -= 2
        elif emotion == "Angry":
            mfcc += 1
        else:
            mfcc += 0

        row = list(mfcc) + [emotion]
        rows.append(row)

# Column names
columns = [f"mfcc{i}" for i in range(1, 21)] + ["label"]

df = pd.DataFrame(rows, columns=columns)

# Save dataset
df.to_csv("voice_data.csv", index=False)

print("✅ voice_data.csv created!")