import pandas as pd

df = pd.read_csv("online_shoppers_intention.csv")

print("Dataset Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Distribution:")
print(df["Revenue"].value_counts())

print("\nTarget Percentage:")
print(df["Revenue"].value_counts(normalize=True) * 100)

print("\nData Types:")
print(df.dtypes)

print("\nBasic Statistics:")
print(df.describe())

print("\nPurchase Rate:")
print(df.groupby("Revenue")["PageValues"].mean())

print("\nAverage Product Related Pages:")
print(df.groupby("Revenue")["ProductRelated"].mean())

print("\nAverage Product Related Duration:")
print(df.groupby("Revenue")["ProductRelated_Duration"].mean())

print("\nAverage Bounce Rate:")
print(df.groupby("Revenue")["BounceRates"].mean())

print("\nAverage Exit Rate:")
print(df.groupby("Revenue")["ExitRates"].mean())

print("\nPurchase by Visitor Type:")
print(pd.crosstab(df["VisitorType"], df["Revenue"], normalize="index") * 100)

print("\nPurchase by Weekend:")
print(pd.crosstab(df["Weekend"], df["Revenue"], normalize="index") * 100)


import matplotlib.pyplot as plt
import seaborn as sns

# 1. Purchase Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Revenue")
plt.title("Purchase vs No Purchase")
plt.xlabel("Purchase")
plt.ylabel("Number of Sessions")
plt.show()


# 2. Product Related Pages vs Purchase
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x="Revenue", y="ProductRelated")
plt.title("Product Related Pages vs Purchase")
plt.xlabel("Purchase")
plt.ylabel("Product Related Pages")
plt.show()


# 3. Page Values vs Purchase
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x="Revenue", y="PageValues")
plt.title("Page Values vs Purchase")
plt.xlabel("Purchase")
plt.ylabel("Page Values")
plt.show()


# 4. Bounce Rate vs Purchase
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x="Revenue", y="BounceRates")
plt.title("Bounce Rate vs Purchase")
plt.xlabel("Purchase")
plt.ylabel("Bounce Rate")
plt.show()