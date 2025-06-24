import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('netflix_movies.csv')
#print(df)
print(df.head(5))
print(df.info())
print(df.describe())

# checking for missing values
print(df.isnull().sum())
df.dropna(inplace=True)
# saving the cleaned data to a new csv file
df.to_csv('netflix_movies.csv', index=False)
print(df)

# checking for duplicates
print(df.duplicated().sum())
df.fillna('NuN',inplace=True)
print(df.duplicated().sum())

null=df.isnull().sum()
print(null)
# drop rows with any NaN values
df.dropna(inplace=True)
print(df.shape)
print(df.head(5))

# check for correct data types
print(df.dtypes)

# convert date column to datetime
df['date_added']=df['date_added'].astype('datetime64[ns]')
print(df.dtypes)

# Convert 'release_year' to integer type
df['release_year'] = df['release_year'].astype(int)
# Reset index to have sequential values starting from 0
df.reset_index(drop=True, inplace=True)

# Now, update the index to start from 1 instead of 0
df.index = df.index + 1

#saving all the changes into csv file 
df.to_csv('netflix_movies.csv', index=False)

# Droping any columns which are unused
df.drop(['show_id', 'description'], axis=1, inplace=True)
print(df.head(5))
# Reviewing the data types after changes
print(df['type'].value_counts())
print(df['rating'].value_counts())

df.to_csv('netflix_movies_cleaned.csv', index=False)
# Plotting the distribution of movie ratings\
plt.figure(figsize=(12,6))
sns.countplot(data=df, x='rating', order=df['rating'].value_counts().index)
plt.xticks(rotation=45)
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')

plt.tight_layout()
plt.show()

# Count of Reloease per year
plt.figure(figsize=(12,6))
sns.countplot(data=df, x='release_year', order=df['release_year'].value_counts().index)
plt.xticks(rotation=50)
plt.title('Count of Releases per Year')
plt.xlabel('Release Year')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Duration Trends
plt.figure(figsize=(12,6))
sns.countplot(data=df, x='duration', order=df['duration'].value_counts().index)
plt.xticks(rotation=65)
plt.title('Duration trends')
plt.xlabel('count')
plt.ylabel('Country')
plt.tight_layout()
plt.show()
# Count of list is
plt.figure(figsize=(12,6))
plt.scatter(df['listed_in'], df['duration'], alpha=0.5,color='r')
plt.title('Count of Listed In')
plt.xlabel('show_id')
plt.ylabel('type')
plt.show()

# Count of Movies and TV Shows
plt.figure(figsize=(12,6))
plt.hist(df['type'], bins=30, color='blue', alpha=0.7)
plt.xticks(rotation=0)
plt.title('Count of Movies and TV Shows')
plt.xlabel('Type')
plt.ylabel('Count')
plt.tight_layout()
plt.show()