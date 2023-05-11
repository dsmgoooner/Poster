# -*- coding: utf-8 -*-
"""
Created on Wed May 10 03:55:53 2023

@author: dpsma
"""

# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit

# Define a function to clean the World Bank data
def clean_data(filename):
    # Read the CSV file and set 'Country Name' and 'Indicator Name' as index
    df = pd.read_csv(filename, index_col=['Country Name', 'Indicator Name'])
    # Drop the unnecessary columns 'Country Code' and 'Indicator Code'
    df = df.drop(columns=['Country Code', 'Indicator Code'])
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(method='ffill', axis=0).fillna(method='bfill', axis=0)
    # Reset the index of the DataFrame
    df = df.reset_index()
    return df

# Call the function to clean the data and store it in a variable
world_bank_data = clean_data('world_bank_data.csv')

# Transpose the data
transposed_data = world_bank_data.set_index(['Country Name', 
                                             'Indicator Name']).transpose()
print(transposed_data)

# Define the indicators of interest
co2_emissions = 'CO2 emissions (kt)'
urban_population = 'Urban population'

# Get the statistics for the selected indicators
co2_emissions_stats = world_bank_data[world_bank_data['Indicator Name']
                                      == co2_emissions]
urban_population_stats = world_bank_data[world_bank_data['Indicator Name']
                                         == urban_population]


# Normalize the data
selected_data = pd.concat([co2_emissions_stats, urban_population_stats],
                          axis=0).pivot(index='Country Name',
                                    columns='Indicator Name', values='2018')
normalized_data = (selected_data - selected_data.mean()) / selected_data.std()

# Perform clustering using k-means
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
clusters = kmeans.fit_predict(normalized_data)

def quadratic_func(x, a, b, c):
    return a*x**2 + b*x + c

# Fit the curve to the data
indicators_of_interest = ['CO2 emissions (kt)', 'Urban population']
x_data = normalized_data[indicators_of_interest[0]].values
y_data = normalized_data[indicators_of_interest[1]].values
popt, _ = curve_fit(quadratic_func, x_data, y_data)


# Plot the cluster membership and cluster centers
plt.figure(figsize=(10, 6))
colors = ['r', 'g', 'b', 'c']
for i in range(n_clusters):
    cluster_points = normalized_data.iloc[clusters == i, :]
    plt.scatter(cluster_points[indicators_of_interest[0]],
                cluster_points[indicators_of_interest[1]],
                color=colors[i], label=f'Cluster {i+1}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            color='k', marker='X', label='Cluster Centers')
plt.title('Clusters of CO2 Emissions and urban_population')
plt.xlabel('Normalized CO2 emissions per capita')
plt.ylabel('Normalized urban_population')
plt.legend()

# Plot the fitted curve using the quadratic function
plt.figure(figsize=(10, 6))
plt.scatter(normalized_data[indicators_of_interest[0]],
            normalized_data[indicators_of_interest[1]],
            color='b', label='Data Points')
plt.plot(x_data, quadratic_func(x_data, *popt), color='r', label='Curve Fit')
plt.title('Curve Fitting of CO2 Emissions and urban_population')
plt.xlabel('Normalized CO2 emissions per capita')
plt.ylabel('Normalized urban_population')
plt.legend()

plt.show()



