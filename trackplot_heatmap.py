import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# read the data into a pandas dataframe
df = pd.read_csv('_x_mm_y_mm.csv')

# Get the number of individuals from the column names
num_individuals = len(df.columns) // 2

# Initialize a list to hold the individual DataFrames
individual_dfs = []

# Iterate through the individuals and create a DataFrame for each one
for i in range(num_individuals):
    # Get the column names for the x and y coordinates for this individual
    x_col_name = 'x_mm_' + str(i+1)
    y_col_name = 'y_mm_' + str(i+1)
    
    # Create a DataFrame for this individual's coordinates
    individual_df = pd.DataFrame({
        'x': df[x_col_name],
        'y': df[y_col_name]
    })
    
    # Append the individual DataFrame to the list
    individual_dfs.append(individual_df)

# Concatenate the individual DataFrames into a single DataFrame
all_df = pd.concat(individual_dfs)
all_df.dropna(inplace=True)







# Define the number of bins
square_side_len = 5      #in mm

x_bins = int((all_df['x'].max() - all_df['x'].min()) / square_side_len)
y_bins = int((all_df['y'].max() - all_df['y'].min()) / square_side_len)

num_bins = (x_bins, y_bins)

# Create a 2D histogram with the specified number of bins
hist, xedges, yedges = np.histogram2d(all_df['x'], all_df['y'], bins=num_bins)


# Normalize the histogram by dividing each bin by the total number of data points and the bin area
#hist_norm = hist/(len(all_df['x'])*bin_area)
hist_norm = hist/(np.sum(hist))


# Plot the heatmap Note: must be transposed because np.histogram2d does not follow normal cartesian convention
plt.imshow(hist_norm.T, cmap='hot', interpolation='nearest', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower')
plt.colorbar()

# Set the axis labels
plt.xlabel('X')
plt.ylabel('Y')

# Show the plot
plt.show()






# Create the figure and 3D axes
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create the X, Y meshgrid with reversed x-axis limits
X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

# Plot the 3D surface  Note: must be transposed because np.histogram2d does not follow normal cartesian convention
ax.plot_surface(X, Y, hist_norm.T, cmap='hot')

# Set the axis labels
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Frequency')

# Show the plot
plt.show()