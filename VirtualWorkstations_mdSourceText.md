# Estimating Virtual Station's Location

### Authenticate and initialize Earth Engine


```python
import ee
ee.Authenticate()
ee.Initialize()
```


<p>To authorize access needed by Earth Engine, open the following
        URL in a web browser and follow the instructions:</p>
        <p><a href=https://code.earthengine.google.com/client-auth?scopes=https%3A//www.googleapis.com/auth/earthengine%20https%3A//www.googleapis.com/auth/devstorage.full_control&request_id=WhZeu8jj8bGOgTlSLFtV7eeOiHBQV45CEaUEat0PnrU&tc=Zb1z0tFm9icecR7_huEFwUvXjuBC6owBjjX26BaJMTk&cc=wmDGbZN7XtUK-0mL_uO5uctXjeCW7q35c8VXYfL8ASI>https://code.earthengine.google.com/client-auth?scopes=https%3A//www.googleapis.com/auth/earthengine%20https%3A//www.googleapis.com/auth/devstorage.full_control&request_id=WhZeu8jj8bGOgTlSLFtV7eeOiHBQV45CEaUEat0PnrU&tc=Zb1z0tFm9icecR7_huEFwUvXjuBC6owBjjX26BaJMTk&cc=wmDGbZN7XtUK-0mL_uO5uctXjeCW7q35c8VXYfL8ASI</a></p>
        <p>The authorization workflow will generate a code, which you should paste in the box below.</p>



    Enter verification code: 4/1Adeu5BWZQWn0p6Sw5M5kWdg8Y2ueejxY8W8Yrlg-lYUc5oR5PXzKnzxuRdI
    
    Successfully saved authorization token.
    

### Import necessary libraries and packages


```python
import numpy as np
import pandas as pd

import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt

from scipy.ndimage import sobel
from skimage.feature import peak_local_max
from scipy.ndimage import distance_transform_edt
```

### Get the Earth Engine's DEM elevation band information in a dataframe

We use the Earth Engine Python API to work with satellite data by loading the SRTM (Shuttle Radar Topography Mission) dataset 
and selecting the 'elevation' band from it.


```python
# Define the SRTM dataset
srtm = ee.Image('CGIAR/SRTM90_V4')

# Select the elevation band
elevation_of_dem = srtm.select('elevation')

# Get the elevation band information
elevation_info = elevation_of_dem.getInfo()

# Flatten the nested dictionaries
flatten_info = {}
for key, value in elevation_info.items():
    if isinstance(value, dict):
        for inner_key, inner_value in value.items():
            flatten_info[f'{key}_{inner_key}'] = inner_value
    else:
        flatten_info[key] = value

# Create a DataFrame from the flattened information
df = pd.DataFrame.from_dict(flatten_info, orient='index', columns=['Value'])

# Set pandas display options
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

# Reset the index and rename the column
df.reset_index(inplace=True)
df.rename(columns={'index': 'Key'}, inplace=True)

# Display the DataFrame
df

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Key</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>type</td>
      <td>Image</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bands</td>
      <td>[{'id': 'elevation', 'data_type': {'type': 'PixelType', 'precision': 'int', 'min': -32768, 'max': 32767}, 'dimensions': [432000, 144000], 'crs': 'EPSG:4326', 'crs_transform': [0.000833333333333, 0, -180, 0, -0.000833333333333, 60]}]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>id</td>
      <td>CGIAR/SRTM90_V4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>version</td>
      <td>1641990053291277</td>
    </tr>
    <tr>
      <th>4</th>
      <td>properties_system:visualization_0_min</td>
      <td>-100.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>properties_type_name</td>
      <td>Image</td>
    </tr>
    <tr>
      <th>6</th>
      <td>properties_keywords</td>
      <td>[cgiar, dem, elevation, geophysical, srtm, topography]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>properties_thumb</td>
      <td>https://mw1.google.com/ges/dd/images/SRTM90_V4_thumb.png</td>
    </tr>
    <tr>
      <th>8</th>
      <td>properties_description</td>
      <td>&lt;p&gt;The Shuttle Radar Topography Mission (SRTM) digital\nelevation dataset was originally produced to provide consistent,\nhigh-quality elevation data at near global scope. This version\nof the SRTM digital elevation data has been processed to fill data\nvoids, and to facilitate its ease of use.&lt;/p&gt;&lt;p&gt;&lt;b&gt;Provider: &lt;a href="https://srtm.csi.cgiar.org/"&gt;NASA/CGIAR&lt;/a&gt;&lt;/b&gt;&lt;br&gt;&lt;p&gt;&lt;b&gt;Bands&lt;/b&gt;&lt;table class="eecat"&gt;&lt;tr&gt;&lt;th scope="col"&gt;Name&lt;/th&gt;&lt;th scope="col"&gt;Description&lt;/th&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;elevation&lt;/td&gt;&lt;td&gt;&lt;p&gt;Elevation&lt;/p&gt;&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;p&gt;&lt;b&gt;Terms of Use&lt;/b&gt;&lt;br&gt;&lt;p&gt;DISTRIBUTION. Users are prohibited from any commercial, non-free resale, or\nredistribution without explicit written permission from CIAT. Users should\nacknowledge CIAT as the source used in the creation of any reports,\npublications, new datasets, derived products, or services resulting from the\nuse of this dataset. CIAT also request reprints of any publications and\nnotification of any redistributing efforts. For commercial access to\nthe data, send requests to &lt;a href="mailto:a.jarvis@cgiar.org"&gt;Andy Jarvis&lt;/a&gt;.&lt;/p&gt;&lt;p&gt;NO WARRANTY OR LIABILITY. CIAT provides these data without any warranty of\nany kind whatsoever, either express or implied, including warranties of\nmerchantability and fitness for a particular purpose. CIAT shall not be\nliable for incidental, consequential, or special damages arising out of\nthe use of any data.&lt;/p&gt;&lt;p&gt;ACKNOWLEDGMENT AND CITATION. Any users are kindly asked to cite this data\nin any published material produced using this data, and if possible link\nweb pages to the &lt;a href="https://srtm.csi.cgiar.org"&gt;CIAT-CSI SRTM website&lt;/a&gt;.&lt;/p&gt;&lt;p&gt;&lt;b&gt;Suggested citation(s)&lt;/b&gt;&lt;ul&gt;&lt;li&gt;&lt;p&gt;Jarvis, A., H.I. Reuter, A. Nelson, E. Guevara. 2008. Hole-filled\nSRTM for the globe Version 4, available from the CGIAR-CSI SRTM\n90m Database: https://srtm.csi.cgiar.org.&lt;/p&gt;&lt;/li&gt;&lt;/ul&gt;&lt;style&gt;\n  table.eecat {\n  border: 1px solid black;\n  border-collapse: collapse;\n  font-size: 13px;\n  }\n  table.eecat td, tr, th {\n  text-align: left; vertical-align: top;\n  border: 1px solid gray; padding: 3px;\n  }\n  td.nobreak { white-space: nowrap; }\n&lt;/style&gt;</td>
    </tr>
    <tr>
      <th>9</th>
      <td>properties_source_tags</td>
      <td>[cgiar]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>properties_visualization_0_max</td>
      <td>8000.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>properties_title</td>
      <td>SRTM Digital Elevation Data Version 4</td>
    </tr>
    <tr>
      <th>12</th>
      <td>properties_product_tags</td>
      <td>[srtm, elevation, topography, dem, geophysical]</td>
    </tr>
    <tr>
      <th>13</th>
      <td>properties_provider</td>
      <td>NASA/CGIAR</td>
    </tr>
    <tr>
      <th>14</th>
      <td>properties_visualization_0_min</td>
      <td>-100.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>properties_visualization_0_name</td>
      <td>Elevation</td>
    </tr>
    <tr>
      <th>16</th>
      <td>properties_date_range</td>
      <td>[950227200000, 951177600000]</td>
    </tr>
    <tr>
      <th>17</th>
      <td>properties_system:visualization_0_gamma</td>
      <td>1.6</td>
    </tr>
    <tr>
      <th>18</th>
      <td>properties_period</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>properties_system:visualization_0_bands</td>
      <td>elevation</td>
    </tr>
    <tr>
      <th>20</th>
      <td>properties_system:time_end</td>
      <td>951177600000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>properties_provider_url</td>
      <td>https://srtm.csi.cgiar.org/</td>
    </tr>
    <tr>
      <th>22</th>
      <td>properties_sample</td>
      <td>https://mw1.google.com/ges/dd/images/SRTM90_V4_sample.png</td>
    </tr>
    <tr>
      <th>23</th>
      <td>properties_tags</td>
      <td>[cgiar, dem, elevation, geophysical, srtm, topography]</td>
    </tr>
    <tr>
      <th>24</th>
      <td>properties_system:time_start</td>
      <td>950227200000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>properties_system:visualization_0_max</td>
      <td>8000.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>properties_system:visualization_0_name</td>
      <td>Elevation</td>
    </tr>
    <tr>
      <th>27</th>
      <td>properties_system:asset_size</td>
      <td>18827626666</td>
    </tr>
    <tr>
      <th>28</th>
      <td>properties_visualization_0_bands</td>
      <td>elevation</td>
    </tr>
  </tbody>
</table>
</div>



### Display the SRTM DEM dataset's info obtained from USGS (United States Geological Survey)  


```python
import rasterio
import pandas as pd

# Load the DEM GeoTIFF file
dem_file = "C:/Users/HP/Downloads/n25_e000_1arc_v3.tif"

dem = rasterio.open(dem_file)

# Get the metadata of the DEM
metadata = dem.meta

# Create a DataFrame from the metadata
df = pd.DataFrame.from_dict(metadata, orient='index', columns=['Value'])

# Set pandas display options
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

# Reset the index and rename the column
df.reset_index(inplace=True)
df.rename(columns={'index': 'Key'}, inplace=True)

# Display the DataFrame
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Key</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>driver</td>
      <td>GTiff</td>
    </tr>
    <tr>
      <th>1</th>
      <td>dtype</td>
      <td>int16</td>
    </tr>
    <tr>
      <th>2</th>
      <td>nodata</td>
      <td>-32767.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>width</td>
      <td>3601</td>
    </tr>
    <tr>
      <th>4</th>
      <td>height</td>
      <td>3601</td>
    </tr>
    <tr>
      <th>5</th>
      <td>count</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>crs</td>
      <td>(init)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>transform</td>
      <td>(0.0002777777777777778, 0.0, -0.0001388888888888889, 0.0, -0.0002777777777777778, 26.00013888888889, 0.0, 0.0, 1.0)</td>
    </tr>
  </tbody>
</table>
</div>



### Print the elevation array of the Earth Engine's DEM dataset for Ganga River


```python
# Define the bounding box coordinates [minX, minY, maxX, maxY] for Ganga River in Ayodhya, India
bounding_box = [82.12, 25.72, 82.18, 25.78]

# Create a geometry from the bounding box coordinates
geometry = ee.Geometry.Rectangle(bounding_box)

# Get the pixel values as an array
elevation_values = elevation_of_dem.reduceRegion(
    reducer=ee.Reducer.toList(),
    geometry=geometry,
    scale=10,
    maxPixels=1e9
).get('elevation')

# Convert the pixel values to a NumPy array
elevation_array = np.array(elevation_values.getInfo())

# Set the NumPy print options to display all array values
np.set_printoptions(threshold=np.inf)

# Print the elevation array
print(elevation_array[:100])

```

    [98 98 98 98 98 98 98 98 98 98 97 97 97 97 97 97 97 97 97 97 97 97 97 97
     97 97 97 97 97 97 97 97 97 97 97 97 97 97 94 94 94 94 94 94 94 94 94 96
     96 96 96 96 96 96 96 96 97 97 97 97 97 97 97 97 97 98 98 98 98 98 98 98
     98 98 98 96 96 96 96 96 96 96 96 96 98 98 98 98 98 98 98 98 98 99 99 99
     99 99 99 99]
    

### Plot the Elevation Profile of the Earth Engine's elevation dataset and draw inferences


```python
# Plot the elevation data
plt.figure(figsize=(10, 5))
plt.plot(elevation_array)
#plt.title('Elevation Profile')
#plt.xlabel('Pixel Index')
#plt.ylabel('Elevation')
plt.grid(True)
plt.show()

# Calculate statistics
min_elevation = np.min(elevation_array)
max_elevation = np.max(elevation_array)
mean_elevation = np.mean(elevation_array)
median_elevation = np.median(elevation_array)
std_deviation = np.std(elevation_array)

# Print the statistics
print('Minimum Elevation:', min_elevation)
print('Maximum Elevation:', max_elevation)
print('Mean Elevation:', mean_elevation)
print('Median Elevation:', median_elevation)
print('Standard Deviation:', std_deviation)

```


    
![png](output_11_0.png)
    


    Minimum Elevation: 88
    Maximum Elevation: 107
    Mean Elevation: 96.17562479830758
    Median Elevation: 96.0
    Standard Deviation: 2.3676893157611203
    

### Plot the Heatmap and 3D Surface of the Earth Engine's elevation dataset and draw inferences


```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the bounding box coordinates for Ganga River in Ayodhya, India
bounding_box = [82.12, 25.72, 82.18, 25.78]

# Calculate the number of rows and columns based on the square root of the elevation array size
n = int(np.sqrt(elevation_array.size))
nrows = ncols = n

# Reshape the elevation array into a matrix and convert it to float
elevation_matrix = elevation_array.reshape((nrows, ncols)).astype(float)

# Plotting the Heatmap
plt.figure(figsize=(8, 6))
plt.imshow(elevation_matrix, cmap='terrain', origin='lower', extent=bounding_box)
plt.colorbar(label='Elevation (m)')
plt.title('Elevation Heatmap')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Plotting the 3D Surface
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
x = np.linspace(bounding_box[0], bounding_box[2], ncols)
y = np.linspace(bounding_box[1], bounding_box[3], nrows)
X, Y = np.meshgrid(x, y)
Z = elevation_matrix
ax.plot_surface(X, Y, Z, cmap='terrain')
ax.set_title('3D Surface Plot of Elevation')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Elevation (m)')
plt.show()

```


    
![png](output_13_0.png)
    



    
![png](output_13_1.png)
    


### Apply Gaussian filter to smoothen the Earth Engine's DEM and visualize the results


```python
# Reshape the elevation array to a 2D grid
length = len(elevation_array)
rows = int(np.sqrt(length))
cols = int(length / rows)
elevation_grid = elevation_array.reshape((rows, cols))

# Apply Gaussian filter to smooth the DEM
smoothed_dem = scipy.ndimage.gaussian_filter(elevation_grid, sigma=1)

# Visualize the original and smoothed DEM
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(elevation_grid, cmap='terrain')
plt.title('Original DEM')
plt.subplot(1, 2, 2)
plt.imshow(smoothed_dem, cmap='terrain')
plt.title('Smoothed DEM')
plt.show()

```


    
![png](output_15_0.png)
    


### Apply Mean filter to smoothen the Earth Engine's DEM and visualize the results

We use "uniform_filter" function from scipy.ndimage to apply a mean filter with a specified size (2*radius+1). Adjust the radius parameter as needed to control the amount of smoothing.


```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

# Calculate the number of rows and columns based on the square root of the elevation array size
n = int(np.sqrt(elevation_array.size))
nrows = ncols = n

# Reshape the elevation array into a matrix
elevation_matrix = elevation_array.reshape((nrows, ncols))

# Apply mean filter
radius = 2  
smoothed_elevation = uniform_filter(elevation_matrix, size=2*radius+1)

# Plotting the original and smoothed elevation data
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(elevation_matrix, cmap='terrain', origin='lower', extent=bounding_box)
axes[0].set_title('Original Elevation')
axes[0].set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')

axes[1].imshow(smoothed_elevation, cmap='terrain', origin='lower', extent=bounding_box)
axes[1].set_title('Smoothed Elevation')
axes[1].set_xlabel('Longitude')
axes[1].set_ylabel('Latitude')

plt.tight_layout()
plt.show()
```


    
![png](output_17_0.png)
    


### Apply Bilateral filter to smoothen the Earth Engine's DEM and visualize the results


```python
from skimage import img_as_float
from skimage.restoration import denoise_bilateral

# Assuming you have the DEM stored in the variable 'elevation_array'

# Calculate the number of rows and columns based on the square root of the elevation array size
n = int(np.sqrt(elevation_array.size))
nrows = ncols = n

# Reshape the elevation array into a matrix
elevation_matrix = elevation_array.reshape((nrows, ncols))

# Convert elevation matrix to float between 0 and 1
elevation_float = img_as_float(elevation_matrix)

# Apply bilateral filter for edge-preserving smoothing
sigma_color = 0.1    # Controls the range similarity
sigma_spatial = 5    # Controls the spatial smoothing
smoothed_elevation = denoise_bilateral(elevation_float, sigma_color=sigma_color, sigma_spatial=sigma_spatial)

# Set up the figure and axes
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot the original DEM
axs[0].imshow(elevation_matrix, cmap='terrain')
axs[0].set_title('Original DEM')
axs[0].axis('off')

# Plot the smoothed DEM
axs[1].imshow(smoothed_elevation, cmap='viridis')
axs[1].set_title('Smoothed DEM')
axs[1].axis('off')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the figures
plt.show()

```


    
![png](output_19_0.png)
    


### Compute and visualize the gradient magnitude of the smoothed Earth Engine's DEM


```python
# Compute gradient magnitude 
gradient_magnitude1 = np.sqrt(sobel(smoothed_elevation, axis=0)**2 + sobel(smoothed_dem, axis=1)**2)
gradient_magnitude2 = np.sqrt(sobel(smoothed_dem, axis=0)**2 + sobel(smoothed_dem, axis=1)**2)

# Visualize gradient magnitude
plt.figure(figsize=(10, 6))
plt.imshow(gradient_magnitude1, cmap='hot')
plt.title('Gradient Magnitude of Smoothed DEM using Mean Filter')
plt.colorbar()
plt.show()

# Visualize gradient magnitude
plt.figure(figsize=(10, 6))
plt.imshow(gradient_magnitude2, cmap='hot')
plt.title('Gradient Magnitude of Smoothed DEM using Gaussian Filter')
plt.colorbar()
plt.show()
```


    
![png](output_21_0.png)
    



    
![png](output_21_1.png)
    


### Compute and visualize the gradient direction of the smoothed Earth Engine's DEM


```python
from scipy.ndimage import uniform_filter

# Calculate the number of rows and columns based on the square root of the elevation array size
n = int(np.sqrt(elevation_array.size))
nrows = ncols = n

# Reshape the elevation array into a matrix
elevation_matrix = elevation_array.reshape((nrows, ncols))

# Apply mean filter
radius = 2
smoothed_elevation = uniform_filter(elevation_matrix, size=2*radius+1)

# Compute the gradient direction of the smoothed DEM
dx = np.gradient(smoothed_elevation, axis=1)
dy = np.gradient(smoothed_elevation, axis=0)
gradient_direction = np.arctan2(dy, dx)

# Set up the figure and axis for arrow visualization
fig, ax = plt.subplots(figsize=(8, 8))

# Plot the gradient direction with arrows
X, Y = np.meshgrid(np.arange(ncols), np.arange(nrows))
U = np.cos(gradient_direction)
V = np.sin(gradient_direction)
ax.quiver(X, Y, U, V, color='black', scale=10, units='xy')

# Set labels for the arrow map
ax.set_title('Arrow Map of Gradient Direction')

# Adjust spacing between subplots
plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.3)

# Set the color map and normalize the gradient direction for the colorbar
cmap = plt.cm.hsv
normalized_direction = (gradient_direction + np.pi) / (2 * np.pi)

# Set up the figure and axis for the gradient direction map
fig, ax = plt.subplots(figsize=(8, 8))

# Plot the normalized gradient direction map
image = ax.imshow(normalized_direction, cmap=cmap)
cbar = plt.colorbar(image, ax=ax, orientation='vertical', pad=0.05)
cbar.set_label('Normalized Gradient Direction')

# Set labels for the gradient direction map
ax.set_title('Gradient Direction Map')

# Show the figures
plt.show()

```


    
![png](output_23_0.png)
    



    
![png](output_23_1.png)
    


### Compute and visualize the respective histograms of each colour and all colours combined of the smoothed Earth Engine's DEM


```python
# Get the red, green, and blue color channels
red_channel = gradient_direction[:, :]
green_channel = gradient_direction[:, :]
blue_channel = gradient_direction[:, :]

# Set up the figure and axes for the histograms
fig, axs = plt.subplots(3, 1, figsize=(8, 8))

# Plot histogram for the red channel
axs[0].hist(red_channel.flatten(), bins=256, color='red', alpha=0.7)
axs[0].set_title('Histogram - Red Channel')

# Plot histogram for the green channel
axs[1].hist(green_channel.flatten(), bins=256, color='green', alpha=0.7)
axs[1].set_title('Histogram - Green Channel')

# Plot histogram for the blue channel
axs[2].hist(blue_channel.flatten(), bins=256, color='blue', alpha=0.7)
axs[2].set_title('Histogram - Blue Channel')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the histograms
plt.show()

```


    
![png](output_25_0.png)
    



```python
# Flatten the gradient direction map
gradient_direction_flat = gradient_direction.flatten()

# Set up the figure and axes for the histogram
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the histogram of the gradient direction values
ax.hist(gradient_direction_flat, bins=256, color='blue', alpha=0.7)
ax.set_title('Histogram - Gradient Direction')
ax.set_xlabel('Gradient Direction')
ax.set_ylabel('Frequency')

# Show the histogram
plt.show()

```


    
![png](output_26_0.png)
    


### Compute and visualize the gradient magnitude, gradient direction and the respective histograms of the USGS DEM


```python
from scipy.ndimage import gaussian_filter
from skimage import io

# Load the DEM from the TIFF file
dem = io.imread("C:/Users/HP/Downloads/n25_e000_1arc_v3.tif")

# Apply Gaussian filter for smoothing
sigma = 1.5  # Controls the level of smoothing
smoothed_dem = gaussian_filter(dem, sigma=sigma)

# Compute the gradient using Sobel operators
dx = np.gradient(smoothed_dem, axis=1)
dy = np.gradient(smoothed_dem, axis=0)

# Compute the gradient magnitude
gradient_magnitude = np.sqrt(dx**2 + dy**2)

# Compute the gradient direction
gradient_direction = np.arctan2(dy, dx)

# Set up the figure and axes
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot the gradient direction map
cmap_direction = 'hsv'
im_direction = axs[0, 0].imshow(gradient_direction, cmap=cmap_direction)
#axs[0, 0].set_title('Gradient Direction')
axs[0, 0].axis('off')
cbar_direction = fig.colorbar(im_direction, ax=axs[0, 0], orientation='vertical')
#cbar_direction.set_label('Direction')

# Compute and plot histogram for the gradient direction map
direction_hist, direction_bins = np.histogram(gradient_direction, bins=256, range=(-np.pi, np.pi))
axs[1, 0].plot(direction_bins[:-1], direction_hist, color='black')
axs[1, 0].set_title('Gradient Direction Histogram')
axs[1, 0].set_xlabel('Gradient Direction (radians)')
axs[1, 0].set_ylabel('Frequency')

# Plot the gradient magnitude map
cmap_magnitude = 'jet'
im_magnitude = axs[0, 1].imshow(gradient_magnitude, cmap=cmap_magnitude)
#axs[0, 1].set_title('Gradient Magnitude')
axs[0, 1].axis('off')
cbar_magnitude = fig.colorbar(im_magnitude, ax=axs[0, 1], orientation='vertical')
#cbar_magnitude.set_label('Magnitude')

# Compute and plot histogram for the gradient magnitude map
magnitude_hist, magnitude_bins = np.histogram(gradient_magnitude, bins=256, range=(0, gradient_magnitude.max()))
axs[1, 1].plot(magnitude_bins[:-1], magnitude_hist, color='black')
axs[1, 1].set_title('Gradient Magnitude Histogram')
axs[1, 1].set_xlabel('Gradient Magnitude')
axs[1, 1].set_ylabel('Frequency')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the figure
plt.show()

```


    
![png](output_28_0.png)
    


### Compute and visualize the gradient magnitude, gradient direction and the respective histograms of the hilly terrain 


```python
from scipy.ndimage import gaussian_filter
from skimage import io

# Load the DEM from the TIFF file
dem = io.imread("C:/Users/HP/OneDrive/Desktop/Surge Project/Kangra_Chamba_Hills.tif")

# Apply Gaussian filter for smoothing
sigma = 1.5  # Controls the level of smoothing
smoothed_dem = gaussian_filter(dem, sigma=sigma)

# Compute the gradient using Sobel operators
dx = np.gradient(smoothed_dem, axis=1)
dy = np.gradient(smoothed_dem, axis=0)

# Compute the gradient magnitude
gradient_magnitude = np.sqrt(dx**2 + dy**2)

# Compute the gradient direction
gradient_direction = np.arctan2(dy, dx)

# Set up the figure and axes
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot the gradient direction map
cmap_direction = 'hsv'
im_direction = axs[0, 0].imshow(gradient_direction, cmap=cmap_direction)
axs[0, 0].set_title('Gradient Direction')
axs[0, 0].axis('off')
cbar_direction = fig.colorbar(im_direction, ax=axs[0, 0], orientation='vertical')
cbar_direction.set_label('Direction')

# Compute and plot histogram for the gradient direction map
direction_hist, direction_bins = np.histogram(gradient_direction, bins=256, range=(-np.pi, np.pi))
axs[1, 0].plot(direction_bins[:-1], direction_hist, color='black')
axs[1, 0].set_title('Gradient Direction Histogram')
axs[1, 0].set_xlabel('Gradient Direction (radians)')
axs[1, 0].set_ylabel('Frequency')

# Plot the gradient magnitude map
cmap_magnitude = 'hot'
im_magnitude = axs[0, 1].imshow(gradient_magnitude, cmap=cmap_magnitude)
axs[0, 1].set_title('Gradient Magnitude')
axs[0, 1].axis('off')
cbar_magnitude = fig.colorbar(im_magnitude, ax=axs[0, 1], orientation='vertical')
cbar_magnitude.set_label('Magnitude')

# Compute and plot histogram for the gradient magnitude map
magnitude_hist, magnitude_bins = np.histogram(gradient_magnitude, bins=256, range=(0, gradient_magnitude.max()))
axs[1, 1].plot(magnitude_bins[:-1], magnitude_hist, color='black')
axs[1, 1].set_title('Gradient Magnitude Histogram')
axs[1, 1].set_xlabel('Gradient Magnitude')
axs[1, 1].set_ylabel('Frequency')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the figure
plt.show()
```


    
![png](output_30_0.png)
    


### Visualize the gradient magnitude image with the identified channel locations marked as blue dots for Earth Engine's DEM 

1. peak_local_max function is called with parameters min_distance=10 and num_peaks=2. 
   These parameters control the minimum distance between identified peaks and the maximum number of peaks to identify.
2. The resulting channel locations are plotted on top of the gradient magnitude image using blue dots ('bo'). 
   This visualization helps you visualize the identified channel locations in relation to the terrain's gradient. 
3. Plot the gradient magnitude using the imshow function from matplotlib.pyplot. The colormap 'hot' is used to emphasize areas of stronger gradients, which can be indicative of significant changes in elevation or proximity to the river channel.   


```python
# Identify potential river channel locations using local maxima
channel_locations = peak_local_max(gradient_magnitude1, min_distance=10, num_peaks=2)

# Visualize the gradient magnitude with channel locations
plt.figure(figsize=(10, 6))
plt.imshow(gradient_magnitude1, cmap='summer')  # Change the colormap to 'viridis'
plt.plot(channel_locations[:, 1], channel_locations[:, 0], 'ko')
#plt.title('Gradient Magnitude with Channel Locations')
plt.colorbar()
plt.show()

```


    
![png](output_32_0.png)
    



```python
# Identify potential river channel locations using local maxima
channel_locations = peak_local_max(gradient_magnitude1, min_distance=10, num_peaks=2)

# Visualize the gradient magnitude with channel locations
plt.figure(figsize=(10, 6))
plt.imshow(gradient_magnitude1, cmap='hot')
plt.plot(channel_locations[:, 1], channel_locations[:, 0], 'bo')
plt.title('Gradient Magnitude with Channel Locations')
plt.colorbar()
plt.show()
```


    
![png](output_33_0.png)
    


### Compute and visualize the distance transform for each channel location for Earth Engine's DEM

1. Iterate over each channel location (loc) in the channel_locations array.
2. Extract the coordinates of the channel location (y and x).
3. Create a binary mask by thresholding the gradient magnitude image based on the current channel location. 
   The thresholding operation (gradient_magnitude > gradient_magnitude[y, x]) creates a binary image where True values represent pixels with gradient magnitudes higher than the magnitude at the current channel location.
4. Compute the distance transform using the distance_transform_edt function from the scipy.ndimage module. 
   The distance transform calculates the Euclidean distance from each pixel to the nearest False (zero) pixel in the binary mask. 
   In this case, it calculates the distance to the nearest pixel outside the region of interest around the river channel.
5. Append the distance transform image (distance_transform) to the distance_transforms list.
6. Plot the distance transform using the imshow function from matplotlib.pyplot. The colormap 'hot' is used to emphasize areas of larger distances, which can be indicative of significant changes in elevation or proximity to the river channel.


```python
# Compute distance transform 
distance_transforms = []
for loc in channel_locations:
    y, x = loc  # Extract coordinates of the channel location
    distance_transform = distance_transform_edt(np.logical_not(gradient_magnitude1 > gradient_magnitude1[y, x]))
    distance_transforms.append(distance_transform)

# Visualize distance transforms
plt.figure(figsize=(10, 6))
for dt in distance_transforms:
    plt.imshow(dt, cmap='hot')
    plt.colorbar()
    plt.title('Distance Transform')
    plt.show()

```


    
![png](output_35_0.png)
    



    
![png](output_35_1.png)
    


### Estimate the river width as the mean of maximum distance from each channel location for Earth Engine's DEM

1. Iterate over each distance transform image (dt) in the distance_transforms list.
2. Calculate the maximum distance value in each distance transform image using np.max(dt). 
   This value represents the maximum distance from any pixel in the image to the nearest river channel.
3. Multiply the maximum distance by 2 to account for both sides of the river and obtain an estimate of the river width for each distance transform image (2 * np.max(dt)).
4. Create a list comprehension to iterate over all the river width estimates and calculate their mean using np.mean(). 
   This provides an average estimate of the river width based on the maximum distances from each channel location.
5. Assign the mean river width value to the variable river_width.


```python
# Estimate river width 
river_width = np.mean([2 * np.max(dt) for dt in distance_transforms])

# Print the estimated river width
print("Estimated river width:", river_width)

```

    Estimated river width: 1606.5706427293383
    

### Project the smoothened DEM onto the river section plane

1. Compute the gradient in x, y directions of the smoothed DEM using the sobel function from the scipy.ndimage module. 
2. Compute the orientation angle at each pixel by applying the arctan2 function to the gradient_y and gradient_x arrays. 
   This will give you the angle of the steepest ascent/descent at each pixel. 
3. Compute the perpendicular orientation to the river channel by adding π/2 radians to each orientation value and taking the        modulo π. This effectively rotates the orientations by 90 degrees counter-clockwise.
4. Determine the pixel coordinates of the cross-section location. We assume the cross-section is in the middle of the river        width and the middle row of the grid.
5. Compute the distances of each pixel from the cross-section location along the perpendicular orientation using the                distance_transform_edt function from scipy.ndimage. This function calculates the Euclidean distance transform.
6. Identify the pixels within the river section by creating a mask based on the distances and the river width. 
   Pixels with distances less than or equal to the river width are considered part of the river section. 
7. Convert the data type of the smoothed DEM to floating-point using the astype method. 
   This is done to ensure compatibility for assigning NaN values in the next step.
8. Project the DEM onto the river section plane by assigning NaN values to pixels outside the river section.  
9. Plot the projected DEM using the imshow function from matplotlib.pyplot. 
   The colormap 'terrain' is used to visualize the elevation values, and the minimum and maximum values of the smoothed DEM are    used for the color scale limits.    


```python
# Compute the gradient in the x and y directions
gradient_x = scipy.ndimage.sobel(smoothed_elevation, axis=1)
gradient_y = scipy.ndimage.sobel(smoothed_elevation, axis=0)

# Compute the orientation angle at each pixel
orientation = np.arctan2(gradient_y, gradient_x)

# Compute the perpendicular orientation to the river channel
perpendicular_orientation = np.mod(orientation + np.pi / 2, np.pi)

# Determine the pixel coordinates of the cross-section location
cross_section_x = int(river_width / 2)  # Assuming the cross-section is in the middle of the river width
cross_section_y = int(rows / 2)  # Assuming the cross-section is in the middle row of the grid

# Compute the distances of each pixel from the cross-section location along the perpendicular orientation
distances = scipy.ndimage.distance_transform_edt(perpendicular_orientation != 0)

# Identify the pixels within the river section by comparing distances with river width
river_section_mask = distances <= river_width

# Convert the data type of smoothed_dem to floating-point
smoothed_elevation = smoothed_elevation.astype(float)

# Project the DEM onto the river section plane by assigning NaN values to pixels outside the river section
dem_projected = np.copy(smoothed_elevation)
dem_projected[~river_section_mask] = np.nan

# Plot the projected DEM
plt.imshow(dem_projected, cmap='terrain', vmin=np.nanmin(smoothed_elevation), vmax=np.nanmax(smoothed_elevation))
plt.colorbar()
plt.title('Projected DEM')
plt.show()
```


    
![png](output_39_0.png)
    


### Compute the latitude, longitude coordinates of the SRTM DEM dataset's channel locations to check and verify their location on map


```python
import rasterio
from skimage.feature import peak_local_max

# Load the drainage network geotiff file
drainage_network_file = "C:/Users/HP/Downloads/n25_e000_1arc_v3.tif"
drainage_network = rasterio.open(drainage_network_file)

# Identify potential river channel locations using local maxima
channel_locations = peak_local_max(gradient_magnitude1, min_distance=10, num_peaks=2)

# Convert pixel coordinates to latitude and longitude
lat_lon_coords = []
for loc in channel_locations:
    lon, lat = drainage_network.xy(loc[0], loc[1])
    lat_lon_coords.append((lat, lon))

# Print the latitude and longitude coordinates
for lat, lon in lat_lon_coords:
    print("Latitude:", lat)
    print("Longitude:", lon)
    print("---")
```

    Latitude: 25.939444444444447
    Longitude: 0.049166666666666664
    ---
    Latitude: 25.973055555555558
    Longitude: 0.08777777777777779
    ---
    

### Compute and print the radar nadir coordinates

1. Filter the Sentinel-1 GRD collection based on the defined geometry (bounding box) and date range.
2. Define a function, get_nadir_location, that takes an image as input and retrieves the nadir location (longitude and latitude) of the image.
3. Use the map function to apply the get_nadir_location function to each image in the filtered collection and retrieve the nadir locations.
4. Convert the nadir locations to a feature collection.
5. Get the nadir location coordinates as a server-side operation using the geometry().coordinates().getInfo() method.
6. Print the nadir locations by iterating over the coordinates and printing the longitude and latitude values


```python
# Define the bounding box coordinates for Ganga river in Ayodhya, India
bounding_box = [82.12, 25.72, 82.18, 25.78]

# Create a geometry from the bounding box coordinates
geometry = ee.Geometry.Rectangle(bounding_box)

# Filter the Sentinel-1 GRD collection based on the bounding box and date range
collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
    .filterBounds(geometry) \
    .filterDate('2019-01-01', '2019-12-31')  # Specify the desired date range

# Define a function to get the nadir location of an image
def get_nadir_location(image):
    lon = ee.Image(image).getNumber('sar:longitude')
    lat = ee.Image(image).getNumber('sar:latitude')
    return ee.Feature(geometry, {'longitude': lon, 'latitude': lat})

# Map over the collection and extract the nadir locations
nadir_locations = collection.map(get_nadir_location)

# Convert the nadir locations to a feature collection
nadir_locations_fc = ee.FeatureCollection(nadir_locations)

# Get the nadir location coordinates as a server-side operation
nadir_locations_info = nadir_locations_fc.geometry().coordinates().getInfo()

# Print the nadir locations
print('Nadir Locations (Longitude, Latitude):')

# Iterate over the coordinates
for coordinates in nadir_locations_info[0]:
    lon, lat = coordinates
    print(lon, lat)
```

    Nadir Locations (Longitude, Latitude):
    82.12 25.72
    82.18 25.72
    82.18 25.78
    82.12 25.78
    82.12 25.72
    

## Using QGIS software, compute flow accumulation and thus extract river drainage network.

### STEP 1: Add SRTM DEM raster layer from the layer panel. Style the layer to give a singleband psedocolour to its elevation band.

![image.png](attachment:image.png)



### STEP 2: Reproject the raster layer to give it Warp Reprojection using Nearest Neighbor resampling method. Make use of  coordinate system according to our location Ayodhya, UP. Change the coordinate unit to deg, min, sec that will be all displayed at the bottom.

![image-2.png](attachment:image-2.png)



### STEP3: Display Filled & Flooded DEM of the reprojected elevation layer using r.terraflow GRASS algorithm and render the result in singleband psedocolor.  Also display the raster histogram.

![image-6.png](attachment:image-6.png)



### STEP4: Compute Flow Direction (MFD) of the reprojected elevation layer using r.terraflow GRASS algorithm and display the result in singleband psedocolor.

![image-7.png](attachment:image-7.png)



### STEP5: Create a Sink Watershed DEM of the reprojected elevation layer using r.terraflow GRASS algorithm and render the result in singleband psedocolor.

![image-4.png](attachment:image-4.png)



### STEP6: Compute Flow Accumulation of the reprojected elevation layer using r.terraflow GRASS algorithm and display the result in 3D grayscale hillshade format.

![image-3.png](attachment:image-3.png)



### STEP7:Compute Topographic Convergence Index of the reprojected elevation layer using r.terraflow GRASS algorithm and render the result in singleband psedocolor. Also display the raster histogram.

![image-5.png](attachment:image-5.png)





### Load the ground track data onto a dataframe, then display the results


```python
import netCDF4 as nc

# Specify the path to the NetCDF file
file_path = "C:/Users/HP/Downloads/enhanced_measurement.nc"

# Open the NetCDF file
dataset = nc.Dataset(file_path)

# Print the header contents
print("NetCDF Header Contents:")
print("-----------------------")
print(dataset)

# Print the attributes of the NetCDF file
print("\nNetCDF Attributes:")
print("--------------------")
for attr_name in dataset.ncattrs():
    print(f"{attr_name}: {getattr(dataset, attr_name)}")
    
# Extract latitude and longitude variables: (lat_01,lon_01), (lat_20_ku,lon_20_ku), (lat_20_c,lon_20_c )
latitude = dataset.variables['lat_20_ku'][:100]   # Display the first 100 entries
longitude = dataset.variables['lon_20_ku'][:100]  # Display the first 100 entries

# Create a DataFrame for the ground track data
ground_track_df = pd.DataFrame({'Latitude': latitude, 'Longitude': longitude})

# Display the ground track data in a table format
print("\nGround Track Data:")
print("------------------")
print(ground_track_df)

```

    NetCDF Header Contents:
    -----------------------
    <class 'netCDF4._netCDF4.Dataset'>
    root group (NETCDF4 data model, file format HDF5):
        Conventions: CF-1.6
        title: IPF SRAL/MWR Level 2 Measurement
        mission_name: Sentinel 3A
        altimeter_sensor_name: SRAL
        radiometer_sensor_name: MWR
        gnss_sensor_name: GNSS
        doris_sensor_name: DORIS
        netcdf_version: 4.2 of Mar 13 2018 10:14:33 $
        product_name: S3A_SR_2_LAN_HY_20230220T161116_20230220T163013_20230407T024613_1136_095_354______LN3_O_NT_005.SEN3
        institution: LN3
        source: IPF-SM-2 
        history:  
        contact: eosupport@copernicus.esa.int
        creation_time: 2023-04-07T02:47:24Z
        references: S3IPF PDS 003.2 - i3r2 - Product Data Format Specification - SRAL-MWR Level 2 Land
        processing_baseline: SM_L2HY.005.03.00
        acq_station_name: CGS
        phase_number: 1
        cycle_number: 95
        absolute_rev_number: 36516
        pass_number: 707
        absolute_pass_number: 73033
        equator_time: 2023-02-20T14:25:09.971084Z
        equator_longitude: 113.76792214340986
        semi_major_ellipsoid_axis: 6378137.0
        ellipsoid_flattening: 0.00335281066474748
        first_meas_time: 2023-02-20T16:11:16.082560Z
        last_meas_time: 2023-02-20T16:30:12.666494Z
        first_meas_lat: 18.260717
        last_meas_lat: 80.559799
        first_meas_lon: 84.38189
        last_meas_lon: 17.437625
        xref_altimeter_level1: S3A_SR_1_LAN_RD_20230220T145023_20230220T154052_20230321T002555_3029_095_353______LN3_O_NT_005.SEN3,S3A_SR_1_LAN_RD_20230220T154052_20230220T163122_20230321T012643_3029_095_353______LN3_O_NT_005.SEN3,S3A_SR_1_LAN_RD_20230220T163122_20230220T172150_20230321T022257_3028_095_354______LN3_O_NT_005.SEN3
        xref_radiometer_level1: S3A_MW_1_MWR____20230220T144701_20230220T162640_20230320T230718_5979_095_353______LN3_O_NT_001.SEN3,S3A_MW_1_MWR____20230220T162641_20230220T180633_20230321T010315_5991_095_354______LN3_O_NT_001.SEN3
        xref_orbit_data: S3A_SR___POESAX_20230219T215523_20230221T002323_20230307T160503___________________CNE_O_NT_001.SEN3
        xref_pf_data: S3A_SR_2_PCPPAX_20230219T215942_20230220T235942_20230317T071221___________________POD_O_NT_001.SEN3
        xref_altimeter_characterisation: S3A_SR___CHDNAX_20160216T000000_20991231T235959_20200312T120000___________________MPC_O_AL_006.SEN3
        xref_radiometer_characterisation: S3A_MW___CHDNAX_20160216T000000_20991231T235959_20210929T120000___________________MPC_O_AL_005.SEN3
        xref_meteorological_files: S3__AX___MA1_AX_20230220T090000_20230220T210000_20230220T175150___________________ECW_O_SN_001.SEN3,S3__AX___MA1_AX_20230220T150000_20230221T030000_20230221T054701___________________ECW_O_SN_001.SEN3,S3__AX___MA2_AX_20230220T090000_20230220T210000_20230220T175244___________________ECW_O_SN_001.SEN3,S3__AX___MA2_AX_20230220T150000_20230221T030000_20230221T054739___________________ECW_O_SN_001.SEN3
        xref_pole_location: S3__SR_2_POL_AX_19870101T000000_20240329T000000_20230404T214835___________________CNE_O_AL_001.SEN3
        xref_iono_data: S3A_SR_2_RGI_AX_20230220T000000_20230220T235959_20230221T114834___________________CNE_O_SN_001.SEN3
        xref_mog2d_data: S3__SR_2_RMO_AX_20230220T120000_20230220T120000_20230312T234855___________________CNE_O_NT_001.SEN3,S3__SR_2_RMO_AX_20230220T180000_20230220T180000_20230313T114859___________________CNE_O_NT_001.SEN3
        xref_seaice_concentration_north: S3__SR_2_SICNAX_20230220T000000_20230220T235959_20230308T082045___________________OSI_O_NT_001.SEN3
        xref_seaice_concentration_south: S3__SR_2_SICSAX_20230220T000000_20230220T235959_20230308T082045___________________OSI_O_NT_001.SEN3
        xref_altimeter_ltm: S3A_SR_1_CA1LAX_20000101T000000_20230320T135111_20230320T230723___________________LN3_O_AL____.SEN3,S3A_SR_1_CA1SAX_20000101T000000_20230320T135117_20230320T230723___________________LN3_O_AL____.SEN3,S3A_SR_1_CA2KAX_20000101T000000_20230320T135124_20230320T230723___________________LN3_O_AL____.SEN3,S3A_SR_1_CA2CAX_20000101T000000_20230320T135124_20230320T230723___________________LN3_O_AL____.SEN3
        xref_doris_uso: S3A_SR_1_USO_AX_20160223T195017_20230320T013334_20230320T084857___________________CNE_O_AL_001.SEN3
        xref_time_correlation: S3A_AX___OSF_AX_20160216T192404_99991231T235959_20220330T090651___________________EUM_O_AL_001.SEN3
        dimensions(sizes): time_01(990), time_20_ku(19961), time_20_c(19326), echo_sample_ind(128), wf_sample_ind(256), max_multi_stack_ind(256)
        variables(dimensions): float64 time_01(time_01), float64 time_20_ku(time_20_ku), float64 time_20_c(time_20_c), int8 echo_sample_ind(echo_sample_ind), int16 wf_sample_ind(wf_sample_ind), int16 max_multi_stack_ind(max_multi_stack_ind), int16 UTC_day_01(time_01), float64 UTC_sec_01(time_01), int16 UTC_day_20_ku(time_20_ku), float64 UTC_sec_20_ku(time_20_ku), int16 UTC_day_20_c(time_20_c), float64 UTC_sec_20_c(time_20_c), float64 UTC_time_1hz_20_ku(time_20_ku), float64 UTC_time_1hz_20_c(time_20_c), int32 lat_01(time_01), int32 lon_01(time_01), int32 lat_20_ku(time_20_ku), int32 lon_20_ku(time_20_ku), int32 lat_20_c(time_20_c), int32 lon_20_c(time_20_c), int8 surf_type_20_ku(time_20_ku), int8 surf_type_20_c(time_20_c), int8 surf_class_01(time_01), int8 surf_class_20_ku(time_20_ku), int8 surf_class_20_c(time_20_c), int8 rad_surf_type_01(time_01), int16 angle_coast_20_ku(time_20_ku), int32 dist_coast_20_ku(time_20_ku), int32 alt_20_ku(time_20_ku), int32 alt_20_c(time_20_c), int16 orb_alt_rate_20_ku(time_20_ku), int16 orb_alt_rate_20_c(time_20_c), int32 tracker_range_20_ku(time_20_ku), int32 tracker_range_20_c(time_20_c), int32 tracker_range_20_plrm_ku(time_20_c), uint32 h0_nav_dem_20_ku(time_20_ku), int16 agc_20_ku(time_20_ku), int16 agc_20_c(time_20_c), int16 agc_20_plrm_ku(time_20_c), int8 agc_qual_20_ku(time_20_ku), int8 agc_qual_20_c(time_20_c), int8 agc_qual_20_plrm_ku(time_20_c), int32 scale_factor_20_ku(time_20_ku), int32 scale_factor_20_c(time_20_c), int32 scale_factor_20_plrm_ku(time_20_c), int32 range_water_20_ku(time_20_ku), int32 range_water_20_c(time_20_c), int32 range_water_20_plrm_ku(time_20_c), int8 range_water_qual_20_ku(time_20_ku), int8 range_water_qual_20_c(time_20_c), int8 range_water_qual_20_plrm_ku(time_20_c), int16 range_water_rms_01_ku(time_01), int16 range_water_rms_01_c(time_01), int16 range_water_rms_01_plrm_ku(time_01), int8 range_water_numval_01_ku(time_01), int8 range_water_numval_01_c(time_01), int8 range_water_numval_01_plrm_ku(time_01), int32 interpolated_c_band_range_water_20_ku(time_20_ku), int16 sig0_water_01_ku(time_01), int16 sig0_water_01_c(time_01), int16 sig0_water_01_plrm_ku(time_01), int16 sig0_water_20_ku(time_20_ku), int16 sig0_water_20_c(time_20_c), int16 sig0_water_20_plrm_ku(time_20_c), int8 sig0_water_qual_01_ku(time_01), int8 sig0_water_qual_01_c(time_01), int8 sig0_water_qual_01_plrm_ku(time_01), int8 sig0_water_qual_20_ku(time_20_ku), int8 sig0_water_qual_20_c(time_20_c), int8 sig0_water_qual_20_plrm_ku(time_20_c), int16 sig0_water_rms_01_ku(time_01), int16 sig0_water_rms_01_c(time_01), int16 sig0_water_rms_01_plrm_ku(time_01), int8 sig0_water_numval_01_ku(time_01), int8 sig0_water_numval_01_c(time_01), int8 sig0_water_numval_01_plrm_ku(time_01), int16 swh_water_20_ku(time_20_ku), int16 swh_water_20_c(time_20_c), int16 swh_water_20_plrm_ku(time_20_c), int8 swh_water_qual_20_ku(time_20_ku), int8 swh_water_qual_20_c(time_20_c), int8 swh_water_qual_20_plrm_ku(time_20_c), int16 swh_water_rms_01_ku(time_01), int16 swh_water_rms_01_c(time_01), int16 swh_water_rms_01_plrm_ku(time_01), int8 swh_water_numval_01_ku(time_01), int8 swh_water_numval_01_c(time_01), int8 swh_water_numval_01_plrm_ku(time_01), int32 epoch_water_20_ku(time_20_ku), int32 epoch_water_20_c(time_20_c), int32 epoch_water_20_plrm_ku(time_20_c), int32 sigmac_water_20_ku(time_20_ku), int32 sigmac_water_20_c(time_20_c), int32 sigmac_water_20_plrm_ku(time_20_c), int32 amplitude_water_20_ku(time_20_ku), int32 amplitude_water_20_c(time_20_c), int32 amplitude_water_20_plrm_ku(time_20_c), int32 thermal_noise_water_20_ku(time_20_ku), int32 thermal_noise_water_20_c(time_20_c), int32 thermal_noise_water_20_plrm_ku(time_20_c), int16 off_nadir_angle_wf_water_20_ku(time_20_ku), int16 off_nadir_angle_wf_water_20_plrm_ku(time_20_c), int8 number_of_iterations_water_20_ku(time_20_ku), int8 number_of_iterations_water_20_c(time_20_c), int8 number_of_iterations_water_20_plrm_ku(time_20_c), int32 mqe_water_20_ku(time_20_ku), int32 mqe_water_20_c(time_20_c), int32 mqe_water_20_plrm_ku(time_20_c), int32 range_ocog_20_ku(time_20_ku), int32 range_ocog_20_c(time_20_c), int16 sig0_ocog_20_ku(time_20_ku), int16 sig0_ocog_20_c(time_20_c), int32 elevation_ocog_20_ku(time_20_ku), int32 range_sea_ice_20_ku(time_20_ku), int16 sig0_sea_ice_sheet_20_ku(time_20_ku), int8 surf_type_class_20_ku(time_20_ku), int32 uso_cor_20_ku(time_20_ku), int32 uso_cor_20_c(time_20_c), int32 int_path_cor_20_ku(time_20_ku), int32 int_path_cor_20_c(time_20_c), int32 int_path_cor_20_plrm_ku(time_20_c), int16 dop_cor_20_ku(time_20_ku), int16 dop_cor_20_c(time_20_c), int16 dop_cor_20_plrm_ku(time_20_c), int16 cog_cor_01(time_01), int16 mod_instr_cor_range_01_ku(time_01), int16 mod_instr_cor_range_01_c(time_01), int16 mod_instr_cor_range_01_plrm_ku(time_01), int32 net_instr_cor_range_20_ku(time_20_ku), int32 net_instr_cor_range_20_c(time_20_c), int32 net_instr_cor_range_20_plrm_ku(time_20_c), int32 agc_cor_20_ku(time_20_ku), int32 agc_cor_20_c(time_20_c), int32 agc_cor_20_plrm_ku(time_20_c), int16 sig0_cal_20_ku(time_20_ku), int16 sig0_cal_20_c(time_20_c), int16 sig0_cal_20_plrm_ku(time_20_c), int16 mod_instr_cor_sig0_01_ku(time_01), int16 mod_instr_cor_sig0_01_c(time_01), int16 mod_instr_cor_sig0_01_plrm_ku(time_01), int16 net_instr_cor_sig0_20_ku(time_20_ku), int16 net_instr_cor_sig0_20_c(time_20_c), int16 net_instr_cor_sig0_20_plrm_ku(time_20_c), int16 mod_instr_cor_swh_01_ku(time_01), int16 mod_instr_cor_swh_01_c(time_01), int16 mod_instr_cor_swh_01_plrm_ku(time_01), int16 net_instr_cor_swh_20_ku(time_20_ku), int16 net_instr_cor_swh_20_c(time_20_c), int16 net_instr_cor_swh_20_plrm_ku(time_20_c), int16 mod_dry_tropo_cor_zero_altitude_01(time_01), int16 mod_dry_tropo_cor_meas_altitude_01(time_01), int16 mod_wet_tropo_cor_zero_altitude_01(time_01), int16 mod_wet_tropo_cor_meas_altitude_01(time_01), int16 comp_wet_tropo_cor_01_ku(time_01), int16 comp_wet_tropo_cor_01_plrm_ku(time_01), int16 rad_wet_tropo_cor_01_ku(time_01), int16 rad_wet_tropo_cor_01_plrm_ku(time_01), int16 rad_wet_tropo_cor_sst_gam_01_ku(time_01), int16 rad_wet_tropo_cor_sst_gam_01_plrm_ku(time_01), int16 iono_cor_alt_20_ku(time_20_ku), int16 iono_cor_alt_20_plrm_ku(time_20_c), int16 iono_cor_gim_01_ku(time_01), int16 sea_state_bias_01_ku(time_01), int16 sea_state_bias_01_c(time_01), int16 sea_state_bias_01_plrm_ku(time_01), int16 atm_cor_sig0_01_ku(time_01), int16 atm_cor_sig0_01_c(time_01), int16 atm_cor_sig0_01_plrm_ku(time_01), int32 geoid_01(time_01), int32 odle_01(time_01), int16 inv_bar_cor_01(time_01), int16 hf_fluct_cor_01(time_01), int32 ocean_tide_sol1_01(time_01), int32 ocean_tide_sol2_01(time_01), int16 ocean_tide_eq_01(time_01), int16 ocean_tide_non_eq_01(time_01), int16 load_tide_sol1_01(time_01), int16 load_tide_sol2_01(time_01), int16 solid_earth_tide_01(time_01), int16 pole_tide_01(time_01), int16 rain_rate_01(time_01), int16 rain_att_01_ku(time_01), int16 rain_att_01_plrm_ku(time_01), int32 sea_ice_concentration_20_ku(time_20_ku), int32 snow_density_20_ku(time_20_ku), int32 snow_depth_20_ku(time_20_ku), int16 wind_speed_mod_u_01(time_01), int16 wind_speed_mod_v_01(time_01), int16 wind_speed_alt_01_ku(time_01), int16 wind_speed_alt_01_plrm_ku(time_01), int16 rad_water_vapor_01_ku(time_01), int16 rad_liquid_water_01_ku(time_01), int16 corrected_off_nadir_angle_wf_water_01_ku(time_01), int16 corrected_off_nadir_angle_wf_water_01_plrm_ku(time_01), int8 val_alt_off_nadir_angle_wf_water_01_ku(time_01), int8 val_alt_off_nadir_angle_wf_water_01_plrm_ku(time_01), int8 off_nadir_angle_used_20_ku(time_20_ku), int8 off_nadir_angle_used_20_plrm_ku(time_20_c), int16 off_nadir_angle_rms_01_ku(time_01), int16 off_nadir_angle_rms_01_plrm_ku(time_01), int8 off_nadir_angle_numval_01_ku(time_01), int8 off_nadir_angle_numval_01_plrm_ku(time_01), int16 mod_instr_cor_off_nadir_angle_01_ku(time_01), int16 mod_instr_cor_off_nadir_angle_01_plrm_ku(time_01), int32 off_nadir_roll_angle_pf_01(time_01), int32 off_nadir_pitch_angle_pf_01(time_01), int32 off_nadir_yaw_angle_pf_01(time_01), int16 tb_238_01(time_01), int16 tb_365_01(time_01), int16 tb_238_std_01(time_01), int16 tb_365_std_01(time_01), uint32 waveform_20_ku(time_20_ku, wf_sample_ind), uint32 waveform_20_c(time_20_c, echo_sample_ind), uint32 waveform_20_plrm_ku(time_20_c, echo_sample_ind), int8 instr_op_mode_20_ku(time_20_ku), int8 instr_op_mode_20_c(time_20_c), int8 mode_id_20_ku(time_20_ku), int8 mode_id_20_c(time_20_c), int8 meteo_map_avail_01(time_01), int8 rain_flag_01_ku(time_01), int8 rain_flag_01_plrm_ku(time_01), int8 open_sea_ice_flag_01_ku(time_01), int8 open_sea_ice_flag_01_plrm_ku(time_01), int8 open_water_class_01_ku(time_01), int8 open_water_class_01_plrm_ku(time_01), int8 first_year_ice_class_01_ku(time_01), int8 first_year_ice_class_01_plrm_ku(time_01), int8 multi_year_ice_class_01_ku(time_01), int8 multi_year_ice_class_01_plrm_ku(time_01), int8 wet_ice_class_01_ku(time_01), int8 wet_ice_class_01_plrm_ku(time_01), int8 interp_flag_water_tide_sol1_01(time_01), int8 interp_flag_water_tide_sol2_01(time_01), int8 rad_along_track_avg_flag_01(time_01), int8 tb_238_quality_flag_01(time_01), int8 tb_365_quality_flag_01(time_01), int8 climato_use_flag_01_ku(time_01), int8 climato_use_flag_01_plrm_ku(time_01), int16 peakiness_1_20_ku(time_20_ku), int16 peakiness_1_20_c(time_20_c), int16 peakiness_1_20_plrm_ku(time_20_c), int16 peakiness_2_20_ku(time_20_ku), int16 peakiness_2_20_c(time_20_c), uint16 nb_stack_20_ku(time_20_ku), uint32 max_stack_20_ku(time_20_ku), uint32 stdev_stack_20_ku(time_20_ku), int32 skew_stack_20_ku(time_20_ku), int32 kurt_stack_20_ku(time_20_ku), int16 beam_ang_stack_20_ku(time_20_ku, max_multi_stack_ind), uint16 beam_form_20_ku(time_20_ku), int16 max_loc_stack_20_ku(time_20_ku), int8 nav_bul_status_20_ku(time_20_ku), int8 nav_bul_status_20_c(time_20_c), int8 nav_bul_source_20_ku(time_20_ku), int8 nav_bul_source_20_c(time_20_c), uint32 nav_bul_coarse_time_20_ku(time_20_ku), uint32 nav_bul_coarse_time_20_c(time_20_c), uint32 nav_bul_fine_time_20_ku(time_20_ku), uint32 nav_bul_fine_time_20_c(time_20_c), uint16 seq_count_20_ku(time_20_ku), uint16 seq_count_20_c(time_20_c), int8 isp_time_status_20_ku(time_20_ku), int8 isp_time_status_20_c(time_20_c), int8 oper_instr_20_ku(time_20_ku), int8 oper_instr_20_c(time_20_c), int8 cl_gain_20_ku(time_20_ku), int8 cl_gain_20_c(time_20_c), int8 acq_stat_20_ku(time_20_ku), int8 acq_stat_20_c(time_20_c), int8 dem_eeprom_20_ku(time_20_ku), int8 dem_eeprom_20_c(time_20_c), int8 weighting_20_ku(time_20_ku), int8 weighting_20_c(time_20_c), int8 loss_track_20_ku(time_20_ku), int8 loss_track_20_c(time_20_c), int8 flag_man_pres_20_ku(time_20_ku), int8 flag_man_pres_20_c(time_20_c), int8 flag_man_thrust_20_ku(time_20_ku), int8 flag_man_thrust_20_c(time_20_c), int8 flag_man_plane_20_ku(time_20_ku), int8 flag_man_plane_20_c(time_20_c), int16 index_1hz_meas_20_ku(time_20_ku), int16 index_1hz_meas_20_c(time_20_c), int32 index_first_20hz_meas_01_ku(time_01), int32 index_first_20hz_meas_01_c(time_01), int16 num_20hz_meas_01_ku(time_01), int16 num_20hz_meas_01_c(time_01), int8 orbit_type_01(time_01), int8 waveform_qual_ice_20_ku(time_20_ku), int16 iono_cor_alt_filtered_01_ku(time_01), int16 iono_cor_alt_filtered_01_plrm_ku(time_01), uint32 rip_20_ku(time_20_ku, max_multi_stack_ind)
        groups: 
    
    NetCDF Attributes:
    --------------------
    Conventions: CF-1.6
    title: IPF SRAL/MWR Level 2 Measurement
    mission_name: Sentinel 3A
    altimeter_sensor_name: SRAL
    radiometer_sensor_name: MWR
    gnss_sensor_name: GNSS
    doris_sensor_name: DORIS
    netcdf_version: 4.2 of Mar 13 2018 10:14:33 $
    product_name: S3A_SR_2_LAN_HY_20230220T161116_20230220T163013_20230407T024613_1136_095_354______LN3_O_NT_005.SEN3
    institution: LN3
    source: IPF-SM-2 
    history:  
    contact: eosupport@copernicus.esa.int
    creation_time: 2023-04-07T02:47:24Z
    references: S3IPF PDS 003.2 - i3r2 - Product Data Format Specification - SRAL-MWR Level 2 Land
    processing_baseline: SM_L2HY.005.03.00
    acq_station_name: CGS
    phase_number: 1
    cycle_number: 95
    absolute_rev_number: 36516
    pass_number: 707
    absolute_pass_number: 73033
    equator_time: 2023-02-20T14:25:09.971084Z
    equator_longitude: 113.76792214340986
    semi_major_ellipsoid_axis: 6378137.0
    ellipsoid_flattening: 0.00335281066474748
    first_meas_time: 2023-02-20T16:11:16.082560Z
    last_meas_time: 2023-02-20T16:30:12.666494Z
    first_meas_lat: 18.260717
    last_meas_lat: 80.559799
    first_meas_lon: 84.38189
    last_meas_lon: 17.437625
    xref_altimeter_level1: S3A_SR_1_LAN_RD_20230220T145023_20230220T154052_20230321T002555_3029_095_353______LN3_O_NT_005.SEN3,S3A_SR_1_LAN_RD_20230220T154052_20230220T163122_20230321T012643_3029_095_353______LN3_O_NT_005.SEN3,S3A_SR_1_LAN_RD_20230220T163122_20230220T172150_20230321T022257_3028_095_354______LN3_O_NT_005.SEN3
    xref_radiometer_level1: S3A_MW_1_MWR____20230220T144701_20230220T162640_20230320T230718_5979_095_353______LN3_O_NT_001.SEN3,S3A_MW_1_MWR____20230220T162641_20230220T180633_20230321T010315_5991_095_354______LN3_O_NT_001.SEN3
    xref_orbit_data: S3A_SR___POESAX_20230219T215523_20230221T002323_20230307T160503___________________CNE_O_NT_001.SEN3
    xref_pf_data: S3A_SR_2_PCPPAX_20230219T215942_20230220T235942_20230317T071221___________________POD_O_NT_001.SEN3
    xref_altimeter_characterisation: S3A_SR___CHDNAX_20160216T000000_20991231T235959_20200312T120000___________________MPC_O_AL_006.SEN3
    xref_radiometer_characterisation: S3A_MW___CHDNAX_20160216T000000_20991231T235959_20210929T120000___________________MPC_O_AL_005.SEN3
    xref_meteorological_files: S3__AX___MA1_AX_20230220T090000_20230220T210000_20230220T175150___________________ECW_O_SN_001.SEN3,S3__AX___MA1_AX_20230220T150000_20230221T030000_20230221T054701___________________ECW_O_SN_001.SEN3,S3__AX___MA2_AX_20230220T090000_20230220T210000_20230220T175244___________________ECW_O_SN_001.SEN3,S3__AX___MA2_AX_20230220T150000_20230221T030000_20230221T054739___________________ECW_O_SN_001.SEN3
    xref_pole_location: S3__SR_2_POL_AX_19870101T000000_20240329T000000_20230404T214835___________________CNE_O_AL_001.SEN3
    xref_iono_data: S3A_SR_2_RGI_AX_20230220T000000_20230220T235959_20230221T114834___________________CNE_O_SN_001.SEN3
    xref_mog2d_data: S3__SR_2_RMO_AX_20230220T120000_20230220T120000_20230312T234855___________________CNE_O_NT_001.SEN3,S3__SR_2_RMO_AX_20230220T180000_20230220T180000_20230313T114859___________________CNE_O_NT_001.SEN3
    xref_seaice_concentration_north: S3__SR_2_SICNAX_20230220T000000_20230220T235959_20230308T082045___________________OSI_O_NT_001.SEN3
    xref_seaice_concentration_south: S3__SR_2_SICSAX_20230220T000000_20230220T235959_20230308T082045___________________OSI_O_NT_001.SEN3
    xref_altimeter_ltm: S3A_SR_1_CA1LAX_20000101T000000_20230320T135111_20230320T230723___________________LN3_O_AL____.SEN3,S3A_SR_1_CA1SAX_20000101T000000_20230320T135117_20230320T230723___________________LN3_O_AL____.SEN3,S3A_SR_1_CA2KAX_20000101T000000_20230320T135124_20230320T230723___________________LN3_O_AL____.SEN3,S3A_SR_1_CA2CAX_20000101T000000_20230320T135124_20230320T230723___________________LN3_O_AL____.SEN3
    xref_doris_uso: S3A_SR_1_USO_AX_20160223T195017_20230320T013334_20230320T084857___________________CNE_O_AL_001.SEN3
    xref_time_correlation: S3A_AX___OSF_AX_20160216T192404_99991231T235959_20220330T090651___________________EUM_O_AL_001.SEN3
    
    Ground Track Data:
    ------------------
         Latitude  Longitude
    0   18.260717  84.381890
    1   18.263608  84.381202
    2   18.266499  84.380514
    3   18.269390  84.379826
    4   18.272281  84.379138
    5   18.275171  84.378450
    6   18.278062  84.377762
    7   18.280953  84.377073
    8   18.283844  84.376385
    9   18.286735  84.375697
    10  18.289626  84.375009
    11  18.292517  84.374321
    12  18.295407  84.373632
    13  18.298298  84.372944
    14  18.301187  84.372256
    15  18.304078  84.371568
    16  18.306968  84.370880
    17  18.309859  84.370191
    18  18.312750  84.369503
    19  18.315641  84.368815
    20  18.318531  84.368127
    21  18.321422  84.367438
    22  18.324312  84.366750
    23  18.327203  84.366062
    24  18.330094  84.365373
    25  18.332984  84.364685
    26  18.335875  84.363996
    27  18.338766  84.363308
    28  18.341657  84.362619
    29  18.344548  84.361931
    30  18.347439  84.361242
    31  18.350330  84.360554
    32  18.353220  84.359865
    33  18.356110  84.359177
    34  18.359001  84.358488
    35  18.361892  84.357800
    36  18.364783  84.357111
    37  18.367674  84.356422
    38  18.370565  84.355734
    39  18.373456  84.355045
    40  18.376347  84.354356
    41  18.379238  84.353668
    42  18.382128  84.352979
    43  18.385019  84.352290
    44  18.387909  84.351602
    45  18.390800  84.350913
    46  18.393691  84.350224
    47  18.396581  84.349535
    48  18.399472  84.348847
    49  18.402363  84.348158
    50  18.405253  84.347469
    51  18.408144  84.346780
    52  18.411035  84.346091
    53  18.413926  84.345402
    54  18.416816  84.344714
    55  18.419707  84.344025
    56  18.422598  84.343336
    57  18.425489  84.342647
    58  18.428380  84.341958
    59  18.431271  84.341269
    60  18.434161  84.340580
    61  18.437052  84.339891
    62  18.439943  84.339202
    63  18.442833  84.338513
    64  18.445724  84.337824
    65  18.448614  84.337135
    66  18.451505  84.336445
    67  18.454396  84.335756
    68  18.457287  84.335067
    69  18.460178  84.334378
    70  18.463068  84.333689
    71  18.465959  84.333000
    72  18.468850  84.332311
    73  18.471741  84.331621
    74  18.474631  84.330932
    75  18.477522  84.330243
    76  18.480412  84.329554
    77  18.483303  84.328864
    78  18.486194  84.328175
    79  18.489085  84.327486
    80  18.491976  84.326796
    81  18.494867  84.326107
    82  18.497757  84.325418
    83  18.500648  84.324728
    84  18.503538  84.324039
    85  18.506429  84.323350
    86  18.509319  84.322660
    87  18.512209  84.321971
    88  18.515101  84.321281
    89  18.517991  84.320592
    90  18.520882  84.319902
    91  18.523773  84.319213
    92  18.526664  84.318523
    93  18.529554  84.317834
    94  18.532445  84.317144
    95  18.535336  84.316455
    96  18.538227  84.315765
    97  18.541118  84.315075
    98  18.544008  84.314386
    99  18.546898  84.313696
    

### Plot the ground track data on a map


```python
from mpl_toolkits.basemap import Basemap

# Specify the path to the NetCDF file
file_path = "C:/Users/HP/Downloads/enhanced_measurement.nc"

# Open the NetCDF file
dataset = nc.Dataset(file_path)

# Extract latitude and longitude variables: (lat_01,lon_01), (lat_20_ku,lon_20_ku), (lat_20_c,lon_20_c )
latitude = dataset.variables['lat_20_ku'][:]
longitude = dataset.variables['lon_20_ku'][:]

# Close the NetCDF file
dataset.close()

# Create a Basemap instance
m = Basemap(projection='cyl', llcrnrlat=latitude.min(), urcrnrlat=90,
            llcrnrlon=longitude.min(), urcrnrlon=90, resolution='l')

# Create a figure and axes
fig, ax = plt.subplots(figsize=(12, 9))

# Plot the ground track data
x, y = m(longitude, latitude)
ax.plot(x, y, color='blue', linewidth=1.5)

# Draw coastlines and country boundaries
m.drawcoastlines(linewidth=0.5)
m.drawcountries(linewidth=0.5)

# Draw parallels and meridians
m.drawparallels(range(int(latitude.min()), int(latitude.max()), 10), labels=[1, 0, 0, 0], fontsize=10)
m.drawmeridians(range(int(longitude.min()), int(longitude.max()), 10), labels=[0, 0, 0, 1], fontsize=10)

# Add country boundaries
m.drawcountries(linewidth=0.5)
m.drawcountries(linewidth=0.5)
m.drawrivers(color='blue', linewidth=0.5)

# Add title and labels
plt.title('Ground Track Data', fontsize=14)

# Customize the map background
m.drawmapboundary(fill_color='lightblue')
m.fillcontinents(color='beige', lake_color='lightblue')

# Show the map
plt.show()

```


    
![png](output_49_0.png)
    


![image.png](attachment:image.png)

### Dataset that should be taken according to the satellite ground track data if we want proper intersection points

### Yamuna river, Kalpi, UP

### Smoothen the DEM using bilateral filter for Yamuna River near Kalpi, UP


```python
import rasterio
from skimage import img_as_float
from skimage.restoration import denoise_bilateral

# Specify the path to the DEM tif file
file_path = "C:/Users/HP/OneDrive/Desktop/Surge Project/Datasets/Kalpi_Yamuna_USGS.tif"

# Open the DEM tif file
with rasterio.open(file_path) as src:
    # Read the elevation data as a numpy array
    elevation_array = src.read(1)

# Calculate the number of rows and columns based on the square root of the elevation array size
n = int(np.sqrt(elevation_array.size))
nrows = ncols = n

# Reshape the elevation array into a matrix
elevation_matrix = elevation_array.reshape((nrows, ncols))

# Convert elevation matrix to float between 0 and 1
elevation_float = img_as_float(elevation_matrix)

# Apply bilateral filter for edge-preserving smoothing
sigma_color = 0.1    # Controls the range similarity
sigma_spatial = 5    # Controls the spatial smoothing
smoothed_elevation = denoise_bilateral(elevation_float, sigma_color=sigma_color, sigma_spatial=sigma_spatial)

# Set up the figure and axes
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot the original DEM
axs[0].imshow(elevation_matrix, cmap='viridis')
axs[0].set_title('Original DEM')
axs[0].axis('off')

# Plot the smoothed DEM
axs[1].imshow(smoothed_elevation, cmap='viridis')
axs[1].set_title('Smoothed DEM')
axs[1].axis('off')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the figures
plt.show()
```


    
![png](output_54_0.png)
    


### Compute gradient direction, magnitude and their respective histograms for Yamuna River near Kalpi, UP


```python
from scipy.ndimage import gaussian_filter
from skimage import io

# Load the DEM from the TIFF file
dem = io.imread("C:/Users/HP/OneDrive/Desktop/Surge Project/Datasets/Kalpi_Yamuna_USGS.tif")

# Apply Gaussian filter for smoothing
sigma = 1.5  # Controls the level of smoothing
smoothed_dem = gaussian_filter(dem, sigma=sigma)

# Compute the gradient using Sobel operators
dx = np.gradient(smoothed_dem, axis=1)
dy = np.gradient(smoothed_dem, axis=0)

# Compute the gradient magnitude
gradient_magnitude = np.sqrt(dx**2 + dy**2)

# Compute the gradient direction
gradient_direction = np.arctan2(dy, dx)

# Set up the figure and axes
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot the gradient direction map
cmap_direction = 'hsv'
im_direction = axs[0, 0].imshow(gradient_direction, cmap=cmap_direction)
axs[0, 0].set_title('Gradient Direction')
axs[0, 0].axis('off')
cbar_direction = fig.colorbar(im_direction, ax=axs[0, 0], orientation='vertical')

# Compute and plot histogram for the gradient direction map
direction_hist, direction_bins = np.histogram(gradient_direction, bins=256, range=(-np.pi, np.pi))
axs[1, 0].plot(direction_bins[:-1], direction_hist, color='black')
#axs[1, 0].set_title('Gradient Direction Histogram')
axs[1, 0].set_xlabel('Gradient Direction (radians)')
axs[1, 0].set_ylabel('Frequency')

# Plot the gradient magnitude map
cmap_magnitude = 'hot'
im_magnitude = axs[0, 1].imshow(gradient_magnitude, cmap=cmap_magnitude)
#axs[0, 1].set_title('Gradient Magnitude')
axs[0, 1].axis('off')
cbar_magnitude = fig.colorbar(im_magnitude, ax=axs[0, 1], orientation='vertical')

# Compute and plot histogram for the gradient magnitude map
magnitude_hist, magnitude_bins = np.histogram(gradient_magnitude, bins=256, range=(0, gradient_magnitude.max()))
axs[1, 1].plot(magnitude_bins[:-1], magnitude_hist, color='black')
#axs[1, 1].set_title('Gradient Magnitude Histogram')
#axs[1, 1].set_xlabel('Gradient Magnitude')
axs[1, 1].set_ylabel('Frequency')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the figure
plt.show()

```


    
![png](output_56_0.png)
    


###  Identify potential river channel locations for Yamuna River near Kalpi, UP


```python
# Identify potential river channel locations using local maxima
channel_locations = peak_local_max(gradient_magnitude, min_distance=10, num_peaks=2)

# Visualize the gradient magnitude with channel locations
plt.figure(figsize=(10, 6))
plt.imshow(gradient_magnitude, cmap='hot')
plt.plot(channel_locations[:, 1], channel_locations[:, 0], 'bo')
plt.title('Gradient Magnitude with Channel Locations')
plt.colorbar()
plt.show()

# Identify potential river channel locations using local maxima
channel_locations = peak_local_max(gradient_direction, min_distance=10, num_peaks=2)

# Visualize the gradient direction with channel locations
plt.figure(figsize=(10, 6))
plt.imshow(gradient_direction, cmap='hot')
plt.plot(channel_locations[:, 1], channel_locations[:, 0], 'bo')
plt.title('Gradient Magnitude with Channel Locations')
plt.colorbar()
plt.show()
```


    
![png](output_58_0.png)
    



    
![png](output_58_1.png)
    


### Compute distance transform for Yamuna River near Kalpi, UP


```python
# Compute distance transform 
distance_transforms = []
for loc in channel_locations:
    y, x = loc  # Extract coordinates of the channel location
    distance_transform = distance_transform_edt(np.logical_not(gradient_magnitude > gradient_magnitude[y, x]))
    distance_transforms.append(distance_transform)

# Visualize distance transforms
plt.figure(figsize=(10, 6))
for dt in distance_transforms:
    plt.imshow(dt, cmap='hot')
    plt.colorbar()
    plt.title('Distance Transform')
    plt.show()

```


    
![png](output_60_0.png)
    



    
![png](output_60_1.png)
    


### Approximate river width for Yamuna River near Kalpi, UP


```python
# Estimate river width 
river_width = np.mean([2 * np.max(dt) for dt in distance_transforms])

# Print the estimated river width
print("Estimated river width:", river_width)
```

    Estimated river width: 31.04834939252005
    


```python
# Compute the gradient in the x and y directions
gradient_x = np.gradient(smoothed_dem, axis=1)
gradient_y = np.gradient(smoothed_dem, axis=0)

# Compute the orientation angle at each pixel
orientation = np.arctan2(gradient_y, gradient_x)

# Compute the perpendicular orientation to the river channel
perpendicular_orientation = np.mod(orientation + np.pi / 2, np.pi)

# Determine the pixel coordinates of the cross-section location
cross_section_x = int(river_width / 2)  # Assuming the cross-section is in the middle of the river width
cross_section_y = int(rows / 2)  # Assuming the cross-section is in the middle row of the grid

# Compute the distances of each pixel from the cross-section location along the perpendicular orientation
distances = scipy.ndimage.distance_transform_edt(perpendicular_orientation != 0)

# Identify the pixels within the river section by comparing distances with river width
river_section_mask = distances <= river_width

# Convert the data type of smoothed_dem to floating-point
smoothed_elevation = smoothed_elevation.astype(float)

# Project the DEM onto the river section plane by assigning NaN values to pixels outside the river section
dem_projected = np.copy(smoothed_elevation)
dem_projected[~river_section_mask] = np.nan

print("Data type of projected DEM:")
print(type(dem_projected))

# Plot the projected DEM
plt.imshow(dem_projected, cmap='terrain', vmin=np.nanmin(smoothed_elevation), vmax=np.nanmax(smoothed_elevation))
plt.colorbar()
plt.title('Projected DEM')
plt.show()
```

    Data type of projected DEM:
    <class 'numpy.ndarray'>
    


    
![png](output_63_1.png)
    



```python
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import netCDF4 as nc
import matplotlib.pyplot as plt

# Load the projected DEM and ground track data
ground_track_file = "C:/Users/HP/Downloads/enhanced_measurement.nc"
ground_track_dataset = nc.Dataset(ground_track_file)
latitude = ground_track_dataset.variables['lat_20_ku'][:]
longitude = ground_track_dataset.variables['lon_20_ku'][:]

# Flatten the latitude and longitude arrays
lat_lon_points = np.column_stack((latitude.flatten(), longitude.flatten()))

# Create a KDTree from the valid points in the projected DEM
valid_indices = np.argwhere(~np.isnan(dem_projected))
valid_points = np.column_stack((longitude[valid_indices[:, 1]], latitude[valid_indices[:, 0]]))
tree = cKDTree(valid_points)

# Find the nearest neighbors for each ground track point
ground_track_points = np.column_stack((longitude, latitude))
distances, indices = tree.query(ground_track_points)

# Identify the intersection points
intersection_indices = np.where(distances == 0)[0]
intersection_points = ground_track_points[intersection_indices]

# Plot the projected DEM
plt.imshow(dem_projected, cmap='terrain', vmin=np.nanmin(smoothed_elevation), vmax=np.nanmax(smoothed_elevation))

# Plot the intersection points on top of the DEM
plt.scatter(intersection_points[:, 0], intersection_points[:, 1], color='red', marker='x', label='Intersection Points')

plt.colorbar()
plt.title('Projected DEM with Intersection Points')
plt.legend()
plt.show()

# Get the shape of the projected DEM
dem_shape = dem_projected.shape

# Print the shape of the projected DEM
print("Shape of projected DEM:", dem_shape)
```


    
![png](output_64_0.png)
    


    Shape of projected DEM: (3601, 3601)
    


```python
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Ground track data with longitude and latitude coordinates
ground_track_data = np.array([
    [79.748398, 26.12096]
])

# Shape of the projected DEM
dem_shape = (3601, 3601)


# Compute the projected grid coordinates for ground track data
lon = ground_track_data[:, 0]
lat = ground_track_data[:, 1]
lon_min, lon_max = np.min(lon), np.max(lon)
lat_min, lat_max = np.min(lat), np.max(lat)
x = np.interp(lon, (lon_min, lon_max), (0, dem_projected.shape[1] - 1)).astype(int)
y = np.interp(lat, (lat_min, lat_max), (0, dem_projected.shape[0] - 1)).astype(int)

# Find intersection points within the river section
intersecting_points = np.column_stack((x, y))[river_section_mask[y, x]]

# Plot the intersection points
plt.imshow(dem_projected, cmap='terrain', origin='lower')
plt.colorbar(label='Elevation')
plt.plot(intersecting_points[:, 1], intersecting_points[:, 0], 'rx', markersize=10, label='Intersection Points')
plt.legend()
plt.title('Intersection Points between DEM and Ground Track Data')
plt.show()


```


    
![png](output_65_0.png)
    



```python
import pandas as pd

# Convert the numpy array to a pandas DataFrame
df_dem_projected = pd.DataFrame(dem_projected)

# Print the first 100 rows of the DataFrame
print(df_dem_projected.head(100))

```

            0         1         2         3         4         5         6     \
    0   0.000767  0.000814  0.000875  0.000953  0.001047  0.001162  0.001296   
    1   0.000954  0.001001  0.001062  0.001140  0.001236  0.001352  0.001488   
    2   0.001147  0.001194  0.001256  0.001334  0.001430  0.001547  0.001685   
    3   0.001334  0.001382  0.001443  0.001522  0.001618  0.001735  0.001874   
    4   0.001504  0.001553  0.001615  0.001694  0.001790  0.001908  0.002047   
    5   0.001650  0.001700  0.001763  0.001842  0.001939  0.002057  0.002196   
    6   0.001769  0.001820  0.001884  0.001963  0.002061  0.002178  0.002317   
    7   0.001860  0.001912  0.001977  0.002057  0.002155  0.002272  0.002412   
    8   0.001926  0.001980  0.002045  0.002126  0.002224  0.002342  0.002481   
    9   0.001971  0.002026  0.002093  0.002174  0.002273  0.002391  0.002530   
    10  0.002001  0.002057  0.002125  0.002207  0.002306  0.002424  0.002563   
    11  0.002019  0.002077  0.002145  0.002228  0.002327  0.002446  0.002585   
    12  0.002030  0.002089  0.002158  0.002241  0.002340  0.002459  0.002599   
    13  0.002036  0.002095  0.002165  0.002248  0.002348  0.002467  0.002607   
    14  0.002040  0.002099  0.002169  0.002252  0.002352  0.002471  0.002612   
    15  0.002042  0.002101  0.002171  0.002254  0.002354  0.002474  0.002614   
    16  0.002042  0.002101  0.002171  0.002254  0.002354  0.002473  0.002614   
    17  0.002042  0.002101  0.002170  0.002253  0.002353  0.002473  0.002614   
    18  0.002042  0.002100  0.002169  0.002252  0.002352  0.002472  0.002613   
    19  0.002042  0.002100  0.002169  0.002251  0.002351  0.002470  0.002612   
    20  0.002042  0.002100  0.002168  0.002250  0.002350  0.002469  0.002611   
    21  0.002043  0.002100  0.002168  0.002250  0.002349  0.002468  0.002610   
    22  0.002044  0.002101  0.002168  0.002250  0.002349  0.002468  0.002609   
    23  0.002045  0.002102  0.002169  0.002250  0.002349  0.002468  0.002608   
    24  0.002046  0.002103  0.002170  0.002250  0.002349  0.002467  0.002608   
    25  0.002046  0.002103  0.002170  0.002251  0.002349  0.002468  0.002608   
    26  0.002047  0.002104  0.002171  0.002252  0.002350  0.002468  0.002608   
    27  0.002046  0.002103  0.002171  0.002252  0.002350  0.002468  0.002609   
    28  0.002045  0.002103  0.002171  0.002252  0.002350  0.002468  0.002609   
    29  0.002044  0.002102  0.002170  0.002252  0.002350  0.002469  0.002610   
    30  0.002043  0.002101  0.002169  0.002251  0.002350  0.002469  0.002610   
    31  0.002041  0.002099  0.002168  0.002250  0.002350  0.002469  0.002611   
    32  0.002040  0.002098  0.002167  0.002250  0.002349  0.002469  0.002611   
    33  0.002038  0.002097  0.002166  0.002249  0.002349  0.002469  0.002611   
    34  0.002037  0.002095  0.002165  0.002248  0.002349  0.002469  0.002612   
    35  0.002036  0.002094  0.002164  0.002248  0.002349  0.002469  0.002612   
    36  0.002035  0.002094  0.002163  0.002247  0.002348  0.002469  0.002612   
    37  0.002035  0.002093  0.002162  0.002247  0.002348  0.002469  0.002612   
    38  0.002034  0.002092  0.002162  0.002247  0.002348  0.002469  0.002612   
    39  0.002034  0.002092  0.002162  0.002246  0.002348  0.002469  0.002612   
    40  0.002034  0.002092  0.002162  0.002247  0.002348  0.002469  0.002612   
    41  0.002034  0.002093  0.002162  0.002247  0.002348  0.002469  0.002612   
    42  0.002035  0.002093  0.002162  0.002247  0.002348  0.002469  0.002612   
    43  0.002035  0.002093  0.002162  0.002247  0.002348  0.002469  0.002611   
    44  0.002035  0.002093  0.002162  0.002247  0.002348  0.002469  0.002611   
    45  0.002035  0.002093  0.002162  0.002247  0.002348  0.002468  0.002610   
    46  0.002035  0.002093  0.002162  0.002246  0.002347  0.002467  0.002609   
    47  0.002036  0.002093  0.002162  0.002246  0.002346  0.002466  0.002607   
    48  0.002037  0.002094  0.002162  0.002246  0.002346  0.002465  0.002606   
    49  0.002038  0.002094  0.002163  0.002246  0.002346  0.002464  0.002605   
    50  0.002039  0.002095  0.002163  0.002246  0.002346  0.002464  0.002604   
    51  0.002040  0.002096  0.002164  0.002247  0.002346  0.002464  0.002603   
    52  0.002040  0.002097  0.002165  0.002247  0.002346  0.002464  0.002603   
    53  0.002040  0.002097  0.002165  0.002248  0.002346  0.002464  0.002603   
    54  0.002040  0.002097  0.002165  0.002248  0.002346  0.002464  0.002603   
    55  0.002040  0.002097  0.002165  0.002247  0.002346  0.002463  0.002603   
    56  0.002039  0.002096  0.002164  0.002246  0.002345  0.002462  0.002602   
    57  0.002038  0.002095  0.002163  0.002245  0.002344  0.002461  0.002601   
    58  0.002036  0.002094  0.002162  0.002244  0.002343  0.002460  0.002600   
    59  0.002035  0.002092  0.002161  0.002243  0.002341  0.002459  0.002599   
    60  0.002034  0.002091  0.002160  0.002242  0.002341  0.002459  0.002598   
    61  0.002033  0.002091  0.002159  0.002242  0.002340  0.002458  0.002598   
    62  0.002033  0.002090  0.002159  0.002241  0.002340  0.002458  0.002597   
    63  0.002032  0.002089  0.002158  0.002241  0.002339  0.002458  0.002597   
    64  0.002032  0.002089  0.002157  0.002240  0.002339  0.002457  0.002597   
    65  0.002032  0.002089  0.002157  0.002240  0.002339  0.002457  0.002597   
    66  0.002032  0.002088  0.002157  0.002239  0.002338  0.002457  0.002596   
    67  0.002032  0.002089  0.002156  0.002239  0.002338  0.002456  0.002596   
    68  0.002032  0.002089  0.002156  0.002238  0.002337  0.002456  0.002595   
    69  0.002032  0.002089  0.002156  0.002238  0.002337  0.002455  0.002595   
    70  0.002032  0.002089  0.002157  0.002238  0.002337  0.002455  0.002595   
    71  0.002032  0.002089  0.002157  0.002238  0.002337  0.002456  0.002595   
    72  0.002032  0.002089  0.002157  0.002239  0.002338  0.002456  0.002596   
    73  0.002031  0.002089  0.002157  0.002239  0.002338  0.002457  0.002597   
    74  0.002030  0.002088  0.002157  0.002239  0.002339  0.002458  0.002598   
    75  0.002029  0.002088  0.002157  0.002239  0.002339  0.002458  0.002598   
    76  0.002029  0.002087  0.002156  0.002239  0.002339  0.002459  0.002599   
    77  0.002028  0.002087  0.002155  0.002239  0.002339  0.002458  0.002599   
    78  0.002028  0.002086  0.002155  0.002238  0.002338  0.002458  0.002598   
    79  0.002028  0.002086  0.002155  0.002237  0.002337  0.002457  0.002597   
    80  0.002027  0.002086  0.002154  0.002237  0.002336  0.002456  0.002596   
    81  0.002028  0.002086  0.002154  0.002236  0.002335  0.002454  0.002595   
    82  0.002028  0.002086  0.002154  0.002236  0.002334  0.002453  0.002594   
    83  0.002028  0.002086  0.002154  0.002235  0.002334  0.002452  0.002593   
    84  0.002029  0.002087  0.002154  0.002236  0.002334  0.002452  0.002592   
    85  0.002030  0.002088  0.002155  0.002236  0.002334  0.002452  0.002592   
    86  0.002031  0.002089  0.002157  0.002237  0.002335  0.002453  0.002593   
    87  0.002033  0.002091  0.002158  0.002239  0.002336  0.002454  0.002594   
    88  0.002034  0.002092  0.002160  0.002241  0.002338  0.002456  0.002595   
    89  0.002035  0.002093  0.002161  0.002242  0.002340  0.002457  0.002597   
    90  0.002036  0.002094  0.002162  0.002243  0.002341  0.002459  0.002599   
    91  0.002037  0.002095  0.002163  0.002245  0.002342  0.002460  0.002600   
    92  0.002038  0.002095  0.002163  0.002245  0.002343  0.002461  0.002601   
    93  0.002038  0.002095  0.002163  0.002246  0.002344  0.002462  0.002602   
    94  0.002038  0.002095  0.002163  0.002246  0.002344  0.002462  0.002602   
    95  0.002038  0.002094  0.002163  0.002246  0.002344  0.002462  0.002602   
    96  0.002037  0.002094  0.002162  0.002245  0.002344  0.002462  0.002601   
    97  0.002036  0.002093  0.002161  0.002244  0.002343  0.002461  0.002601   
    98  0.002034  0.002091  0.002160  0.002244  0.002343  0.002461  0.002600   
    99  0.002033  0.002090  0.002159  0.002243  0.002342  0.002460  0.002599   
    
            7         8         9     ...      3591      3592      3593      3594  \
    0   0.001451  0.001625  0.001815  ...  0.002080  0.002044  0.002017  0.001996   
    1   0.001647  0.001825  0.002022  ...  0.002219  0.002174  0.002138  0.002111   
    2   0.001845  0.002026  0.002228  ...  0.002337  0.002280  0.002235  0.002200   
    3   0.002035  0.002218  0.002422  ...  0.002436  0.002367  0.002312  0.002270   
    4   0.002208  0.002392  0.002598  ...  0.002520  0.002440  0.002375  0.002324   
    5   0.002357  0.002542  0.002748  ...  0.002591  0.002501  0.002428  0.002368   
    6   0.002479  0.002664  0.002869  ...  0.002650  0.002552  0.002471  0.002405   
    7   0.002573  0.002758  0.002963  ...  0.002697  0.002593  0.002506  0.002434   
    8   0.002643  0.002827  0.003032  ...  0.002733  0.002625  0.002534  0.002458   
    9   0.002691  0.002875  0.003081  ...  0.002759  0.002648  0.002555  0.002476   
    10  0.002725  0.002909  0.003114  ...  0.002776  0.002664  0.002569  0.002489   
    11  0.002746  0.002930  0.003135  ...  0.002786  0.002673  0.002578  0.002498   
    12  0.002760  0.002944  0.003149  ...  0.002791  0.002678  0.002582  0.002502   
    13  0.002769  0.002953  0.003157  ...  0.002794  0.002680  0.002584  0.002504   
    14  0.002774  0.002958  0.003163  ...  0.002796  0.002680  0.002584  0.002504   
    15  0.002777  0.002962  0.003167  ...  0.002797  0.002681  0.002584  0.002504   
    16  0.002778  0.002963  0.003168  ...  0.002798  0.002681  0.002583  0.002503   
    17  0.002778  0.002963  0.003169  ...  0.002801  0.002683  0.002583  0.002503   
    18  0.002777  0.002963  0.003169  ...  0.002804  0.002685  0.002585  0.002504   
    19  0.002776  0.002962  0.003168  ...  0.002808  0.002688  0.002588  0.002506   
    20  0.002775  0.002961  0.003168  ...  0.002811  0.002692  0.002591  0.002508   
    21  0.002774  0.002959  0.003166  ...  0.002815  0.002696  0.002595  0.002511   
    22  0.002773  0.002958  0.003165  ...  0.002818  0.002700  0.002598  0.002514   
    23  0.002772  0.002957  0.003164  ...  0.002821  0.002703  0.002602  0.002517   
    24  0.002771  0.002956  0.003163  ...  0.002824  0.002706  0.002605  0.002520   
    25  0.002771  0.002956  0.003163  ...  0.002827  0.002709  0.002608  0.002524   
    26  0.002771  0.002956  0.003162  ...  0.002829  0.002712  0.002611  0.002527   
    27  0.002772  0.002956  0.003162  ...  0.002832  0.002714  0.002614  0.002530   
    28  0.002772  0.002957  0.003162  ...  0.002834  0.002716  0.002616  0.002532   
    29  0.002773  0.002958  0.003163  ...  0.002836  0.002718  0.002618  0.002534   
    30  0.002774  0.002959  0.003164  ...  0.002838  0.002720  0.002620  0.002536   
    31  0.002775  0.002960  0.003165  ...  0.002839  0.002721  0.002621  0.002537   
    32  0.002775  0.002961  0.003166  ...  0.002840  0.002722  0.002622  0.002538   
    33  0.002776  0.002962  0.003167  ...  0.002840  0.002723  0.002623  0.002539   
    34  0.002776  0.002963  0.003169  ...  0.002840  0.002723  0.002624  0.002539   
    35  0.002777  0.002963  0.003170  ...  0.002839  0.002723  0.002624  0.002539   
    36  0.002777  0.002964  0.003171  ...  0.002838  0.002722  0.002623  0.002539   
    37  0.002777  0.002964  0.003171  ...  0.002836  0.002720  0.002622  0.002539   
    38  0.002777  0.002964  0.003171  ...  0.002835  0.002719  0.002621  0.002538   
    39  0.002777  0.002964  0.003172  ...  0.002835  0.002718  0.002619  0.002537   
    40  0.002777  0.002964  0.003172  ...  0.002836  0.002718  0.002619  0.002536   
    41  0.002777  0.002964  0.003171  ...  0.002838  0.002719  0.002619  0.002535   
    42  0.002776  0.002963  0.003171  ...  0.002842  0.002721  0.002620  0.002535   
    43  0.002776  0.002963  0.003171  ...  0.002846  0.002724  0.002622  0.002537   
    44  0.002775  0.002962  0.003170  ...  0.002851  0.002728  0.002625  0.002539   
    45  0.002775  0.002962  0.003169  ...  0.002855  0.002732  0.002628  0.002541   
    46  0.002773  0.002960  0.003168  ...  0.002859  0.002736  0.002632  0.002544   
    47  0.002772  0.002959  0.003166  ...  0.002861  0.002739  0.002635  0.002547   
    48  0.002770  0.002957  0.003165  ...  0.002863  0.002741  0.002638  0.002550   
    49  0.002769  0.002955  0.003163  ...  0.002865  0.002743  0.002640  0.002553   
    50  0.002767  0.002954  0.003161  ...  0.002866  0.002745  0.002642  0.002555   
    51  0.002766  0.002952  0.003159  ...  0.002867  0.002746  0.002644  0.002558   
    52  0.002766  0.002951  0.003157  ...  0.002868  0.002748  0.002646  0.002560   
    53  0.002765  0.002950  0.003156  ...  0.002870  0.002751  0.002649  0.002562   
    54  0.002765  0.002949  0.003155  ...  0.002873  0.002753  0.002651  0.002565   
    55  0.002764  0.002948  0.003154  ...  0.002876  0.002756  0.002654  0.002568   
    56  0.002764  0.002948  0.003153  ...  0.002880  0.002760  0.002658  0.002571   
    57  0.002763  0.002947  0.003153  ...  0.002884  0.002764  0.002661  0.002574   
    58  0.002762  0.002946  0.003152  ...  0.002888  0.002768  0.002665  0.002577   
    59  0.002761  0.002946  0.003151  ...  0.002893  0.002772  0.002668  0.002580   
    60  0.002761  0.002945  0.003151  ...  0.002898  0.002776  0.002672  0.002584   
    61  0.002760  0.002944  0.003150  ...  0.002904  0.002781  0.002676  0.002587   
    62  0.002759  0.002944  0.003150  ...  0.002910  0.002786  0.002681  0.002592   
    63  0.002759  0.002943  0.003149  ...  0.002918  0.002793  0.002687  0.002597   
    64  0.002759  0.002943  0.003148  ...  0.002927  0.002802  0.002695  0.002604   
    65  0.002758  0.002942  0.003148  ...  0.002937  0.002812  0.002705  0.002613   
    66  0.002758  0.002942  0.003147  ...  0.002948  0.002823  0.002716  0.002624   
    67  0.002757  0.002941  0.003146  ...  0.002958  0.002834  0.002727  0.002635   
    68  0.002756  0.002940  0.003146  ...  0.002967  0.002845  0.002739  0.002647   
    69  0.002756  0.002940  0.003145  ...  0.002976  0.002855  0.002750  0.002658   
    70  0.002756  0.002940  0.003145  ...  0.002983  0.002863  0.002759  0.002668   
    71  0.002756  0.002940  0.003145  ...  0.002988  0.002869  0.002766  0.002676   
    72  0.002757  0.002940  0.003145  ...  0.002992  0.002873  0.002771  0.002682   
    73  0.002758  0.002941  0.003146  ...  0.002994  0.002876  0.002774  0.002686   
    74  0.002759  0.002943  0.003147  ...  0.002995  0.002876  0.002775  0.002688   
    75  0.002760  0.002944  0.003148  ...  0.002994  0.002876  0.002774  0.002688   
    76  0.002761  0.002944  0.003149  ...  0.002994  0.002875  0.002773  0.002687   
    77  0.002761  0.002945  0.003149  ...  0.002994  0.002874  0.002772  0.002686   
    78  0.002760  0.002944  0.003149  ...  0.002994  0.002874  0.002771  0.002685   
    79  0.002760  0.002944  0.003149  ...  0.002996  0.002874  0.002771  0.002685   
    80  0.002759  0.002943  0.003149  ...  0.002999  0.002877  0.002773  0.002685   
    81  0.002758  0.002942  0.003148  ...  0.003004  0.002880  0.002775  0.002686   
    82  0.002756  0.002941  0.003147  ...  0.003010  0.002885  0.002778  0.002689   
    83  0.002756  0.002940  0.003147  ...  0.003017  0.002890  0.002783  0.002692   
    84  0.002755  0.002940  0.003146  ...  0.003024  0.002896  0.002788  0.002696   
    85  0.002755  0.002939  0.003146  ...  0.003031  0.002902  0.002793  0.002700   
    86  0.002755  0.002940  0.003146  ...  0.003037  0.002907  0.002797  0.002704   
    87  0.002756  0.002941  0.003147  ...  0.003040  0.002911  0.002800  0.002707   
    88  0.002757  0.002942  0.003148  ...  0.003041  0.002912  0.002802  0.002710   
    89  0.002759  0.002943  0.003150  ...  0.003040  0.002912  0.002802  0.002710   
    90  0.002761  0.002945  0.003151  ...  0.003036  0.002908  0.002800  0.002709   
    91  0.002762  0.002946  0.003152  ...  0.003030  0.002903  0.002796  0.002706   
    92  0.002763  0.002947  0.003153  ...  0.003023  0.002897  0.002790  0.002702   
    93  0.002764  0.002948  0.003154  ...  0.003016  0.002890  0.002784  0.002696   
    94  0.002764  0.002948  0.003153  ...  0.003010  0.002884  0.002778  0.002691   
    95  0.002764  0.002947  0.003152  ...  0.003005  0.002880  0.002774  0.002688   
    96  0.002763  0.002946  0.003151  ...  0.003003  0.002877  0.002772  0.002685   
    97  0.002762  0.002944  0.003149  ...  0.003003  0.002877  0.002772  0.002685   
    98  0.002760  0.002942  0.003147  ...  0.003006  0.002880  0.002774  0.002687   
    99  0.002759  0.002941  0.003145  ...  0.003011  0.002885  0.002779  0.002692   
    
            3595      3596      3597      3598      3599      3600  
    0   0.001976  0.001956  0.001936  0.001913  0.001889  0.001861  
    1   0.002087  0.002063  0.002040  0.002016  0.001990  0.001960  
    2   0.002171  0.002143  0.002117  0.002091  0.002063  0.002032  
    3   0.002234  0.002202  0.002172  0.002144  0.002115  0.002082  
    4   0.002282  0.002245  0.002212  0.002181  0.002150  0.002116  
    5   0.002319  0.002277  0.002240  0.002207  0.002174  0.002139  
    6   0.002349  0.002303  0.002263  0.002227  0.002192  0.002156  
    7   0.002374  0.002324  0.002281  0.002243  0.002206  0.002168  
    8   0.002395  0.002342  0.002296  0.002256  0.002217  0.002178  
    9   0.002411  0.002356  0.002309  0.002267  0.002227  0.002186  
    10  0.002423  0.002367  0.002319  0.002276  0.002235  0.002193  
    11  0.002431  0.002375  0.002327  0.002283  0.002242  0.002199  
    12  0.002436  0.002381  0.002332  0.002289  0.002247  0.002204  
    13  0.002439  0.002384  0.002336  0.002293  0.002251  0.002208  
    14  0.002439  0.002385  0.002338  0.002295  0.002254  0.002211  
    15  0.002439  0.002385  0.002338  0.002296  0.002255  0.002214  
    16  0.002438  0.002384  0.002338  0.002296  0.002256  0.002215  
    17  0.002438  0.002384  0.002337  0.002296  0.002256  0.002215  
    18  0.002438  0.002384  0.002337  0.002296  0.002256  0.002216  
    19  0.002439  0.002385  0.002338  0.002296  0.002256  0.002217  
    20  0.002441  0.002385  0.002338  0.002296  0.002256  0.002217  
    21  0.002443  0.002387  0.002339  0.002296  0.002257  0.002217  
    22  0.002445  0.002388  0.002340  0.002297  0.002257  0.002217  
    23  0.002447  0.002390  0.002341  0.002298  0.002257  0.002217  
    24  0.002450  0.002392  0.002342  0.002299  0.002258  0.002218  
    25  0.002453  0.002394  0.002344  0.002300  0.002259  0.002218  
    26  0.002456  0.002397  0.002346  0.002302  0.002261  0.002219  
    27  0.002459  0.002400  0.002349  0.002304  0.002262  0.002220  
    28  0.002462  0.002402  0.002351  0.002306  0.002264  0.002221  
    29  0.002464  0.002405  0.002353  0.002308  0.002265  0.002222  
    30  0.002466  0.002406  0.002355  0.002309  0.002267  0.002224  
    31  0.002467  0.002408  0.002357  0.002311  0.002268  0.002225  
    32  0.002468  0.002409  0.002358  0.002312  0.002270  0.002227  
    33  0.002468  0.002410  0.002359  0.002313  0.002271  0.002228  
    34  0.002469  0.002410  0.002360  0.002314  0.002272  0.002229  
    35  0.002469  0.002411  0.002360  0.002315  0.002273  0.002230  
    36  0.002469  0.002411  0.002361  0.002316  0.002274  0.002231  
    37  0.002469  0.002412  0.002362  0.002317  0.002275  0.002232  
    38  0.002469  0.002411  0.002362  0.002318  0.002275  0.002233  
    39  0.002468  0.002411  0.002362  0.002318  0.002276  0.002233  
    40  0.002467  0.002410  0.002361  0.002317  0.002275  0.002233  
    41  0.002466  0.002409  0.002360  0.002316  0.002274  0.002232  
    42  0.002466  0.002408  0.002359  0.002315  0.002273  0.002230  
    43  0.002466  0.002407  0.002358  0.002313  0.002271  0.002229  
    44  0.002467  0.002408  0.002358  0.002313  0.002270  0.002227  
    45  0.002469  0.002409  0.002358  0.002313  0.002269  0.002226  
    46  0.002471  0.002411  0.002360  0.002313  0.002270  0.002226  
    47  0.002474  0.002414  0.002362  0.002315  0.002271  0.002227  
    48  0.002477  0.002417  0.002365  0.002317  0.002273  0.002228  
    49  0.002480  0.002420  0.002368  0.002320  0.002275  0.002230  
    50  0.002483  0.002423  0.002371  0.002323  0.002278  0.002232  
    51  0.002486  0.002426  0.002374  0.002326  0.002280  0.002235  
    52  0.002488  0.002429  0.002377  0.002329  0.002283  0.002238  
    53  0.002491  0.002431  0.002379  0.002331  0.002285  0.002240  
    54  0.002493  0.002433  0.002381  0.002333  0.002287  0.002241  
    55  0.002495  0.002435  0.002383  0.002335  0.002288  0.002243  
    56  0.002498  0.002437  0.002385  0.002336  0.002290  0.002243  
    57  0.002501  0.002439  0.002386  0.002337  0.002291  0.002244  
    58  0.002503  0.002441  0.002388  0.002339  0.002291  0.002245  
    59  0.002506  0.002443  0.002389  0.002340  0.002293  0.002246  
    60  0.002509  0.002446  0.002391  0.002341  0.002294  0.002247  
    61  0.002512  0.002448  0.002393  0.002343  0.002295  0.002248  
    62  0.002516  0.002451  0.002395  0.002345  0.002297  0.002251  
    63  0.002521  0.002455  0.002398  0.002347  0.002300  0.002254  
    64  0.002527  0.002461  0.002403  0.002352  0.002304  0.002258  
    65  0.002535  0.002468  0.002409  0.002357  0.002310  0.002263  
    66  0.002545  0.002477  0.002417  0.002365  0.002317  0.002270  
    67  0.002556  0.002487  0.002427  0.002374  0.002326  0.002278  
    68  0.002567  0.002498  0.002438  0.002385  0.002336  0.002288  
    69  0.002579  0.002510  0.002449  0.002396  0.002348  0.002299  
    70  0.002589  0.002520  0.002461  0.002408  0.002359  0.002311  
    71  0.002598  0.002530  0.002471  0.002419  0.002371  0.002322  
    72  0.002605  0.002538  0.002480  0.002429  0.002381  0.002333  
    73  0.002610  0.002544  0.002487  0.002436  0.002390  0.002343  
    74  0.002613  0.002549  0.002492  0.002442  0.002397  0.002351  
    75  0.002614  0.002551  0.002495  0.002446  0.002401  0.002357  
    76  0.002614  0.002552  0.002497  0.002448  0.002404  0.002360  
    77  0.002613  0.002551  0.002496  0.002448  0.002405  0.002362  
    78  0.002612  0.002550  0.002496  0.002448  0.002404  0.002361  
    79  0.002611  0.002549  0.002495  0.002446  0.002403  0.002360  
    80  0.002611  0.002549  0.002494  0.002445  0.002401  0.002358  
    81  0.002612  0.002549  0.002494  0.002445  0.002401  0.002357  
    82  0.002614  0.002550  0.002495  0.002446  0.002401  0.002356  
    83  0.002616  0.002552  0.002498  0.002449  0.002403  0.002358  
    84  0.002619  0.002555  0.002501  0.002452  0.002406  0.002361  
    85  0.002623  0.002559  0.002504  0.002456  0.002410  0.002365  
    86  0.002627  0.002563  0.002508  0.002460  0.002415  0.002370  
    87  0.002630  0.002566  0.002512  0.002464  0.002420  0.002376  
    88  0.002633  0.002569  0.002515  0.002468  0.002425  0.002382  
    89  0.002634  0.002571  0.002518  0.002472  0.002429  0.002387  
    90  0.002634  0.002572  0.002519  0.002474  0.002433  0.002392  
    91  0.002632  0.002571  0.002519  0.002475  0.002435  0.002395  
    92  0.002629  0.002569  0.002518  0.002475  0.002436  0.002398  
    93  0.002625  0.002566  0.002516  0.002474  0.002436  0.002399  
    94  0.002621  0.002563  0.002514  0.002473  0.002436  0.002400  
    95  0.002618  0.002560  0.002512  0.002471  0.002435  0.002400  
    96  0.002616  0.002558  0.002510  0.002470  0.002435  0.002399  
    97  0.002615  0.002558  0.002510  0.002470  0.002434  0.002399  
    98  0.002617  0.002559  0.002512  0.002471  0.002435  0.002399  
    99  0.002621  0.002563  0.002515  0.002474  0.002437  0.002400  
    
    [100 rows x 3601 columns]
    


```python
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import netCDF4 as nc

# Load the projected DEM and ground track data
ground_track_file = "C:/Users/HP/Downloads/enhanced_measurement.nc"
ground_track_dataset = nc.Dataset(ground_track_file)
latitude = ground_track_dataset.variables['lat_20_ku'][:]
longitude = ground_track_dataset.variables['lon_20_ku'][:]

# Create a KDTree from the valid points in the projected DEM
valid_indices = np.argwhere(~np.isnan(dem_projected))
valid_points = np.column_stack((longitude[valid_indices[:, 1]], latitude[valid_indices[:, 0]]))
tree = cKDTree(valid_points)

# Find the nearest neighbors for each ground track point
ground_track_points = np.column_stack((longitude, latitude))
distances, indices = tree.query(ground_track_points)

# Identify the intersection points
intersection_indices = np.where(distances == 0)[0]
intersection_points = ground_track_points[intersection_indices]

# Create a DataFrame for the intersection points
intersection_df = pd.DataFrame(intersection_points, columns=['Longitude', 'Latitude'])

# Output the first 100 intersection points
print("Intersection Points:")
print(intersection_df.head(200))
```

    Intersection Points:
         Longitude   Latitude
    0    84.381890  18.260717
    1    84.381202  18.263608
    2    84.380514  18.266499
    3    84.379826  18.269390
    4    84.379138  18.272281
    5    84.378450  18.275171
    6    84.377762  18.278062
    7    84.377073  18.280953
    8    84.376385  18.283844
    9    84.375697  18.286735
    10   84.375009  18.289626
    11   84.374321  18.292517
    12   84.373632  18.295407
    13   84.372944  18.298298
    14   84.372256  18.301187
    15   84.371568  18.304078
    16   84.370880  18.306968
    17   84.370191  18.309859
    18   84.369503  18.312750
    19   84.368815  18.315641
    20   84.368127  18.318531
    21   84.367438  18.321422
    22   84.366750  18.324312
    23   84.366062  18.327203
    24   84.365373  18.330094
    25   84.364685  18.332984
    26   84.363996  18.335875
    27   84.363308  18.338766
    28   84.362619  18.341657
    29   84.361931  18.344548
    30   84.361242  18.347439
    31   84.360554  18.350330
    32   84.359865  18.353220
    33   84.359177  18.356110
    34   84.358488  18.359001
    35   84.357800  18.361892
    36   84.357111  18.364783
    37   84.356422  18.367674
    38   84.355734  18.370565
    39   84.355045  18.373456
    40   84.354356  18.376347
    41   84.353668  18.379238
    42   84.352979  18.382128
    43   84.352290  18.385019
    44   84.351602  18.387909
    45   84.350913  18.390800
    46   84.350224  18.393691
    47   84.349535  18.396581
    48   84.348847  18.399472
    49   84.348158  18.402363
    50   84.347469  18.405253
    51   84.346780  18.408144
    52   84.346091  18.411035
    53   84.345402  18.413926
    54   84.344714  18.416816
    55   84.344025  18.419707
    56   84.343336  18.422598
    57   84.342647  18.425489
    58   84.341958  18.428380
    59   84.341269  18.431271
    60   84.340580  18.434161
    61   84.339891  18.437052
    62   84.339202  18.439943
    63   84.338513  18.442833
    64   84.337824  18.445724
    65   84.337135  18.448614
    66   84.336445  18.451505
    67   84.335756  18.454396
    68   84.335067  18.457287
    69   84.334378  18.460178
    70   84.333689  18.463068
    71   84.333000  18.465959
    72   84.332311  18.468850
    73   84.331621  18.471741
    74   84.330932  18.474631
    75   84.330243  18.477522
    76   84.329554  18.480412
    77   84.328864  18.483303
    78   84.328175  18.486194
    79   84.327486  18.489085
    80   84.326796  18.491976
    81   84.326107  18.494867
    82   84.325418  18.497757
    83   84.324728  18.500648
    84   84.324039  18.503538
    85   84.323350  18.506429
    86   84.322660  18.509319
    87   84.321971  18.512209
    88   84.321281  18.515101
    89   84.320592  18.517991
    90   84.319902  18.520882
    91   84.319213  18.523773
    92   84.318523  18.526664
    93   84.317834  18.529554
    94   84.317144  18.532445
    95   84.316455  18.535336
    96   84.315765  18.538227
    97   84.315075  18.541118
    98   84.314386  18.544008
    99   84.313696  18.546898
    100  84.313007  18.549789
    101  84.312317  18.552680
    102  84.311627  18.555570
    103  84.310938  18.558461
    104  84.310248  18.561352
    105  84.309558  18.564243
    106  84.308868  18.567134
    107  84.308178  18.570025
    108  84.307489  18.572915
    109  84.306799  18.575806
    110  84.306109  18.578697
    111  84.305419  18.581588
    112  84.304729  18.584478
    113  84.304040  18.587368
    114  84.303350  18.590259
    115  84.302660  18.593149
    116  84.301970  18.596040
    117  84.301280  18.598930
    118  84.300590  18.601821
    119  84.299900  18.604712
    120  84.299210  18.607602
    121  84.298520  18.610493
    122  84.297830  18.613383
    123  84.297140  18.616274
    124  84.296450  18.619164
    125  84.295760  18.622055
    126  84.295070  18.624946
    127  84.294380  18.627836
    128  84.293690  18.630726
    129  84.293000  18.633617
    130  84.292310  18.636508
    131  84.291619  18.639398
    132  84.290929  18.642289
    133  84.290239  18.645180
    134  84.289549  18.648071
    135  84.288859  18.650961
    136  84.288168  18.653851
    137  84.287478  18.656741
    138  84.286788  18.659631
    139  84.286098  18.662521
    140  84.285408  18.665412
    141  84.284717  18.668303
    142  84.284027  18.671193
    143  84.283337  18.674083
    144  84.282647  18.676973
    145  84.281957  18.679863
    146  84.281266  18.682753
    147  84.280576  18.685644
    148  84.279885  18.688534
    149  84.279195  18.691424
    150  84.278505  18.694314
    151  84.277814  18.697204
    152  84.277124  18.700094
    153  84.276433  18.702985
    154  84.275743  18.705875
    155  84.275053  18.708765
    156  84.274362  18.711655
    157  84.273672  18.714545
    158  84.272981  18.717435
    159  84.272291  18.720325
    160  84.271600  18.723215
    161  84.270910  18.726105
    162  84.270219  18.728995
    163  84.269529  18.731885
    164  84.268838  18.734775
    165  84.268147  18.737665
    166  84.267457  18.740557
    167  84.266766  18.743447
    168  84.266075  18.746336
    169  84.265385  18.749226
    170  84.264694  18.752116
    171  84.264003  18.755007
    172  84.263313  18.757897
    173  84.262622  18.760787
    174  84.261931  18.763677
    175  84.261240  18.766566
    176  84.260549  18.769457
    177  84.259859  18.772348
    178  84.259168  18.775238
    179  84.258477  18.778128
    180  84.257786  18.781017
    181  84.257095  18.783908
    182  84.256404  18.786798
    183  84.255713  18.789688
    184  84.255023  18.792578
    185  84.254331  18.795468
    186  84.253641  18.798358
    187  84.252950  18.801248
    188  84.252259  18.804138
    189  84.251568  18.807028
    190  84.250876  18.809919
    191  84.250185  18.812809
    192  84.249494  18.815699
    193  84.248803  18.818589
    194  84.248112  18.821479
    195  84.247421  18.824370
    196  84.246730  18.827260
    197  84.246039  18.830150
    198  84.245348  18.833040
    199  84.244656  18.835930
    


```python
#"C:/Users/HP/OneDrive/Desktop/Surge Project/Kangra_Chamba_Hills.tif"
```

![1.1.png](attachment:1.1.png)

![1.2.png](attachment:1.2.png)

![1.3.png](attachment:1.3.png)

![image.png](attachment:image.png)

![1.4.png](attachment:1.4.png)

![1.5.png](attachment:1.5.png)

![1.6.png](attachment:1.6.png)


```python
import xarray as xr
import rasterio

# Load the ground track data
netcdf_file = "C:/Users/HP/Downloads/enhanced_measurement.nc"
ground_track_data = xr.open_dataset(netcdf_file)

# Load the GeoTIFF files
flow_accumulation_file = "C:/Users/HP/OneDrive/Desktop/Surge Project/QGIS Files 1/Flow_Accumulation_Yamuna.tif"
flow_direction_file = "C:/Users/HP/OneDrive/Desktop/Surge Project/QGIS Files 1/Flow_Direction_Yamuna.tif"
sink_watershed_file = "C:/Users/HP/OneDrive/Desktop/Surge Project/QGIS Files 1/Sink_Watershed_Yamuna.tif"
topographic_convergence_file = "C:/Users/HP/OneDrive/Desktop/Surge Project/QGIS Files 1/Topographic_Convergence_Index_Yamuna.tif"

# Open the GeoTIFF files
flow_accumulation_data = rasterio.open(flow_accumulation_file)
flow_direction_data = rasterio.open(flow_direction_file)
sink_watershed_data = rasterio.open(sink_watershed_file)
topographic_convergence_data = rasterio.open(topographic_convergence_file)
```


```python
import numpy as np
import matplotlib.pyplot as plt

# Threshold value for flow accumulation to determine river network
threshold_value = 10

# Read the flow accumulation data as a numpy array
flow_accumulation_array = flow_accumulation_data.read(1)

# Apply thresholding to extract the river network
river_network = np.where(flow_accumulation_array >= threshold_value, 1, 0)

# Create a figure and axes
fig, ax = plt.subplots()

# Display the river network array
choice = 'Blues'
ax.imshow(river_network, cmap=choice)

# Add a colorbar for reference
cbar = plt.colorbar(ax.imshow(river_network, cmap=choice), ax=ax)

# Show the plot
plt.show()

# Find the outlet point coordinates
outlet_points = np.where(river_network == 1)

# Get the first outlet point coordinates
outlet_latitude = outlet_points[0][0]
outlet_longitude = outlet_points[1][0]

# Print the coordinates of the outlet point
print("Outlet Point Coordinates:")
print("Latitude:", outlet_latitude)
print("Longitude:", outlet_longitude)

```


    
![png](output_77_0.png)
    


    Outlet Point Coordinates:
    Latitude: 0
    Longitude: 4
    


```python
# Threshold value for flow accumulation to determine river network
threshold_value = 10

# Read the flow accumulation data as a numpy array
flow_accumulation_array = flow_accumulation_data.read(1)

# Convert the flow accumulation array to logarithmic scale
log_flow_accumulation = np.log(flow_accumulation_array)

# Apply thresholding to extract the river network
river_network = np.where(flow_accumulation_array >= threshold_value, 1, 0)

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the log flow accumulation array plot thr flow accumu
im = ax.imshow(river_network, cmap='binary')

# Add a colorbar to the plot
cbar = plt.colorbar(im, ax=ax)

# Add title and labels to the plot
ax.set_title('Flow Accumulation (Log Scale)')

# Show the plot
plt.show()

```


    
![png](output_78_0.png)
    



```python
import cv2

# Apply thresholding to extract the river network
river_network = np.where(flow_accumulation_array >= threshold_value, 1, 0)

# Find contours in the binary image using cv2.RETR_CCOMP retrieval mode
contours, hierarchy = cv2.findContours(river_network.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# Create an empty list to store the vectorized river network
vector_lines = []

# Iterate over each contour
for contour in contours:
    # Approximate the contour to reduce the number of vertices
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # Append the simplified contour as a line to the vector lines list
    vector_lines.append(approx.squeeze())

# Create a new figure
fig, ax = plt.subplots()

# Iterate over each line segment in 'vector_lines'
for line in vector_lines:
    # Check if the line segment has more than one point (is 2-dimensional)
    if line.ndim == 2 and line.shape[0] > 1:
        # Extract x and y coordinates from the line segment
        x = line[:, 0]
        y = line[:, 1]

        # Plot the line segment
        ax.plot(x, y, color='blue')

# Set the aspect ratio and axis labels
ax.set_aspect('equal')

# Display the plot
plt.show()

```


    
![png](output_79_0.png)
    


![image.png](attachment:image.png)

![image.png](attachment:image.png)


```python
import shapefile

# Specify the path to your .shx file
shx_file_path = 'C:/Users/HP/Downloads/HydroRIVERS_v10_as_shp/HydroRIVERS_v10_as_shp/HydroRIVERS_v10_as.shx'

# Open the .shx file
shx_reader = shapefile.Reader(shx_file_path)

# Get the shapefile fields
fields = shx_reader.fields[1:]  # Exclude the DeletionFlag field
field_names = [field[0] for field in fields]

# Get the shapefile records (attributes)
records = shx_reader.records()

# Get the shapefile shapes (geometry)
shapes = shx_reader.shapes()

# Print the field names
print("Field names:")
print(field_names)

# Print the first few records
print("\nRecords:")
for record in records[:5]:
    print(record)

# Print the first few shapes (geometry)
print("\nShapes (Geometry):")
for shape in shapes[:5]:
    print(shape)

```

    Field names:
    ['HYRIV_ID', 'NEXT_DOWN', 'MAIN_RIV', 'LENGTH_KM', 'DIST_DN_KM', 'DIST_UP_KM', 'CATCH_SKM', 'UPLAND_SKM', 'ENDORHEIC', 'DIS_AV_CMS', 'ORD_STRA', 'ORD_CLAS', 'ORD_FLOW', 'HYBAS_L12']
    
    Records:
    Record #0: [40000001, 40000019, 40017702, 2.48, 3056.6, 6.9, 14.93, 14.9, 0, 0.133, 1, 5, 7, 4121166050]
    Record #1: [40000002, 40000019, 40017702, 1.23, 3056.8, 6.0, 12.76, 12.8, 0, 0.12, 1, 6, 7, 4121166050]
    Record #2: [40000003, 40000015, 40017702, 1.26, 3052.8, 7.4, 11.8, 11.8, 0, 0.1, 1, 7, 7, 4120080930]
    Record #3: [40000004, 40000015, 40017702, 1.3, 3052.6, 7.9, 2.17, 24.9, 0, 0.195, 2, 6, 7, 4120080930]
    Record #4: [40000005, 40000004, 40017702, 0.69, 3054.1, 6.4, 10.84, 10.8, 0, 0.09, 1, 7, 8, 4120080930]
    
    Shapes (Geometry):
    Shape #0: POLYLINE
    Shape #1: POLYLINE
    Shape #2: POLYLINE
    Shape #3: POLYLINE
    Shape #4: POLYLINE
    
