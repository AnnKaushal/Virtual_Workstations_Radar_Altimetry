# Virtual_Workstations_Radar_Altimetry

# Estimating optimal location of Virtual Stations using Satellite Radar Altimetry and QGIS.

This research project is a part of my IIT Kanpur's Summer Research Internship program conducted by SURGE for the year 2023 under the guidance of my mentor, Dr. Balaji Devaraju. In this project, I try to compute the intersection points between satellite ground track data and the Ganga and Yamuna river bodies in UP, India. These virtual stations are of significant importance in scientific analysis and interpretation related to Earth's topography and hydrological processes.

## Table of Contents
1. Abstract
2. Acknowledgment
3. Principle of Satellite Radar Altimetry
4. River width estimation
5. River network extraction
6. Finding intersection points.

## Abstract

Satellite altimetry is among the widely used ocean remote sensing techniques. It measures sea surface topography and offers valuable insights into Earth’s spatial & temporal changes, river dynamics, and soil’s moisture contents. As a means of estimating these changes, we need to first define virtual stations, these being the points of intersection between a water body surface and a satellite ground track. 

This project report aims to precisely locate optimal virtual stations, state the procedure on how to get there, and then briefly discuss the results we get.

We utilise two primary datasets: Shuttle Radar Topography Mission’s digital elevation model (SRTM-DEM) data obtained from Google Earth Engine and the United States Geological Survey (USGS) site, along with ground track waveform data acquired from Copernicus Open Access Hub. The DEM dataset is initially loaded, and its elevation band information is extracted and organised into a data frame. Various plotting techniques are employed to visualise the dataset, and the gradient of a smoothed elevation matrix is computed. Potential channel locations are identified, and their geographical representation is visualised using a folium map. Subsequently, the estimation of river width as a function of each channel location is performed. The dataset is further utilised to compute flow accumulation using QGIS software for the selected river basin, which serves as the area of interest. The drainage network is extracted by applying a threshold value to the resulting flow accumulation map. Finally, the intersection between this extracted drainage network and the ground track data yields the precise coordinates of the corresponding virtual stations.

Accurate virtual station estimation is important because it provides crucial reference points for understanding Earth's topography, bridging spatial data gaps, and enhancing radar altimetry data processing. These locations supplement models and algorithms, improving accuracy and enabling comprehensive analysis of climate patterns, sea level rise, and hydrological processes. Moreover, virtual station data informs future satellite mission planning and supports scientific research and environmental management.

Keywords used: Satellite altimetry, digital elevation model (DEM), google earth engine (GEE), quantum geographic information system (QGIS) software.

## Acknowledgment

I would like to express my sincere gratitude to all those who have contributed to the successful completion of this project. Their support, guidance, and encouragement have been invaluable throughout this journey and made this endeavor possible in the first place.

I want to start by saying how grateful I am to my supervisor, Dr. Balaji Devaraju, for his counsel and expertise. He has been instrumental in helping me understand concepts better and providing valuable feedback that has significantly improved this work.

I would extend my appreciation to Abhilasha mam, a Ph.D. scholar and Shubhi Kant mam, an MTech student who both study here at IIT Kanpur, for always lending a helping hand whenever I needed it or had a doubt or question.

I am also obliged to IIT Kanpur Institute for selecting me through their Surge Program and giving me this opportunity to learn and explore.

Lastly, I would love to mention my parents, my brother, and all my friends here at IIT Kanpur who stayed with me throughout and kept my spirits high whenever things got rough. Their understanding, patience, and encouragement have been the pillars that propelled me forward.

In closing, I would like to acknowledge the trailblazers and visionaries whose groundbreaking work has paved the way for this study. Their tireless pursuit of knowledge and passion for discovery continues to inspire and fuel our collective progress.

Thank you all for your contributions.

## Principle of Satellite Radar Altimetry

The altimeter emits a radar pulse towards the nadir direction (directly below the observer) and measures the time it takes back for the pulse to reflect off the Earth's surface and return to the altimeter. 

By knowing the speed of wave propagation through the traversed medium, the travel time enables the calculation of the distance (R) between the altimeter and the reflecting surface. Given the satellite's orbit (and consequently its height, H, relative to an ellipsoid), the height of the reflective surface (h) can be derived as h = H - R. 

However, practical measurements require various corrections to account for atmospheric disturbances, ionosphere effects, and solid and liquid earth tides.

## Methodology

The process can be broadly divided into 2 segments:
1. River Width Estimation.
2. River Network Extraction.
3. Finding Intersection Points.

We undertake a series of multiple steps and side steps to estimate river width from satellite data and extract the river's drainage network using Python code and QGIS software. The resulting datasets obtained are used to compute optimal location of valid intersection points. The details of each step are mentioned in the SURGE project report file. Please read the report for detailed visual explanation.

---

## 1. Estimating Virtual Station's Location

## Overview
This code aims to estimate the location of a virtual station using elevation data from the Shuttle Radar Topography Mission (SRTM) dataset and various geospatial tools. It covers several tasks, including importing elevation data, processing it, and visualizing the results.

## Prerequisites
Before running this code, please ensure that you have the following libraries and tools installed:

- Python
- Earth Engine Python API
- NumPy
- Pandas
- SciPy
- Matplotlib
- Rasterio
- Scikit-image

Additionally, you need to authenticate and initialize the Earth Engine API by running `ee.Authenticate()` and `ee.Initialize()`.

## Code Sections
1. **Import Necessary Libraries and Packages**: This section imports the required Python libraries.

2. **Get the Earth Engine's DEM Elevation Band Information in a DataFrame**: Retrieves elevation data from the SRTM dataset using the Earth Engine API and displays relevant information in a DataFrame.

3. **Display the SRTM DEM Dataset's Info Obtained from USGS**: Loads elevation data from a GeoTIFF file and displays its metadata.

4. **Print the Elevation Array of the Earth Engine's DEM Dataset for Ganga River**: Extracts elevation data for a specific geographic area and prints the elevation values.

5. **Plot the Elevation Profile of the Earth Engine's Elevation Dataset**: This section plots an elevation profile and calculates statistics like minimum, maximum, mean, median, and standard deviation of the elevation data.

6. **Plot the Heatmap and 3D Surface of the Earth Engine's Elevation Dataset**: Creates a heatmap and 3D surface plot of the elevation data.

7. **Apply Gaussian Filter to Smoothen the Earth Engine's DEM**: Applies a Gaussian filter to smooth the elevation data and visualizes the results.

8. **Apply Mean Filter to Smoothen the Earth Engine's DEM**: Demonstrates the application of a mean filter to smooth the elevation data.

9. **Apply Bilateral Filter to Smoothen the Earth Engine's DEM**: This section applies a bilateral filter for edge-preserving smoothing of the elevation data.

## Usage
1. Ensure you have the required libraries and tools installed as mentioned in the prerequisites.

2. Authenticate and initialize the Earth Engine API using `ee.Authenticate()` and `ee.Initialize()`.

3. Run the code sections as needed based on your requirements. You can copy and paste individual sections into your Python environment.

4. Adjust parameters and input data paths as necessary.

5. Observe the output visualizations and analysis results as described in the code comments.

---

## 2. Compute and Visualize the Gradient Magnitude of the Smoothed Earth Engine's DEM

This section calculates the gradient magnitude of the smoothed DEM using both mean and Gaussian filters and then visualizes the results.

### Compute and Visualize the Gradient Direction of the Smoothed Earth Engine's DEM

Here, the code calculates the gradient direction of the smoothed DEM using a mean filter. It visualizes the gradient direction using arrows and a normalized gradient direction map.

### Compute and Visualize the Respective Histograms of Each Color and All Colors Combined of the Smoothed Earth Engine's DEM

This part focuses on computing and visualizing histograms of the gradient direction for individual color channels (red, green, and blue) and a combined histogram for all colors. It provides insights into the distribution of gradient direction values.

### Compute and Visualize the Gradient Magnitude, Gradient Direction, and Respective Histograms of the USGS DEM

This section deals with a different DEM dataset (USGS DEM). It performs similar operations as previous sections, including smoothing, gradient magnitude, gradient direction computation, and histograms. It allows you to compare these results with those of the Earth Engine's DEM.

### Compute and Visualize the Gradient Magnitude, Gradient Direction, and Respective Histograms of the Hilly Terrain

Similar to the previous section, this part works with another terrain dataset (Hilly Terrain). It calculates gradient magnitude, gradient direction, and respective histograms. This can be useful for analyzing different types of terrain.

---

## References

Berry, P. A. M. (2006) Two decades of inland water monitoring using satellite radar altimetry In: 15 Years of Progress in Radar Altimetry (Proc. Symp., Venice Lido, Italy, 13–18 March 2006). European Space Agency Special Publ. ESA SP614. ESA, Noordwijk, The Netherlands.

Chelton, D. B., Ries, J. C., Haines, B. J., Fu, L. & Callahan, P. S. (2001) Satellite altimetry. In: Satellite Altimetry and Earth Sciences: A Handbook of Techniques and Applications (ed. by J. Fu & A. Cazenave). Academic Press, San Diego, California, USA.

Roux, E., Santos da Silva, J., Vieira Getirana, A. C., Bonnet, M.-P., Calmant, S., Martinez, J.-M. & Seyler, F. (2010) Producing time series of river water height by means of satellite radar altimetry—comparative study. Hydrol. Sci. J. 55(1), 104–120.

Flow computation on massive grids. Lars Arge, Jeffrey S. Chase, Patrick N. Halpin, Laura Toma, Jeffrey S. Vitter, Dean Urban and Rajiv Wickremesinghe. In Proc. ACM Symposium on Advances in Geographic Information Systems, 2001.

GRASS Development Team (2006) Geographic Resources Analysis Support System (GRASS) Software. ITC-irst, Trento, Italy. http://grass.itc.it.

Berry, P. A. M., Garlick, J. D., Freeman, J. A. & Mathers, E. L. (2005) Global inland water monitoring from multi-mission altimetry. Geophys. Res. Lett. 32(L16401), 1–4.

Saunders, W. (1999) Preparation of DEMs for use in environmental modeling analysis. ESRI User Conference (24–30 July 1999, San Diego, California). Available at: http://proceedings.esri.com/library/userconf/proc99/navigate/proceed.htm

--- 

You can run these code sections as needed and adjust parameters to suit your specific analysis. Each section provides insights into the gradient properties and distribution of elevation data. Please feel free to reach out if you have any questions or need further assistance.

For any questions or issues, please contact 20bce034@nith.ac.in.

Enjoy exploring and analyzing elevation data!

---










