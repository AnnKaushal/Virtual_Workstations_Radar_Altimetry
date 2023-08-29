# Virtual_Workstations_Radar_Altimetry

# Estimating optimal location of Virtual Stations using Satellite Radar Altimetry and QGIS.

This research project is a part of my IIT Kanpur's Summer Research Internship program conducted by SURGE for the year 2023 under the guidance of my mentor, Dr. Balaji Devaraju. In this project, I try to compute the intersection points between satellite ground track data and the Ganga and Yamuna river bodies in UP, India. These virtual stations are of significant importance in scientific analysis and interpretation related to Earth's topography and hydrological processes.

## Table of Contents
1. Abstract
2. Acknowledgment
3. Principle of Satellite Radar Altimetry
4. Methodology 
 4.1. River width estimation
 4.2. River network extraction.
 4.3. Finding intersection points.

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









