# AI-Driven Metro Infrastructure and Operations Analysis for DMRC

## Table of Contents
- [Introduction](#introduction)
- [Data Source](#data-source)
- [Installation](#installation)
- [Usage](#usage)
- [Data Generation](#data-generation)
- [Simulations](#simulations)
- [Visualizations](#visualizations)
- [Results and Discussion](#results-and-discussion)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)
- [Requirements File](#requirements-file)

## Introduction

This project focuses on enhancing the operational efficiency and passenger experience of the Delhi Metro Rail Corporation (DMRC) system through an AI-driven agent-based simulation model. Specifically targeting the Magenta Line, which serves around 500,000 daily passengers, the project aims to provide actionable insights into crowd management, train occupancy, and platform crowding.

## Data Source

- **GTFS Data**: Acquired from the official DMRC website, including detailed information on stations, routes, trips, and schedules.
- **Stops File**: Contains station IDs, names, and geographical coordinates specific to the Magenta Line.
- **Stop Times File**: Reflects actual subway schedules, compiled from DMRC's GTFS feeds.
- **Trips File**: Integrates stop times with trip data, focusing on peak and off-peak weekday schedules.

## Installation

To set up this project:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/AI-Driven-Metro-Infrastructure-and-Operations-Analysis-for-DMRC.git

2. **Navigate to the project directory**:

   ```bash
   cd AI-Driven-Metro-Infrastructure-and-Operations-Analysis-for-DMRC

3. **Install the required packages:**:

   ```bash
   pip install -r requirements.txt


## Usage

Once the installation is complete, you can start exploring the data and running the code for analysis, in this sequence:

### Data Preprocessing:

Run the notebook:

[magenta_line_gtfs_generation.ipynb](/code/1.%20magenta_line_gtfs_generation.ipynb)

### Generate Synthetic OD Data:

Run the notebook:

[magenta_line_od_synthetic_generation.ipynb](/code/2.%20magenta_line_od_synthetic_generation.ipynb)

### Run Simulations:

Execute the notebook:

[transit_sim_magenta_line.ipynb](/code/3.%20transit_sim_magenta_line.ipynb)

### Visualize Results:

Use the notebook:

[transit_viz_magenta_line.ipynb](/code/4.%20transit_viz_magenta_line.ipynb)



## Data Generation
- **Demand Data Synthesis**: Generated synthetic Origin-Destination (OD) data for approximately 332,000 passengers across 14 stations using the OD matrix method.
- **Real-Time Data Integration**: Utilized actual ridership statistics and peak hour data from DMRC to calibrate and validate the simulation models.
  
## Simulations
Conducted agent-based simulations to analyze:

- **Passenger Flow Dynamics**: Understanding how passengers move through the network during different times of the day.
- **Station Congestion Levels**: Identifying bottlenecks and peak congestion times at various stations.
- **Train Occupancy Rates**: Monitoring train loads to assess crowding and optimize scheduling.
- **Impact of Operational Changes**: Evaluating how adjustments in schedules or train frequency affect overall system performance.

## Visualizations
Developed comprehensive visualizations to depict:

- **Departure Plots**: Showing passenger departures over time from each station.
- **Train and Traveler** Trajectories: Visualizing the movement of trains and passengers along the Magenta Line.
- **Platform Crowdedness**: Heatmaps and graphs indicating platform occupancy levels at different times.
- **Train Occupancy**: Charts displaying the number of passengers onboard trains throughout their routes.
- **Validation Plots**: Comparing simulated data with actual ridership figures for model validation.

## Results and Discussion
Evaluate the potential benefits of operational strategies and discuss the robustness of the models in predicting real-world scenarios.

## Future Work
- **Enhanced AI Integration**: Incorporate advanced machine learning models for better prediction of passenger behavior and system disruptions.
- **Real-Time Data Feeds**: Integrate live data sources for dynamic simulation adjustments.
- **Expanded Scope**: Extend the model to include other lines of the Delhi Metro for a comprehensive system analysis.
- **Passenger App Development**: Create a user-facing application to provide real-time crowding information and travel suggestions.

## Acknowledgements
We would like to thank the Delhi Metro Rail Corporation for providing the GTFS data and ridership statistics that made this project possible.

## Requirements File
For a list of required packages and their versions, please refer to the requirements.txt file.
