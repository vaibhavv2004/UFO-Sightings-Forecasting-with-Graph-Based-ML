# UFO-Sightings-Forecasting-with-Graph-Based-ML
End-to-end project that analyzes global UFO sighting data (1900–2024), builds a region similarity graph, and trains machine learning models to forecast monthly sightings and predict 2026 hotspot regions.


### Objective  
The UFO-Sightings-Forecasting-Graph-ML project implements an end-to-end spatio-temporal forecasting pipeline that transforms historical UFO sighting data into actionable predictions. By modeling regions as a graph network based on temporal similarity patterns and applying machine learning forecasting models, the system predicts monthly UFO sighting counts for 2025–2026 and identifies emerging hotspot regions. This project demonstrates how graph analytics combined with time-series ML can uncover hidden patterns in large-scale event data—applicable to domains like crime prediction, incident monitoring, and anomaly surveillance.

### Skills Learned  
- Spatio-temporal data preprocessing and feature engineering from raw event records
- Graph construction and network analysis using temporal correlation and similarity metrics
- Community detection and centrality analysis (Louvain algorithm, PageRank, betweenness)
- Time-series forecasting with supervised machine learning (lag features, rolling predictions)
- Model training, evaluation, and hyperparameter tuning with scikit-learn
- Iterative multi-step forecasting for future period predictions
- Data visualization and storytelling with matplotlib and pandas 

### Tools & Technologies Used  
- Python 3.x – Core programming language
- pandas & NumPy – Data manipulation and aggregation
- NetworkX – Graph construction, centrality metrics, and community detection
- scikit-learn – Random Forest regression, model evaluation (MAE, RMSE)
- matplotlib & seaborn – Time-series and hotspot visualizations
- dateutil – Datetime manipulation for rolling forecasts 

### Key Features  
- 100,000 UFO sighting records spanning 1900–2024 with attributes: date, location, shape, duration, airport distance, evidence flag, credibility score
- 8-region global network constructed via monthly sighting correlation (absolute threshold = 0.01), yielding 18 edge
- Graph analytics: degree centrality, weighted degree, betweenness, PageRank, and Louvain community detection identifying 2 main communities
- Random Forest forecasting model trained on lag features (3 months) + graph features + temporal features
→ Train MAE: 0.78, Test MAE: 2.04 (≈2 sightings error per region-month)
- Iterative 2025–2026 rolling forecast generating monthly predictions for all regions
- 2026 hotspot ranking: Canada (126 total predicted), Brazil (118), India (112) identified as top hotspot regions
- Visualizations: global time-series, 2026 hotspot bar chart, 2024 vs 2026 comparison, multi-region forecast curves


 ### Methodology & Pipeline
1. Data Loading & Cleaning
- Parse dates with dayfirst=True for DD-MM-YYYY format
- Extract time features: year, month, year_month
- Clean numeric fields (duration, airport distance) with pd.to_numeric(..., errors="coerce")

2. Monthly Aggregation
- Group by (region, year_month) → sightings_count, avg_duration, avg_credibility, avg_airport_distance
-Create global monthly series for trend analysis

3. Graph Construction
- Pivot to region × month matrix
- Compute pairwise Pearson correlation → take absolute values
- Build NetworkX graph with edges where |corr| ≥ 0.01
- Compute centrality metrics and Louvain communities

4. Feature Engineering
- Merge graph features (degree, PageRank, community_id) into monthly dataset
- Create lag features (lag_1, lag_2, lag_3) per region
- Add temporal features (month_num, year)

5. Model Training & Evaluation
- Train/val/test split by time (train ≤2015, val 2016–2019, test 2020–2024)
-Train Random Forest (300 estimators) on 12 features
-Evaluate with MAE and RMSE

6. Rolling Forecast (2025–2026)
- Seed lags from last 3 months of 2024
- Iteratively predict next month, update lags, roll forward 24 months
- Aggregate 2026 totals per region for hotspot ranking


### Results & Impact
- Validated forecasting accuracy: MAE ≈ 2 sightings per region-month on test data (2020–2024)
- 2026 hotspot predictions: Canada, Brazil, India top the list with 110–126 predicted sightings
- Network insights: Germany and India emerge as central "hub" regions with highest PageRank and degree centrality, despite moderate sighting counts
- Transferable framework: demonstrates how graph-based features improve time-series forecasting and how this approach applies to crime hotspot detection, disease outbreak prediction, and infrastructure anomaly monitoring

  
### Demo Screenshots  

## Fig 1: Global Monthly UFO Sightings (1900–2024)
https://github.com/vaibhavv2004/UFO-Sightings-Forecasting-with-Graph-Based-ML/blob/main/global%20sighting(1900-2024.jpeg
Historical time-series showing trends and seasonality across 1500 months

## Fig 2: Predicted 2026 Hotspot Regions (Bar Chart)
https://github.com/vaibhavv2004/UFO-Sightings-Forecasting-with-Graph-Based-ML/blob/main/Predicted%20ufo%20sighting%20(2026).jpeg
Canada leads with 126 predicted sightings, followed by Brazil and India

## Fig 3: 2024 vs 2026 Comparison
https://github.com/vaibhavv2004/UFO-Sightings-Forecasting-with-Graph-Based-ML/blob/main/Actual%20vs%20predicted.jpeg
Side-by-side bar chart showing actual 2024 counts vs forecasted 2026 totals per region

## Fig 4: Monthly Forecast Curves for Top Regions (2025–2026)
https://github.com/vaibhavv2004/UFO-Sightings-Forecasting-with-Graph-Based-ML/blob/main/Predicted%20monthly%20ufo%20by%20region.jpeg

Multi-line plot showing predicted monthly sightings for Canada, Brazil, India over 24 months

## code
https://github.com/vaibhavv2004/UFO-Sightings-Forecasting-with-Graph-Based-ML/blob/main/ufo_project.ipynb
