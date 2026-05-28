# RetailPulse Customer Segmentation Report

- KMeans silhouette score: 0.9164
- DBSCAN silhouette score: not available
- DBSCAN eps: 0.65
- DBSCAN min_samples: 10

## KMeans Cluster Profile

|   kmeans_cluster |   Recency |   Frequency |   Monetary |
|-----------------:|----------:|------------:|-----------:|
|                0 |    202    |        5.78 |    2366.66 |
|                1 |     23.09 |      143.05 |  176558    |

## DBSCAN Cluster Profile

|   dbscan_cluster |   Recency |   Frequency |   Monetary |
|-----------------:|----------:|------------:|-----------:|
|               -1 |     78.12 |       83.48 |   87760.7  |
|                0 |    202.69 |        5.44 |    2085.78 |

## Persona Distribution

| persona      |   customer_count |
|:-------------|-----------------:|
| Big Spenders |             5856 |
| Champions    |               22 |

## Business Insights

- Champions and Loyal Customers should be prioritized for retention, upsell, and loyalty campaigns.
- Big Spenders are ideal for premium bundles, exclusive offers, and white-glove service.
- At Risk and Lost customers should be targeted with win-back journeys and operational root-cause analysis.
- The cluster structure should be revisited after each major promotion cycle to keep the personas current.