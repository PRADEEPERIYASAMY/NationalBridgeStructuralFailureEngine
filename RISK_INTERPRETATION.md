# NBI Structural Risk Analysis & Vulnerability Report (2025)

This report details the structural risk scoring results computed by the machine learning models on the **624,193 standing bridges** in the 2025 US inventory.

---

## 1. Risk Score Distributions

The risk scores represent estimated structural collapse/closure probability values between `0.0` (0%) and `1.0` (100%):

*   **Scour Risk (Hydraulic Washout)**: Average risk is **4.07%** | Maximum risk peaks at **99.78%**
*   **Structural Deterioration**: Average risk is **74.00%** | Maximum risk peaks at **99.93%**
*   **Vehicle Overload**: Average risk is **6.60%** | Maximum risk peaks at **99.33%**
*   **Vehicle Collision**: Average risk is **4.25%** | Maximum risk peaks at **97.81%**
*   **Fire Damage**: Average risk is **2.18%** | Maximum risk peaks at **91.12%**

### Extreme Outlier Thresholds (99.9th Percentile)
Bridges with scores exceeding these limits represent the top **0.1% most anomalous risk profiles** in the country:
*   **Scour risk threshold**: $\ge 99.55\%$
*   **Deterioration risk threshold**: $\ge 99.85\%$
*   **Overload risk threshold**: $\ge 97.44\%$

---

## 2. Top Identified Vulnerabilities (Top 3 per Category)

By joining the ML model scores back to the 2025 NBI metadata, we isolated the specific facility carried, features intersected, and states for the most vulnerable bridges in the country:

### A. Category: Scour (Hydraulic Wash-Out)
Hydraulic washing out of foundations during flood events is the most common cause of collapse.
1.  **Facility**: `PR 920 7.8 KM` over `WATERWAY` (Puerto Rico)
    *   **Built**: 1973 | **Condition Rating**: 7 (Good) | **Est. Collapse Prob**: **99.78%**
2.  **Facility**: `NORTH TONGASS HWY` over `CARLANNA CREEK` (Alaska)
    *   **Built**: 1973 | **Condition Rating**: 6 (Fair) | **Est. Collapse Prob**: **99.76%**
3.  **Facility**: `ROAD` over `BLACK ROCK CREEK` (Oregon)
    *   **Built**: 1973 | **Condition Rating**: **2 (Critical Condition)** | **Est. Collapse Prob**: **99.74%**
    *   *Note*: The Oregon bridge represents an extremely high-signal threat due to its structural condition rating of 2 (imminent failure).

### B. Category: Structural Deterioration
Steady structural decay of concrete and structural members.
1.  **Facility**: `BOWLING GREEN ROAD` over `THOMPSON CREEK` (Mississippi)
    *   **Built**: 1940 | **Condition Rating**: **3 (Serious Decay)** | **Est. Failure Prob**: **99.93%**
2.  **Facility**: `STH 100 SB-MAYFAIR` over `MENOMONEE RIVER` (Wisconsin)
    *   **Built**: 1968 | **Condition Rating**: **3 (Serious Decay)** | **Est. Failure Prob**: **99.93%**
3.  **Facility**: `TR127` over `CLOSED` (Ohio)
    *   **Built**: 1900 | **Condition Rating**: **1 (Imminent Failure)** | **Est. Failure Prob**: **99.93%**

### C. Category: Vehicle Overload
Bridges with structural designs susceptible to sudden shear failure when overloaded by heavy freight.
1.  **Facility**: `CO 43` over `SILAS CREEK` (Alabama)
    *   **Built**: 1955 | **Condition Rating**: 5 (Fair) | **Est. Collapse Prob**: **99.33%**
2.  **Facility**: `S-20-303` over `BR OF EAST FORK` (South Carolina)
    *   **Built**: 1970 | **Condition Rating**: 5 (Fair) | **Est. Collapse Prob**: **99.31%**
3.  **Facility**: `NFA A214` over `STEWART BRANCH` (Tennessee)
    *   **Built**: 1971 | **Condition Rating**: 6 (Fair) | **Est. Collapse Prob**: **99.23%**

---

## 3. High-Risk Candidate Shortlist Composition

Out of the 624,193 scored bridges, our logical filters isolated a targeted shortlist of **25,144 bridges (4.0% of the entire US inventory)**:
*   **18,237 bridges (72.5%)** are flagged for **vehicle load posting deficiencies** (open but structurally restricted from standard legal freight weight).
*   **9,522 bridges (37.9%)** are in **critical condition (lowest rating $\le 3$)**.
*   **5,434 bridges (21.6%)** are flagged for **severe hydraulic/scour risk** (streambed erosion under structural foundations).

---

## 4. Key Physical Drivers (Model SHAP Rankings)

The primary physical attributes that the models identified as driving the collapse and failure predictions:
*   **Scour Risk**: Driven by `scour_code` (foundational status), `reconstruction_age` (rehabilitation recency), and `adt` (average daily traffic).
*   **Overload Risk**: Driven by `load_deficient_flag` (structural operational capacity), `scour_code`, and `pct_truck_traffic` (heavy load frequency).
*   **Collision Risk**: Driven by `pct_truck_traffic` (commercial truck density), `bridge_age`, and `superstructure_cond` (physical health).
