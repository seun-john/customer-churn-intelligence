# Customer behaviour analysis

All figures describe the 7,043 fictional records. Categorical rates divide churn labels by the displayed category count. Numeric summaries retain observed values; eleven missing TotalCharges cells are excluded only from summaries of that field. No customer rows are deleted. Intervals are 95% Wilson binomial intervals calculated separately for each category; multiple comparisons are exploratory and unadjusted.

## Interpretation by business theme

Contract and tenure comparisons can inform an investigation of onboarding and contract-related experiences. Contract choice may reflect customer preferences or prior experience; tenure is affected by survivorship. Neither supports forcing a contract change or claiming a measured retention lever.

Internet type, payment method and paperless billing differences can help define interview or service-review strata. Service mix, pricing, contract and customer preferences are entangled. An unadjusted payment-method association is not evidence that changing the payment method reduces churn.

Security, backup, device protection, technical support and streaming subscriptions should be read with internet availability. The 'No internet service' group is structurally different from internet customers declining an add-on. Phone service and multiple lines have an analogous dependency. Separate category tables and joint service combinations make these denominators visible; they do not isolate an add-on effect.

Senior status, gender, partner and dependent status describe the sample. Group differences may reflect household composition, service mix and unmeasured circumstances. They call for a fairness and service-access audit before operational targeting, not demographic eligibility rules. No demographic group is certified as fairly served by this benchmark.

## Categorical evidence

### gender

| gender | customers | churned | churn_rate | rate_lower_95 | rate_upper_95 |
| --- | --- | --- | --- | --- | --- |
| Female | 3,488 | 939 | 0.2692 | 0.2547 | 0.2842 |
| Male | 3,555 | 930 | 0.2616 | 0.2474 | 0.2763 |

The highest observed rate is Female (26.92%, n=3,488); the lowest is Male (26.16%, n=3,555). The unadjusted gap is 0.76 percentage points. Counts and Wilson 95% intervals describe sample precision, not population representativeness or a causal effect.

### SeniorCitizen

| SeniorCitizen | customers | churned | churn_rate | rate_lower_95 | rate_upper_95 |
| --- | --- | --- | --- | --- | --- |
| 0 | 5,901 | 1,393 | 0.2361 | 0.2254 | 0.2471 |
| 1 | 1,142 | 476 | 0.4168 | 0.3885 | 0.4456 |

The highest observed rate is 1 (41.68%, n=1,142); the lowest is 0 (23.61%, n=5,901). The unadjusted gap is 18.08 percentage points. Counts and Wilson 95% intervals describe sample precision, not population representativeness or a causal effect.

### Partner

| Partner | customers | churned | churn_rate | rate_lower_95 | rate_upper_95 |
| --- | --- | --- | --- | --- | --- |
| No | 3,641 | 1,200 | 0.3296 | 0.3145 | 0.3450 |
| Yes | 3,402 | 669 | 0.1966 | 0.1836 | 0.2103 |

The highest observed rate is No (32.96%, n=3,641); the lowest is Yes (19.66%, n=3,402). The unadjusted gap is 13.29 percentage points. Counts and Wilson 95% intervals describe sample precision, not population representativeness or a causal effect.

### Dependents

| Dependents | customers | churned | churn_rate | rate_lower_95 | rate_upper_95 |
| --- | --- | --- | --- | --- | --- |
| No | 4,933 | 1,543 | 0.3128 | 0.3000 | 0.3259 |
| Yes | 2,110 | 326 | 0.1545 | 0.1397 | 0.1706 |

The highest observed rate is No (31.28%, n=4,933); the lowest is Yes (15.45%, n=2,110). The unadjusted gap is 15.83 percentage points. Counts and Wilson 95% intervals describe sample precision, not population representativeness or a causal effect.

### PhoneService

| PhoneService | customers | churned | churn_rate | rate_lower_95 | rate_upper_95 |
| --- | --- | --- | --- | --- | --- |
| No | 682 | 170 | 0.2493 | 0.2183 | 0.2831 |
| Yes | 6,361 | 1,699 | 0.2671 | 0.2564 | 0.2781 |

The highest observed rate is Yes (26.71%, n=6,361); the lowest is No (24.93%, n=682). The unadjusted gap is 1.78 percentage points. Counts and Wilson 95% intervals describe sample precision, not population representativeness or a causal effect.

### MultipleLines

| MultipleLines | customers | churned | churn_rate | rate_lower_95 | rate_upper_95 |
| --- | --- | --- | --- | --- | --- |
| No | 3,390 | 849 | 0.2504 | 0.2361 | 0.2653 |
| No phone service | 682 | 170 | 0.2493 | 0.2183 | 0.2831 |
| Yes | 2,971 | 850 | 0.2861 | 0.2701 | 0.3026 |

The highest observed rate is Yes (28.61%, n=2,971); the lowest is No phone service (24.93%, n=682). The unadjusted gap is 3.68 percentage points. Counts and Wilson 95% intervals describe sample precision, not population representativeness or a causal effect.

### InternetService

| InternetService | customers | churned | churn_rate | rate_lower_95 | rate_upper_95 |
| --- | --- | --- | --- | --- | --- |
| DSL | 2,421 | 459 | 0.1896 | 0.1745 | 0.2057 |
| Fiber optic | 3,096 | 1,297 | 0.4189 | 0.4017 | 0.4364 |
| No | 1,526 | 113 | 0.0740 | 0.0620 | 0.0883 |

The highest observed rate is Fiber optic (41.89%, n=3,096); the lowest is No (7.40%, n=1,526). The unadjusted gap is 34.49 percentage points. Counts and Wilson 95% intervals describe sample precision, not population representativeness or a causal effect.

### OnlineSecurity

| OnlineSecurity | customers | churned | churn_rate | rate_lower_95 | rate_upper_95 |
| --- | --- | --- | --- | --- | --- |
| No | 3,498 | 1,461 | 0.4177 | 0.4014 | 0.4341 |
| No internet service | 1,526 | 113 | 0.0740 | 0.0620 | 0.0883 |
| Yes | 2,019 | 295 | 0.1461 | 0.1314 | 0.1622 |

The highest observed rate is No (41.77%, n=3,498); the lowest is No internet service (7.40%, n=1,526). The unadjusted gap is 34.36 percentage points. Counts and Wilson 95% intervals describe sample precision, not population representativeness or a causal effect.

### OnlineBackup

| OnlineBackup | customers | churned | churn_rate | rate_lower_95 | rate_upper_95 |
| --- | --- | --- | --- | --- | --- |
| No | 3,088 | 1,233 | 0.3993 | 0.3821 | 0.4167 |
| No internet service | 1,526 | 113 | 0.0740 | 0.0620 | 0.0883 |
| Yes | 2,429 | 523 | 0.2153 | 0.1994 | 0.2321 |

The highest observed rate is No (39.93%, n=3,088); the lowest is No internet service (7.40%, n=1,526). The unadjusted gap is 32.52 percentage points. Counts and Wilson 95% intervals describe sample precision, not population representativeness or a causal effect.

### DeviceProtection

| DeviceProtection | customers | churned | churn_rate | rate_lower_95 | rate_upper_95 |
| --- | --- | --- | --- | --- | --- |
| No | 3,095 | 1,211 | 0.3913 | 0.3742 | 0.4086 |
| No internet service | 1,526 | 113 | 0.0740 | 0.0620 | 0.0883 |
| Yes | 2,422 | 545 | 0.2250 | 0.2088 | 0.2421 |

The highest observed rate is No (39.13%, n=3,095); the lowest is No internet service (7.40%, n=1,526). The unadjusted gap is 31.72 percentage points. Counts and Wilson 95% intervals describe sample precision, not population representativeness or a causal effect.

### TechSupport

| TechSupport | customers | churned | churn_rate | rate_lower_95 | rate_upper_95 |
| --- | --- | --- | --- | --- | --- |
| No | 3,473 | 1,446 | 0.4164 | 0.4001 | 0.4328 |
| No internet service | 1,526 | 113 | 0.0740 | 0.0620 | 0.0883 |
| Yes | 2,044 | 310 | 0.1517 | 0.1368 | 0.1679 |

The highest observed rate is No (41.64%, n=3,473); the lowest is No internet service (7.40%, n=1,526). The unadjusted gap is 34.23 percentage points. Counts and Wilson 95% intervals describe sample precision, not population representativeness or a causal effect.

### StreamingTV

| StreamingTV | customers | churned | churn_rate | rate_lower_95 | rate_upper_95 |
| --- | --- | --- | --- | --- | --- |
| No | 2,810 | 942 | 0.3352 | 0.3180 | 0.3529 |
| No internet service | 1,526 | 113 | 0.0740 | 0.0620 | 0.0883 |
| Yes | 2,707 | 814 | 0.3007 | 0.2837 | 0.3182 |

The highest observed rate is No (33.52%, n=2,810); the lowest is No internet service (7.40%, n=1,526). The unadjusted gap is 26.12 percentage points. Counts and Wilson 95% intervals describe sample precision, not population representativeness or a causal effect.

### StreamingMovies

| StreamingMovies | customers | churned | churn_rate | rate_lower_95 | rate_upper_95 |
| --- | --- | --- | --- | --- | --- |
| No | 2,785 | 938 | 0.3368 | 0.3195 | 0.3546 |
| No internet service | 1,526 | 113 | 0.0740 | 0.0620 | 0.0883 |
| Yes | 2,732 | 818 | 0.2994 | 0.2825 | 0.3169 |

The highest observed rate is No (33.68%, n=2,785); the lowest is No internet service (7.40%, n=1,526). The unadjusted gap is 26.28 percentage points. Counts and Wilson 95% intervals describe sample precision, not population representativeness or a causal effect.

### Contract

| Contract | customers | churned | churn_rate | rate_lower_95 | rate_upper_95 |
| --- | --- | --- | --- | --- | --- |
| Month-to-month | 3,875 | 1,655 | 0.4271 | 0.4116 | 0.4427 |
| One year | 1,473 | 166 | 0.1127 | 0.0975 | 0.1299 |
| Two year | 1,695 | 48 | 0.0283 | 0.0214 | 0.0373 |

The highest observed rate is Month-to-month (42.71%, n=3,875); the lowest is Two year (2.83%, n=1,695). The unadjusted gap is 39.88 percentage points. Counts and Wilson 95% intervals describe sample precision, not population representativeness or a causal effect.

### PaperlessBilling

| PaperlessBilling | customers | churned | churn_rate | rate_lower_95 | rate_upper_95 |
| --- | --- | --- | --- | --- | --- |
| No | 2,872 | 469 | 0.1633 | 0.1502 | 0.1773 |
| Yes | 4,171 | 1,400 | 0.3357 | 0.3215 | 0.3501 |

The highest observed rate is Yes (33.57%, n=4,171); the lowest is No (16.33%, n=2,872). The unadjusted gap is 17.24 percentage points. Counts and Wilson 95% intervals describe sample precision, not population representativeness or a causal effect.

### PaymentMethod

| PaymentMethod | customers | churned | churn_rate | rate_lower_95 | rate_upper_95 |
| --- | --- | --- | --- | --- | --- |
| Bank transfer (automatic) | 1,544 | 258 | 0.1671 | 0.1493 | 0.1865 |
| Credit card (automatic) | 1,522 | 232 | 0.1524 | 0.1352 | 0.1714 |
| Electronic check | 2,365 | 1,071 | 0.4529 | 0.4329 | 0.4730 |
| Mailed check | 1,612 | 308 | 0.1911 | 0.1726 | 0.2110 |

The highest observed rate is Electronic check (45.29%, n=2,365); the lowest is Credit card (automatic) (15.24%, n=1,522). The unadjusted gap is 30.04 percentage points. Counts and Wilson 95% intervals describe sample precision, not population representativeness or a causal effect.

### TenureGroup

| TenureGroup | customers | churned | churn_rate | rate_lower_95 | rate_upper_95 |
| --- | --- | --- | --- | --- | --- |
| 0–12 months | 2,186 | 1,037 | 0.4744 | 0.4535 | 0.4953 |
| 13–24 months | 1,024 | 294 | 0.2871 | 0.2602 | 0.3156 |
| 25–48 months | 1,594 | 325 | 0.2039 | 0.1848 | 0.2244 |
| 49–72 months | 2,239 | 213 | 0.0951 | 0.0837 | 0.1080 |

The highest observed rate is 0–12 months (47.44%, n=2,186); the lowest is 49–72 months (9.51%, n=2,239). The unadjusted gap is 37.93 percentage points. Counts and Wilson 95% intervals describe sample precision, not population representativeness or a causal effect.

## Numerical distributions and label differences

| Churn | tenure_count | tenure_mean | tenure_median | tenure_std | MonthlyCharges_count | MonthlyCharges_mean | MonthlyCharges_median | MonthlyCharges_std | TotalCharges_count | TotalCharges_mean | TotalCharges_median | TotalCharges_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| No | 5,174 | 37.5700 | 38.0000 | 24.1138 | 5,174 | 61.2651 | 64.4250 | 31.0926 | 5,163 | 2555.3441 | 1683.6000 | 2329.4570 |
| Yes | 1,869 | 17.9791 | 10.0000 | 19.5311 | 1,869 | 74.4413 | 79.6500 | 24.6661 | 1,869 | 1531.7961 | 703.5500 | 1890.8230 |

Counts, means, medians and standard deviations show both central tendency and spread. Means can be influenced by outliers; medians provide a complementary comparison. Monthly and accumulated charges are billing quantities, not measured profitability. Histograms in figures/numeric_distributions.png show overlapping distributions rather than separable customer types.

### Pearson correlations

| variable | tenure | MonthlyCharges | TotalCharges |
| --- | --- | --- | --- |
| tenure | 1.0000 | 0.2479 | 0.8259 |
| MonthlyCharges | 0.2479 | 1.0000 | 0.6511 |
| TotalCharges | 0.8259 | 0.6511 | 1.0000 |

Accumulated charges, monthly charges and tenure have structural and nonlinear relationships. Correlation describes linear association and does not establish independence, availability before the target, causality or an intervention effect.

## Service combinations

Highest observed churn combinations among groups with at least 50 customers are shown below. This descriptive count cutoff prevents highlighting tiny cells; it is not a statistical significance rule. All cells, including smaller groups, remain available in service_combinations.csv.

| InternetService | TechSupport | OnlineSecurity | customers | churned | churn_rate |
| --- | --- | --- | --- | --- | --- |
| Fiber optic | No | No | 1,765 | 971 | 0.5501 |
| DSL | No | No | 788 | 279 | 0.3541 |
| Fiber optic | Yes | No | 492 | 143 | 0.2907 |
| Fiber optic | No | Yes | 465 | 130 | 0.2796 |
| DSL | Yes | No | 453 | 68 | 0.1501 |
| DSL | No | Yes | 455 | 66 | 0.1451 |
| Fiber optic | Yes | Yes | 374 | 53 | 0.1417 |
| No | No internet service | No internet service | 1,526 | 113 | 0.0740 |

Joint patterns may help design qualitative follow-up. Contracts, tenure and other services remain uncontrolled, and the fictional sampling process is unknown. These comparisons do not establish that adding support or security prevents churn.

## Sampling and data-quality concerns

The source has no verified sampling frame, collection dates, population weights or temporal outcome horizon. Exact record and customer-ID duplicates were absent; near-duplicate profiles can occur legitimately in a small categorical schema. Eleven zero-tenure customers have blank accumulated charges. Retaining them avoids silent sample selection, while fold-local imputation prevents evaluation leakage. The 26.54% churn-label share is a sample property, not a population forecast. No causal, temporal or production-validity claim follows from these analyses.
