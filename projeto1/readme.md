# Customer Segmentation and Targeted Marketing Strategies

## Project Overview
This project focuses on applying machine learning techniques to segment customers based on their demographics, spending behavior, and purchasing patterns. By leveraging clustering methods and association rules, the project delivers actionable insights for crafting personalized marketing campaigns. The analysis was conducted on two datasets: one containing customer demographic and behavioral data and another detailing transaction baskets.

## Objectives
1. Perform exploratory data analysis to clean, preprocess, and transform the datasets.
2. Apply clustering techniques to segment customers into meaningful groups based on shared characteristics.
3. Analyze customer behavior within each segment to uncover motivations, preferences, and needs.
4. Develop targeted promotions tailored to each segment using association rule mining.

## Problem Statement
In highly competitive markets, understanding customer behavior and tailoring strategies to meet their preferences is critical for success. The challenge was to:
- Identify customer segments with similar purchasing behaviors.
- Create targeted marketing strategies for each segment to boost customer engagement, loyalty, and profitability.

## Methodology
### Data
1. **Customer Information Dataset**:
   - Includes demographic details, spending habits, and loyalty information.
   - Key variables: lifetime spending across categories, percentage of purchases on promotion, customer age, and tenure.

2. **Customer Basket Dataset**:
   - Includes 100,000 transactions linking customer IDs to lists of purchased goods.

3. **Product Mapping File**:
   - Maps product names to categories for association rule mining.

### Exploratory Data Analysis and Preprocessing
1. Treated missing values using KNN imputation.
2. Transformed variables for better interpretability:
   - Birthdate → Age.
   - Gender → Binary variable.
   - Loyalty card → Binary variable.
   - Spending categories → Percentage of total spend.
3. Scaled data using RobustScaler to handle outliers without removing them.
4. Analyzed correlations and distributions to ensure data readiness for clustering.

### Clustering Techniques
1. **Hierarchical Clustering (Ward Linkage)**:
   - Used dendrograms to determine the optimal number of clusters.
   - Applied for initial segmentation to identify clear groups such as "Fishermen" and "Gamers."
   
2. **K-Means**:
   - Applied to refine segmentation for remaining customers.
   - Used the elbow method to determine the optimal number of clusters.

3. **Cluster Characteristics**:
   - Segments included "Pet Lovers," "Vegetarians," "Young Party People," "Parents," "Loyal Customers," "Promotions," and "Young with Electronics."

### Targeted Promotions
- Association rule mining was performed to identify product combinations frequently purchased together.
- Created promotional campaigns tailored to each cluster’s unique behavior:
  - Example: Fishermen were offered discounts on fish products, while Gamers received promotions on electronics and complementary items like champagne.

## Key Insights
1. **Fishermen**:
   - Spend primarily on fish, often in single-store visits.
   - Campaigns: Discounts on fish products and bundles like tuna and salmon.
   
2. **Gamers**:
   - High spending on electronics and video games.
   - Campaigns: Discounts on headphones with phone purchases and promotions on gaming accessories.

3. **Pet Lovers**:
   - Spend heavily on pet-related products.
   - Campaigns: Buy-two-get-one-free deals on pet toys and food.

4. **Promotions Lovers**:
   - Highly sensitive to discounts.
   - Campaigns: Basket-wide discounts on purchases exceeding $100.

5. **Young Party People**:
   - High spending on alcohol and party-related items.
   - Campaigns: Discounts on wine and beer bundles.

6. **Loyal Customers**:
   - Long tenure with diverse spending.
   - Campaigns: Incentives to explore new product categories.

7. **Parents**:
   - Balanced spending with emphasis on family essentials.
   - Campaigns: Discounts on baby products and family meal bundles.

## Conclusion
This project successfully segmented customers into distinct groups and crafted targeted promotions to enhance engagement and profitability. The use of clustering techniques and association rules provided actionable insights, allowing businesses to tailor their strategies effectively. By implementing these recommendations, businesses can unlock growth opportunities and foster stronger customer relationships.
