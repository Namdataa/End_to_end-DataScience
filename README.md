![image](https://github.com/user-attachments/assets/6dc51844-5cfe-481a-80c6-2a1db77b67b3)# Analysis And Forecast Of Perfume Demand
![FRAMEWORK](https://github.com/user-attachments/assets/a694c5fc-accf-432b-bf9e-dcbe28298515)
With the project “Analysis & Forecasting Market Demand and Optimizing the Perfume Supply Chain”, our goal is to use the business’s own data to
predict product demand, help optimize the import process, minimize unnecessary inventory
and at the same time increase efficiency in supply chain management.
This report not only provides specific analysis of market demand, but also
proposes solutions for applying data science to create long-term competitive advantages for businesses in the perfume business.
# (Phase 1): Extract - Transform - Load (ETL) (Project focus)
**The Analysis And Forecast Of Perfume Demand Project:** requires data on products, customers and purchase information. For the project we aggregate data from different sources from python's faker library, API and kaggle.
![image](https://github.com/user-attachments/assets/b0d0e145-33bd-4d8c-a865-a4e3d3cae0e6)
# (Phase 2): Overview and Forecasting 
**Visualization and Forecasting for Project:** Here we use a web framework Streamlit to help us present the results of the report and forecast. In Streamlit, users can interact with the chart at will and it can work online near realtime.
![image](https://github.com/user-attachments/assets/1b1338ec-ea74-4192-a569-ca26519daaf0)
## Team members
* **Team Leader/Analytics Engineer/BI Engineer:** Oversee the entire project, planning and support on both technical and non-technical aspects.
  * [Hoang Nam](https://www.facebook.com/namnew2003/): Data Science & Business Analytics at DUE
* **Other members:** Conduct data discovery and documentation, uncover business insights and provide client-driven recommendations.
  * [Bich Hoa](https://www.facebook.com/bich.hoa.050303): Data Science & Business Analytics at DUE
  * [My Linh](https://www.facebook.com/mylinhh1853): Data Science & Business Analytics at DUE
## About the data
### Data Source
#### (Data is taken from Kaggle and API from Ebay e-commerce platform)
* **Product Data:**
  * `Product_id`: Identifier of each product.
  * `Brand`: The brand of the perfume.
  * `Title`: The title of the listing.
  * `Type`:  The type of perfume (e.g., Eau de Parfum, Eau de Toilette).
  * `Price`: The price of the perfume.
  * `PriceWithCurrency`: The price with currency notation.
  * `Available`: The number of items available.
  * `AvailableText`: Text description of availability.
  * `Sold`: The number of items sold.
  * `LastUpdated`: The last updated timestamp of the listing.
  * `ItemLocation`: The location of the item.
#### (Data is extracted from API and created in ETL_data.ipynb file)
* **Weather_location Data:**
  * `time`: Time of weather data recording (by day, month, or year). Used to identify a specific time for weather data.
  * `tavg`: Average temperature (°C), showing the average temperature during the recorded time period.
  * `tmin`: Minimum temperature (°C), recording the minimum temperature during the day or observation period.
  * `tmax`: Maximum temperature (°C), recording the maximum temperature during the day or observation period.
  * `prcp`: Precipitation (mm), measuring the amount of rain recorded during the time period.
  * `snow`: Snowfall (mm), measuring the amount of snowfall during the time period (if any).
  * `wdir`: Wind direction (degrees), showing the wind direction in degrees from 0° (North) to 360° (South).
  * `wspd`: Average wind speed (m/s), measuring the average wind speed during the time period.
  * `wpgt`: Maximum wind speed (m/s), records the strongest wind speed during the observation period.
  * `pres`: Average air pressure (hPa), measures the atmospheric pressure at the recorded time.
  * `tsun`: Sunshine duration (minutes), total time the sun appears during the day.
* **Coordinates_location Data:**
  * `city`: City name. This is information used to identify a specific location in the data.
  * `lat`: The latitude of the city, showing the geographic location in the North-South direction on Earth.
  * `lon`: The longitude of the city, showing the geographic location in the East-West direction on Earth.
#### (The data is generated from Python libraries and is created in the ETL_data.ipynb file.)
* **Customer Data:**
  * `Customer_id`: A unique identifier for each customer. Used to distinguish customers in the database.
  * `Name`: The customer's full name.
  * `Address`: The customer's address, which helps identify the customer's place of residence or geographic location.
  * `Job`: The customer's occupation, which provides information about the field of work or professional role.
  * `Sex`: The customer's gender (e.g., Male, Female).
  * `Age`: The customer's age, often used for demographic analysis.
  * `Phone`: The customer's contact phone number, used for communication or support purposes.
  * `Email`: The customer's email address, often used for notifications, communications, or email marketing.
* **Order Data:**
  * `Order_time`: Order time, showing the date and time the order was created. This information is important for tracking and analyzing order trends over time.
  * `Order_id`: A unique identifier for each order. Used to distinguish and retrieve detailed information about each order.
  * `Customer_id`: The identifier of the customer associated with the order. This is a foreign key, connecting data to the customer table.
  * `Total`: The total value of the order, calculated based on the price and quantity of each product in the order. 
* **Order_details Data:**
  * `Orderdetail_id`: Unique identifier for each detail item in the order. Used to distinguish items in the order details table.
  * `Order_id`: The identifier of the order linked to the order details. This is a foreign key, connecting data to the order table.
  * `Sanpham_id`: The identifier of the product, linked to the product table. Used to identify the type of product in the order details.
  * `Quanlity`: The quantity of the product in the order.
  * `Price`: The price of the product (per unit), recorded to calculate the total order value.
### Cleaned Data
  * `order_time`: Order time, showing the date and time the order was created. This information is important for tracking and analyzing order trends over time.
  * `orderdetail_id`: Unique identifier for each detail item in the order. Used to distinguish items in the order details table.
  * `order_id`: A unique identifier for each order. Used to distinguish and retrieve detailed information about each order.
  * `customer_id`: A unique identifier for each customer. Used to distinguish customers in the database.
  * `product_id`: Identifier of each product.
  * `brand`: The brand of the perfume.
  * `type`:  The type of perfume (e.g., Eau de Parfum, Eau de Toilette).
  * `price`: The price of the perfume.
  * `quanlity`: The quantity of the product in the order.
  * `itemLocation`: The location of the item.
  * `weather`: Weather that the product is suitable for.
  * `address`: The customer's address, which helps identify the customer's place of residence or geographic location.
  * `sex`: The customer's gender (e.g., Male, Female).
  * `age`: The customer's age, often used for demographic analysis.
  * `lat`: The latitude of the location, which supports spatial analysis or visualization on a map.
  * `lon`: The longitude of the location, combined with lat to determine the exact geographic location.
  * `Doanh thu`: The total revenue of the order, calculated as `price` × `quantity` for each product in the order.



  
