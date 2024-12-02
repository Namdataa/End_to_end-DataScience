# Analysis And Forecast Of Perfume Demand
![FRAMEWORK](https://github.com/user-attachments/assets/a694c5fc-accf-432b-bf9e-dcbe28298515)
With the project “**Analysis & Forecasting Market Demand and Optimizing the Perfume Supply Chain**”, our goal is to use the business’s own data to
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
#### (Data is extracted from API and created in `ETL_data.ipynb` file)
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
#### (The data is generated from Python libraries and is created in the `ETL_data.ipynb` file.)
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
# Forecast And Business Plan Proposal

## Temperature and product type:
### Northern region (Hanoi, Hai Phong):
- The lowest average temperature ranges from **19.93°C** to **20.64°C**, indicating that winter is colder than other regions.
![image](https://github.com/user-attachments/assets/0c9cfb48-fc0f-4b4b-954d-10a5f7675a5e)
![image](https://github.com/user-attachments/assets/dff4300c-3fb9-4e4e-b4d6-f790fdf4262d)

- Here, with the seasonal nature of temperature, we will look at the past temperature in **Ha Noi** and **Hai Phong** from 12/2023 to the end of 02/2024.

![image](https://github.com/user-attachments/assets/e00f07c2-3462-4cea-81fe-6e88e66cc150)
![image](https://github.com/user-attachments/assets/1e062c7f-3458-4437-a279-96ff3bea7aeb)

- From the forecast temperature and the past temperature of **Ha Noi** and **Hai Phong** in the period from 12/2023 to the end of 02/2024, the temperature is very low and it always fluctuates with the temperature from **14.9°C** đến **22.15°C**  and in this temperature range with the type of product suitable for this customer is similar to our forecast that **Eau De Parfum** and **Perfume** have a concentration of **Eau De Parfum (15-20%)** and **Perfume (20-30%)** from 15-30%.

![image](https://github.com/user-attachments/assets/a749f336-1195-4134-a880-1cf2efeb226e)

- In the Northern market at 2 locations, **Ha Noi** and **Hai Phong**, **Eau De Parfum** is more dominant than **Perfume** and people prefer to use Eau De **Parfum**.
### Central region (Da Nang, Hue):
- The lowest average temperature ranges from **22.64°C (Hue)** to **25.45°C (Nha Trang)**, warmer than the North.

![image](https://github.com/user-attachments/assets/49ffb7a3-474f-4d08-b706-4ecd7304ee75)
![image](https://github.com/user-attachments/assets/3dd026db-3858-45c8-8582-67921173dab3)

- Here, with the seasonal nature of temperature, we will look at the past temperature in **Hue** and **Da Nang** from December 2023 to the end of February 2024.

![image](https://github.com/user-attachments/assets/73fa2a76-fd6b-40aa-8f6d-f89911c3b8a4)
![image](https://github.com/user-attachments/assets/913ca52f-53d4-4b08-ace0-0f15fbbac987)

- From the forecast temperature and the past temperature of **Hue**, **Da Nang** in the period of 12/2023 to the end of 02/2024, the average temperature is warmer than the North and it always goes sideways with the temperature from **20°C** to **25°C** and in this temperature range with the type of product suitable for this customer is similar to our forecast of **Eau De Toilette** and **Eau De Parfum** with the concentration of **Eau De Toilette (5-15%)** and **Eau De Parfum (15-20%)** from 5-20%.

![image](https://github.com/user-attachments/assets/1a5d3134-3623-4e95-a66c-70500e15040c)

- In **Da Nang** market, **Eau De Toilette** is higher than **Eau De Parfum**, but in **Hue** market, there is another advantage, **Eau De Parfum** is much higher than **Eau De Toilette**.
 ### Southern and South Central region (Ho Chi Minh, Ca Mau, Nha Trang):
- The lowest average temperature ranges from **26.16°C** to **26.99°C**, higher than both the North and Central regions, reflecting the typical hot and humid tropical monsoon climate.

![image](https://github.com/user-attachments/assets/354cd4ec-8286-4ca3-bf5c-f2c22fc278ec)
![image](https://github.com/user-attachments/assets/17fabe5c-cd41-4fe2-9d56-f2216fb52e3c)
![image](https://github.com/user-attachments/assets/626981b2-6580-4478-8157-ca5692ae2822)

- Here, with the seasonal nature of temperature, we will look at the past temperature in **Ho Chi Minh**, **Ca Mau** and **Nha Trang** from December 2023 to the end of February 2024.

![image](https://github.com/user-attachments/assets/24882092-c998-4d3d-9540-f20dc8ecfd29)
![image](https://github.com/user-attachments/assets/0c1d16ed-49af-4e90-8fb6-082f1a79326b)
![image](https://github.com/user-attachments/assets/85e5768e-5b65-4d81-8771-a47025bf2bc0)

- From the forecast temperature and historical temperature of **Ho Chi Minh City**, **Ca Mau** and **Nha Trang** in the period from 12/2023 to the end of 02/2024, the temperature is high, slightly higher than that of the North and Central regions and it always goes sideways with the temperature from **27°C** to **29°C** and is in this temperature range with **Eau De Cologne** and **Eau De Toilette** with concentration of **Eau De Cologne (2-4%)** and **Eau De Toilette (5-15%)**.

![image](https://github.com/user-attachments/assets/51a86a5b-d0a2-49a2-99bf-866d11954630)

- In the Southern and South Central markets, people use two types of perfumes: **Eau De Cologne** and **Eau De Toilette**, but consumers prefer **Eau De Toilette**.
## Manufacturer:
- With big perfume manufacturers like **Dior** and **Chanel** will be the top priority because it has a large traffic compared to other perfume brands.

![image](https://github.com/user-attachments/assets/0cd7ab80-7ce9-4ebf-98cc-24a44467294e)
![image](https://github.com/user-attachments/assets/924d11cf-30b9-4b7f-99d3-ff8e9820aa01)
![image](https://github.com/user-attachments/assets/be2d3ac1-f6b1-4c82-ab6a-a326be9c0f91)
![image](https://github.com/user-attachments/assets/094b6b65-81c0-431e-aa33-33f836066f01)

- With the search volume for products of two big brands like Dior and Chanel in the past being very high, it can be said that it is double that of other brands. With the forecast for the import and distribution plan for the next 90 days, the forecast for the search volume of Dior and Chanel perfumes tends to increase well, with Gucci and Bharara also being two famous brands in the market but the forecast for the next 90 days is that the search volume will gradually decrease.
## Business Plan Recommendations for next 90 days:
### Manufacturer:
- **Focus on two major perfume brands:** Dior and Chanel.
### Product Type:
 **Northern Region (Hanoi, Hai Phong):**
- Advertise and reach customers with **Eau De Parfum (15-20%)** concentration and **Perfume (20-30%)** concentration products.
- Prioritize importing and marketing **Eau De Parfum** as the main product.

 **Central region (Da Nang, Hue):**
- Advertise and reach customers with **Eau De Toilette (5-15%)** concentration and **Eau De Parfum (15-20%)** concentration products.
- Prioritize importing and marketing **Eau De Parfum** in **Hue** as the main product.

 **Southern and South Central region (Ho Chi Minh, Ca Mau, Nha Trang):**
- Advertise and reach customers with **Eau De Cologne (2-4%)** concentration and **Eau De Toilette (5-15%)** concentration products.
- Prioritize importing and marketing **Eau De Toilette** in **Ho Chi Minh** as the main product.
## Project Flow Chart

## Project Steps
### 1. Extract - Transform - Load (ETL) - ETL Pipeline
The pipeline helps update and store data contained in the `ETL_data.ipynb` file. The data in this system is run automatically by the APScheduler library, which synthesizes product data files, data from libraries and APIs, then transforms and extracts it and stores it in the local Database.

**- Data Extraction:**
+ Synthesize product data from the `data_sanpham.csv` file.
+ Combine MeteoStat and Faker libraries to create simulated customer data, orders, detailed orders and nominatim API to get data on the coordinates of cities.

**- Data Cleaning:**
+ Use the pandas library to clean the data in the `data_sanpham.csv` file such as removing duplicates, handling null values ​​and handling outliers

**- Feature Engineering and Transformation:**
+ Here I use 2 models, ARIMA and LSTM, extracting values ​​and using models in the code in the `Metric.py` file
+ Training the LSTM model to predict the weather in the file
#### 1.1 Usage
Here are the steps to setup and run the pipeline:
1. Need a code running environment like VS Code or Jupyter Notebook, ... and download the above files to the same place as the code running environment folder.
2. Open the `ETL_data.ipynb` file and need to install the necessary libraries first, open your SQL server and fill in your server information in the section that needs to be replaced in the `ETL_data.ipynb` file and run it, because it is scheduled to run every 5 minutes so you need to wait.
### 2. Data Analysis & Predictive Modelling
- We present actionable results overview charts for senior executives on the Dashboard side, and on the Analytics side we include more in-depth data on customers, products, and orders.

![image](https://github.com/user-attachments/assets/751b7d9b-cfc4-4019-8b58-33e32f1d8bc0)
- With the forecasting model we are using to predict what the weather will be like in the next 90 days and what the search traffic for keywords related to the product will be in the coming months. This data combined with the company's historical data will help senior management make better decisions.

![image](https://github.com/user-attachments/assets/00562da4-6484-4c9c-811e-f656163b43d5)
