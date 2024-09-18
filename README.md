# Stock Price Prediction
<br>
<img src='images/Stock-Price-Prediction.jpg' width = '100%' height='350px'>

## Project Overview

Investment firms, hedge funds and even individuals have been using financial models to understand market behaviour better and make profitable investments and trades. A wealth of information is available in the form of historical stock prices and company performance data, suitable for machine learning algorithms to process.

Can we predict stock prices with machine learning? Investors make educated guesses by analyzing data. They will read the news, study the company history, industry trends, and other data points that go into making a prediction. The prevailing theories are that stock prices are totally random and unpredictable, raising the question of why top firms like Morgan Stanley and Citigroup hire quantitative analysts to build predictive models. We have this idea of the trading floor being filled with adrenaline infuse men with loose ties running around yelling something into a phone. However, these days we are more likely to see rows of machine learning experts quietly sitting in front of computer screens. About 70% of all orders on Wall Street are now placed by software. We are now living in the age of algorithms.

This project utilizes the ARIMA model for base predictions and then built a Deep Learning model to improve it further. Stock prices are predicted for Tech Giants like Apple, Google, Tesla, Microsoft and Amazon.


## Dataset
Webscraped [ https://in.finance.yahoo.com](https://finance.yahoo.com/) using selenium and BeautifulSoup.

## Exploratory Data Analysis

### Closing Price v/s Time
<br>
<img src='images/Closing_Price_and_Time.png'>

We can see from the above graph that Telsa shares have tremendous growth in the 2020-2021 period.
<br>
If we follow the news, it can be due to

1. Emission Credit Sales
2. Tesla entering the Fast-Growing Compact SUV Market
3. Starting production in China

For the rest of the Companies, we can see that COVID-19 is the primary factor affecting the 2020-2021 period.
<br>

### Histogram plot of Percentage Daily Return 
<br>
<img src='images/Daily_Returns.png'>
