from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd
import os

def Database(Companies):
    # Set up WebDriver (switch to Chrome for better support)
    driver = webdriver.Safari()  # Use Chrome WebDriver (make sure to install chromedriver)
    driver.maximize_window()

    # Check if the output folder exists
    if not os.path.exists('Database'):
        os.makedirs('Database')

    for company in Companies:
        driver.get("https://finance.yahoo.com")

        # Wait for the search box to appear
        search_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "yfin-usr-qry"))
        )
        search_box.clear()  # Clear the search box before entering new text
        search_box.send_keys(company)

        # Wait for and click the search button
        search_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "search-button"))
        )
        search_button.click()

        # Wait for the page to load and go to the 'Historical Data' tab
        historical_data_tab = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//li[contains(@data-test, "HISTORICAL_DATA")]'))
        )
        historical_data_tab.click()

        time.sleep(5)  # Wait for the page to fully load

        # Open the date range menu
        time_range_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@data-test="date-picker-full-range"]'))
        )
        time_range_button.click()

        # Select the 'Max' option
        max_range_option = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//button[@data-value="MAX"]'))
        )
        max_range_option.click()

        # Apply the date range
        apply_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//button[@data-test="date-picker-confirm"]'))
        )
        apply_button.click()

        time.sleep(5)  # Wait for the data to reload

        # Scroll to load more data
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  # Wait for new content to load
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        # Parse the page source and extract stock prices
        data = []
        webpage = driver.page_source
        soup = BeautifulSoup(webpage, 'lxml')
        stock_prices = soup.find_all('tr', class_='BdT Bdc($seperatorColor) Ta(end) Fz(s) Whs(nw)')

        for stock_price in stock_prices:
            stock_price = stock_price.find_all('td')
            price = [stock.text for stock in stock_price]
            data.append(price)

        # Save data to CSV
        if data:  # Check if data is not empty
            database = pd.DataFrame(data, columns=["Date", "Open", "High", "Low", "Close", "Adj. Close", "Volume"])
            database.to_csv(f'Database/{company}.csv', index=False)
        else:
            print(f"No data found for {company}")

    # Quit the browser once all companies are processed
    driver.quit()

if __name__ == "__main__":
    List_of_Companies = ['GOOG', 'AAPL', 'MSFT', 'TSLA', 'AMZN']
    Database(List_of_Companies)
