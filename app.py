# Import required libraries
import streamlit as st
import pandas as pd
from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from time import sleep
import re
import time
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import json
import requests
from io import BytesIO
from urllib3.exceptions import MaxRetryError, NameResolutionError
import logging
from selenium.common.exceptions import TimeoutException, WebDriverException

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def get_chrome_options():
    options = ChromeOptions()

    # Basic settings
    options.add_argument("--start-maximized")
    options.page_load_strategy = 'eager'

    # Headless mode configuration
    options.add_argument('--headless=new')

    # Graphics and rendering settings
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-software-rasterizer')
    options.add_argument('--disable-dev-shm-usage')

    # Security and sandbox settings
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-setuid-sandbox')

    # Memory and process settings
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-logging')
    options.add_argument('--disable-in-process-stack-traces')
    options.add_argument('--log-level=3')

    # Automation settings
    options.add_experimental_option("useAutomationExtension", False)
    options.add_experimental_option("excludeSwitches", [
        "enable-automation",
        "enable-logging"
    ])

    # Browser preferences
    options.add_experimental_option(
        "prefs",
        {
            "credentials_enable_service": False,
            "profile.password_manager_enabled": False,
            "profile.default_content_setting_values.notifications": 2
        }
    )

    return options


def create_driver_with_retry(max_retries=3, retry_delay=1):
    for attempt in range(max_retries):
        try:
            driver = webdriver.Chrome(options=get_chrome_options())
            return driver
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(retry_delay)
            continue
    return None


def is_website_accessible(url, timeout=5):
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        return 200 <= response.status_code < 400
    except Exception:
        return False


def safe_get_text(element, default=""):
    """Safely get text from element with error handling"""
    try:
        return element.text if element else default
    except Exception:
        return default


def clean_url(url):
    """Clean and validate URL"""
    if not url:
        return ""
    url = url.split('?')[0]  # Remove query parameters
    return url if url.startswith('http') else f"https://{url}"


def scrape_data(inp, limit=50):
    base_url = 'https://www.google.com/maps/search/'
    question = '+'.join(inp.strip().split())
    question = base_url + question
    all_links = set()
    driver = None

    try:
        driver = create_driver_with_retry()
        if not driver:
            st.error("Failed to initialize Chrome driver")
            return []

        driver.get(question)
        wait = WebDriverWait(driver, 10)

        scroll_attempts = 0
        max_scroll_attempts = 20
        last_links_count = 0

        while len(all_links) < limit and scroll_attempts < max_scroll_attempts:
            try:
                # Wait for feed to load
                feed = wait.until(EC.presence_of_element_located(
                    (By.CSS_SELECTOR, 'div[role="feed"]')
                ))

                # Get all visible links
                link_elements = driver.find_elements(By.CLASS_NAME, "hfpxzc")
                for element in link_elements:
                    href = element.get_attribute("href")
                    if href and href not in all_links:
                        all_links.add(href)
                        if len(all_links) >= limit:
                            break

                # Check if we're still getting new links
                if len(all_links) == last_links_count:
                    scroll_attempts += 1
                else:
                    scroll_attempts = 0
                last_links_count = len(all_links)

                # Scroll
                driver.execute_script(
                    "arguments[0].scrollBy(0, 500);",
                    feed
                )
                sleep(0.5)

            except TimeoutException:
                scroll_attempts += 1
                continue
            except Exception as e:
                logger.error(f"Error while scrolling: {str(e)}")
                break

    except Exception as e:
        st.error(f"Error in scraping: {str(e)}")
        return []
    finally:
        if driver:
            driver.quit()

    return list(all_links)[:limit]


def get_llm_analysis(content, prompt_template):
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("GROQ API key not found in environment variables")
            return None

        model = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="mixtral-8x7b-32768",
            temperature=0
        )

        # Extract information prompt
        prompt = PromptTemplate(
            input_variables=["document"],
            template="""Analyze this automotive business content and extract the following information:

Document: {document}

Please extract the following details in a structured format:
1. Services: List all automotive services, repairs, and maintenance offerings
2. Infrastructure: Available facilities, equipment, and tools
3. Contact Details: Phone numbers, addresses, and other contact information
4. Brands: Car brands serviced, partnerships, and certifications

Format the response as a JSON object:
{{
    "services": "<extracted services>",
    "infrastructure": "<extracted infrastructure>",
    "contact": "<extracted contact details>",
    "brands": "<extracted brand information>"
}}"""
        )

        chain = prompt | model | JsonOutputParser()
        return chain.invoke({"document": content})

    except Exception as e:
        logger.error(f"Error in LLM analysis: {str(e)}")
        return None


def process_location_data(driver, wait):
    data = {
        'name': '',
        'rating': '',
        'service_type': '',
        'address': '',
        'contact': '',
        'website': ''
    }

    try:
        # Get basic info with explicit waits
        data['name'] = safe_get_text(
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "DUwDvf")))
        )

        rating_elem = wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "F7nice"))
        )
        data['rating'] = safe_get_text(rating_elem).split()[0] if rating_elem else ""

        data['service_type'] = safe_get_text(
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "DkEaL")))
        )

        # Get address and contact
        address_elements = driver.find_elements(By.CLASS_NAME, 'Io6YTe')
        for elem in address_elements:
            text = safe_get_text(elem)
            if ',' in text:
                data['address'] = text
            elif text.startswith('+91'):
                data['contact'] = text

        # Get website
        website_elems = driver.find_elements(By.CLASS_NAME, "CsEnBe")
        for elem in website_elems:
            href = elem.get_attribute("href")
            if href and ('.com' in href or '.in' in href):
                data['website'] = clean_url(href)
                break

    except Exception as e:
        logger.error(f"Error extracting location data: {str(e)}")

    return data


def process_website_content(url):
    if not is_website_accessible(url):
        logger.warning(f"Website not accessible: {url}")
        return None

    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        content = re.sub(r'\s+', ' ', docs[0].page_content).strip()

        analysis = get_llm_analysis(content, "")

        if analysis:
            return analysis
        return None

    except Exception as e:
        logger.error(f"Error processing website {url}: {str(e)}")
        return None


def process_each_url(all_links):
    data = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, link in enumerate(all_links):
        try:
            status_text.text(f"Processing {idx + 1} of {len(all_links)} locations...")

            driver = create_driver_with_retry()
            if not driver:
                continue

            driver.get(link)
            wait = WebDriverWait(driver, 10)

            # Get basic location data
            location_data = process_location_data(driver, wait)
            driver.quit()

            # Process website content if available
            website_data = None
            if location_data.get('website'):
                website_data = process_website_content(location_data['website'])

            # Combine the data
            entry = {
                'Name': location_data.get('name', ''),
                'Rating': location_data.get('rating', ''),
                'Service Type': location_data.get('service_type', ''),
                'Address': location_data.get('address', ''),
                'Contact': location_data.get('contact', ''),
                'Website': location_data.get('website', ''),
                'Services': website_data.get('services', '') if website_data else '',
                'Infrastructure': website_data.get('infrastructure', '') if website_data else '',
                'Contact Details': website_data.get('contact', '') if website_data else '',
                'Brands': website_data.get('brands', '') if website_data else ''
            }

            data.append(entry)
            progress_bar.progress((idx + 1) / len(all_links))

        except Exception as e:
            logger.error(f"Error processing URL {link}: {str(e)}")
            continue

    status_text.text("Processing complete!")
    return pd.DataFrame(data)


def main():
    st.set_page_config(layout="wide")
    st.title("🚗 Automotive Business Analyzer")

    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input(
                "Enter your search query (e.g., 'Car repair services in New York')"
            )
        with col2:
            limit = st.number_input(
                "Number of results",
                min_value=1,
                max_value=100,
                value=10
            )

    if st.button("Search"):
        if search_query:
            with st.spinner("Fetching data..."):
                try:
                    # Scrape and process data
                    links = scrape_data(search_query, limit)
                    if not links:
                        st.warning("No results found. Try modifying your search query.")
                        return

                    df = process_each_url(links)

                    # Display results
                    st.subheader("Results")
                    st.dataframe(df)

                    # Export options
                    csv = df.to_csv(index=False).encode('utf-8')
                    buffer = BytesIO()

                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=False)
                    buffer.seek(0)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "Download CSV",
                            csv,
                            "automotive_business_data.csv",
                            "text/csv",
                            key='download-csv'
                        )
                    with col2:
                        st.download_button(
                            "Download Excel",
                            buffer,
                            "automotive_business_data.xlsx",
                            "application/vnd.ms-excel",
                            key='download-excel'
                        )

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    logger.error(f"Main execution error: {str(e)}")
        else:
            st.warning("Please enter a search query")


if __name__ == "__main__":
    main()