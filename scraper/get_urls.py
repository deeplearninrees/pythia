from selenium import webdriver
import selenium
from selenium.webdriver.common.keys import Keys
import time
import atexit
import random
import copy

from selenium.webdriver.common.by import By

#Scrapes and finds video links from YT recommendations on homepage in Chrome using webdriver and autosaves them...
#Assumes that you are on the scraper/ subdirectory of  the principal Ari folder.

opts = webdriver.ChromeOptions()
opts.headless = True
driver = webdriver.Chrome('../webdriver/chromedriver', options=opts)

driver.get('https://youtube.com')
try:
    set_of_hrefs  = set(open('../links.txt', 'r').read().splitlines())
except FileNotFoundError as e:
    set_of_hrefs = set()
old_set = copy.deepcopy(set_of_hrefs)
atexit.register(lambda: open("../links.txt", mode='w').write('\n'.join(set_of_hrefs)))
acc = 0
tabs = 0
prev_url = ''
print("HEADLESS YOUTUBE RECOMENDATIONS SCRAPER 5000")
while True:
    try:
        old_set = copy.deepcopy(set_of_hrefs)
        try:
            hrefs = [video.get_attribute("href") for video in driver.find_elements(By.ID, 'thumbnail')] 
        except:
            hrefs = []
        for href in hrefs:
            if isinstance(href,str):
                if href.startswith("https://www.youtube.com/watch?v=") and len(href) == 43: #Length of youtube video link.
                    set_of_hrefs.add(href)
        html = driver.find_element(By.TAG_NAME, 'html')
        html.send_keys(Keys.PAGE_DOWN)
        if len(set_of_hrefs) != len(old_set):
            link = random.choice(list(set_of_hrefs-old_set))
            driver.get(link)
            # time_to_sleep = 0
            # for i in range(20):
            #     time_to_sleep += np.random.choice(4, p=[0.9, 0.1/3, 0.1/3, 0.1/3])
            print("On page", link)
            # print("Sleeping for {} seconds...".format(time_to_sleep))
            # time.sleep(time_to_sleep)
        if acc % 80 == 0 and acc != 0:
            if "https://www.youtube.com/watch?v=" in driver.current_url:
                if acc % (80 * 10) == 0:
                    driver.close()
                    print('Restarting Chrome browser to get a fresh start...')
                    driver = webdriver.Chrome('../webdriver/chromedriver', options=opts)            
                    driver.get("https://www.youtube.com")
                else:
                    print('Refreshing to home page...')
                    driver.get("https://www.youtube.com")
                open("../links.txt", mode='w').write('\n'.join(set_of_hrefs))
        acc += 1
        if len(set_of_hrefs) != len(old_set):
            print("Time:", acc, "\nLinks gathered:", len(set_of_hrefs), "\nAdded:", len(set_of_hrefs-old_set))

        if ("https://www.youtube.com/watch?v=" not in driver.current_url and driver.current_url != "https://www.youtube.com/") or len(driver.window_handles) > 1:
            driver.close()
            driver = webdriver.Chrome('../webdriver/chromedriver', options=opts)
            driver.get("https://www.youtube.com")
        prev_url = driver.current_url
    except selenium.common.exceptions.WebDriverException as e:
        print("Received WebDriverException, meaning there's probably no internet. Attempting to reconnect...")
        driver.close()
        driver = webdriver.Chrome('../webdriver/chromedriver', options=opts)
        c = True
        while c:
            try:
                driver.get("https://www.youtube.com")
                c = False
            except Exception as e:
                pass
            time.sleep(1)