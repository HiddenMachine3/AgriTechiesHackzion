from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

import os

# Initialize WebDriver
driver = webdriver.Chrome()


def get_images(query: str, folder_path: str):
    global driver
    # Go to Google Images
    driver.get("https://www.google.com/imghp?hl=EN")

    # Find the search box, enter "apples", and submit the form
    search_box = driver.find_element(By.NAME, "q")
    search_box.send_keys(query)
    search_box.send_keys(Keys.RETURN)

    # Wait for the results to load
    driver.implicitly_wait(10)
    # Find the image results
    image_results = driver.find_elements(By.CSS_SELECTOR, ".rg_i")

    # save the images
    count = 0
    for image in image_results:
        if count == 100:
            break
        # when i do drive.find_elements for an image search using selenium in google chrome, i want to get the url of each image. Write code to print all attributes present in each result
        image.screenshot(f'{folder_path}/{query.replace(" ","_")}_{count}.png')
        count += 1


for query in [
    "unripe apple",
    "ripe apple",
    "rotten apple",
]:
    #  i have a folder in my project directory. I have  to get the absolute path of this folder write me code to get the absolute path of this folder
    absolute_path = os.path.abspath("data/object_detection")
    folder_name = query.split(" ")[0]
    folder_path = os.path.join(absolute_path, folder_name)

    # if folder_path doesn't exist, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    print(folder_path)
    get_images(query, folder_path)


driver.quit()
