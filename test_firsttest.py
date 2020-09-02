import time
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

path_to_driver = 'C:\Program Files (x86)\chromedriver_altered2.exe'


class TestFirsttest():
  def setup_method(self, method):
    self.options = webdriver.ChromeOptions()
    self.options.add_experimental_option("excludeSwitches", ["enable-automation"])
    self.options.add_experimental_option('useAutomationExtension', False)
    self.options.add_argument('--disable-infobars')
    self.options.add_argument('--disable-extensions')
    self.options.add_argument('--profile-directory=Default')
 #   self.options.add_argument('--incognito')
    self.options.add_argument('--disable-plugins-discovery')
    self.options.add_argument('--start-maximized')

    self.driver = webdriver.Chrome(options = self.options, executable_path = method)
    self.vars = {}
  
  def teardown_method(self, method):
    self.driver.quit()
  
  def test_firsttest(self):
    self.driver.get("https://ras.arbitr.ru/")
    self.driver.set_window_size(1000, 800)
    self.driver.find_element(By.CSS_SELECTOR, ".b-promo_notification-popup-close").click()
#    self.driver.find_element(By.CSS_SELECTOR, "#caseType .b-icon > i").click()
    self.driver.find_element(By.CSS_SELECTOR, "#caseType .down-button").click()
    self.driver.find_element(By.CSS_SELECTOR, ".b-suggest:nth-child(14) li:nth-child(7)").click()
    self.driver.find_element(By.CSS_SELECTOR, "button").click()
    #self.driver.find_element(By.LINK_TEXT, "2").click()
    element = self.driver.find_element(By.ID, "pages")
    actions = ActionChains(self.driver)
    actions.move_to_element(element).release().perform()
  def get_user_agent(self):
    self.driver.get("https://google.com/")
    self.driver.set_window_size(988, 824)
    self.user_agent = self.driver.execute_script("return navigator.userAgent;")
    print(self.user_agent)

#%%

first_test = TestFirsttest()

first_test.setup_method(path_to_driver)

#first_test.get_user_agent()
# My Selenium user agent is 
# Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36


# Maybe try to change it to this
# Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36
#%%


first_test.test_firsttest()

# %%
first_test.teardown_method("dmm")

# %%
path_to_driver = 'C:\Program Files (x86)\geckodriver.exe'


class TestFirsttest():
  def setup_method(self, method):
    self.options = webdriver.FirefoxOptions()
    self.options.add_experimental_option("excludeSwitches", ["enable-automation"])
    self.options.add_experimental_option('useAutomationExtension', False)
    self.driver = webdriver.Firefox(options = self.options, executable_path = method)
    self.vars = {}
  
  def teardown_method(self, method):
    self.driver.quit()
  
  def test_firsttest(self):
    self.driver.get("https://ras.arbitr.ru/")
    self.driver.set_window_size(988, 824)
    self.driver.find_element(By.CSS_SELECTOR, ".b-promo_notification-popup-close").click()
    time.sleep(1)
#    self.driver.find_element(By.CSS_SELECTOR, "#caseType .b-icon > i").click()
    self.driver.find_element(By.CSS_SELECTOR, "#caseType .down-button").click()
    time.sleep(0.5)
    self.driver.find_element(By.CSS_SELECTOR, ".b-suggest:nth-child(14) li:nth-child(7)").click()
    time.sleep(1.5)
    self.driver.find_element(By.CSS_SELECTOR, "button").click()
    #self.driver.find_element(By.LINK_TEXT, "2").click()
    #element = self.driver.find_element(By.ID, "pages")
    actions = ActionChains(self.driver)
    actions.move_to_element(element).release().perform()


first_test = TestFirsttest()

first_test.setup_method(path_to_driver)
first_test.test_firsttest()
