from xxlimited import foo
from bs4 import BeautifulSoup
import requests
import pandas as pd
from time import sleep
from random import randint
from urllib.parse import urljoin
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from selenium.webdriver.common.by import By

from selenium import webdriver
import time

class DataManager:
    def __init__(self):
        self._soup = {}
        self._need = {}
        self._url = {}

    def getData(self, pages):
        j = 0
        url = self._url.get(j)
        if url is None:
            #self._url[1] = "https://www.vivareal.com.br/venda/santa-catarina/florianopolis/apartamento_residencial/"
            self._url[1] = "https://www.vivareal.com.br/venda/santa-catarina/sao-jose/apartamento_residencial/"
            #self._url[1] = "https://www.vivareal.com.br/venda/santa-catarina/florianopolis/apartamento_residencial/com-espaco-gourmet/#area-ate=120&area-desde=60&com=espaco-gourmet,piscina&onde=BR-Santa_Catarina-NULL-Florianopolis,BR-Santa_Catarina-NULL-Sao_Jose,BR-Santa_Catarina-NULL-Biguacu,BR-Santa_Catarina-NULL-Palhoca&preco-ate=5000000&preco-desde=100000"
            #self._url[1] = 'https://www.vivareal.com.br/venda/santa-catarina/florianopolis/apartamento_residencial/?pagina=2#area-ate=120&area-desde=70'

        need = self._need.get(j)
        if need is None:
            url = self._url.get(1)

            driver = webdriver.Chrome()
            driver.get(url)
            for page in range(1,pages):

                _address = driver.find_elements("class name","property-card__address")
                _area = driver.find_elements("xpath","//li[contains(@class, 'property-card__detail-item property-card__detail-area')]")
                _room = driver.find_elements("xpath","//li[contains(@class, 'property-card__detail-item property-card__detail-room js-property-detail-rooms')]")
                _bath = driver.find_elements("xpath","//li[contains(@class, 'property-card__detail-item property-card__detail-bathroom js-property-detail-bathroom')]")
                _price = driver.find_elements("xpath","//div[contains(@class, 'property-card__price js-property-card-prices js-property-card__price-small')]")
                _link = driver.find_elements("xpath","//div[contains(@class, 'property-card__carousel js-property-carousel')]/a")

                address = [address.text for address in _address]
                area = [area.text for area in _area]
                room = [room.text for room in _room]
                bath = [bath.text for bath in _bath]
                price = [price.text for price in _price]

                link = [link.get_attribute('href') for link in _link]

                for lenght in range(0,len(address)):
                    self._need[j] = {'address':address[lenght],'area':area[lenght],'rooms':room[lenght],'bathroom':bath[lenght],'price':price[lenght], 'link':link[lenght]}
                    j+=1

                nextoption = driver.find_element('xpath',"//li[@class='pagination__item']//button[@title='Próxima página']")
                driver.execute_script("arguments[0].scrollIntoView(true);", nextoption)
                driver.execute_script("window.scrollBy(0,-300)")
                nextoption.click()
                time.sleep(5)
            driver.quit()
        
        df = pd.DataFrame.from_dict(self._need,orient='index')

        return df
