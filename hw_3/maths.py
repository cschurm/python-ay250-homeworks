#code for maths operation of Monty code hw3
import requests
from bs4 import BeautifulSoup

from urllib import urlencode
words = command.split()
the_calculation = ' '.join(words[2:])
words = command.split()

#url_values = urlencode(query)
q = '+'.join(the_calculation.split())
url = 'https://www.google.com/search?q=' + q + '&ie=utf-8&oe=utf-8'
req = requests.get(url)
#print(req.text)
soup = BeautifulSoup(req.text, "lxml")
print(list(x.get_text() for x in soup.find_all("h2", class_="r")))
