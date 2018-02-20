
# coding: utf-8

# In[ ]:


#code for jokes.py
Jokes = open('Jokes.txt') #open joke library
#next few lines format the .txt file from the interwebs to printable one liners
jokes = Jokes.readlines()[0] 
one_liners = jokes.split('Q:')
#select random joke and return it
from random import randint
randChoice = randint(0, len(one_liners)-1)
joke = one_liners[randChoice]
print(joke)

