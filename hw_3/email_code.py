
# coding: utf-8

# In[ ]:


import smtplib #initialize email
# sign into gmail
gmail_user = 'cschurman125@gmail.com' #sorry, to grade this you're gonna have to use your own!  
gmail_password = 'marimbaluver'
# open smtp server
try:  
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
except:  
    print 'Something went wrong...'

# login to email
try:  
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.ehlo()
    server.login(gmail_user, gmail_password)
except:  
    print 'Something went wrong...'

#construct email text
sent_from = gmail_user  
to = ['cschurman125@gmail.com']   

subject = words[words.index("subject")+1] #extracts the subject line, this method only works on single words following "subject
body = words[words.index("body")+1] #extracts the body word, this method only works on single words following "body"

email_text = """\  
From: %s  
To: %s  
Subject: %s

%s
""" % (sent_from, ", ".join(to), subject, body)

try:  
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.ehlo()
    server.login(gmail_user, gmail_password)
    server.sendmail(sent_from, to, email_text)
    server.close()

    print 'Email sent!'
except:  
    print 'Something went wrong...'


