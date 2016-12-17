# coding=utf-8
import urllib2
import re
import time
import pymysql.cursors
import requests
from bs4 import BeautifulSoup

# XDF红皮 单选 134题
# Connect to the database
connection = pymysql.connect(user='root', password='root',
                             database='GRE')

cursor = connection.cursor()
commit = "CREATE TABLE IF NOT EXISTS GREQ (No int, Former VARCHAR(500), Later VARCHAR(500));"
cursor.execute(commit)
connection.commit()
commit = "CREATE TABLE IF NOT EXISTS GREA (No int, A VARCHAR(50), B VARCHAR(50), C VARCHAR(50), D VARCHAR(50), E VARCHAR(50), Answer VARCHAR(50));"
cursor.execute(commit)
connection.commit()
ba = 0
No = 39
for i in range(3, 4):
    url = "http://gre.kmf.com/learn/newworkbookindex?&source=tc&sub=3&cate_id=64&p=%d" % (1 + i)
    content = urllib2.urlopen(url).read()
    bases = re.findall(r' <a href="(.*?)" target="_blank" class="list-oplink">查看详情</a>', content)
    for base in bases:
        ba += 1
        url0 = "http://gre.kmf.com" + base
        content0 = urllib2.urlopen(url0).read()
        # print content0
        sets = re.findall(r' <td class="tc"><p><a href="(.*?)" target="_blank">查看详情</a></p></td>', content0)
        set = 0
        for j in range(len(sets)):
            set += 1
            url1 = "http://gre.kmf.com" + sets[j]
            response = requests.get(url1)
            soup = BeautifulSoup(response.text, "lxml")
            # print soup
            question = soup.find_all("div", class_="mb20")[1].find("div", class_="mb20").string
            print "Base：%d   Set：%d" % (ba, set)
            print question.replace("\n", "")
            Right = soup.find("div", class_="exa-correctanswer", id="ShowSiderAnswer").find("span").string
            if len(Right) == 1:
                No += 1
                if No >= 135:
                    question = question.replace("    ", "").replace("\n", "")
                    question = question.replace(u"\u2018", "'").replace(u"\u2019", "'").replace(u'\u2014',u'-').replace(u"\u201c","\"").replace(u"\u201d","\"")
                    question = question.split("_____")
                    former = question[0].replace(".", "").replace(",", "")
                    later = question[1].replace(".", "").replace(",", "")
                    options = soup.find_all("li", class_="clearfix")
                    A = re.findall(r'<span><strong>.*?</strong>(.*?)</span>', str(options[0]))[0]
                    B = re.findall(r'<span><strong>.*?</strong>(.*?)</span>', str(options[1]))[0]
                    C = re.findall(r'<span><strong>.*?</strong>(.*?)</span>', str(options[2]))[0]
                    D = re.findall(r'<span><strong>.*?</strong>(.*?)</span>', str(options[3]))[0]
                    E = re.findall(r'<span><strong>.*?</strong>(.*?)</span>', str(options[4]))[0]
                    print A, B, C, D, E
                    print Right
                    with connection.cursor() as cursor:
                        # Create a new record
                        sql = "INSERT INTO GREQ "
                        sql += "(No, Former, Later) VALUES (%s, %s, %s)"
                        cursor.execute(sql, (No, former, later))
                    connection.commit()
                    with connection.cursor() as cursor:
                        # Create a new record
                        sql = "INSERT INTO GREA "
                        sql += "(No, A, B, C, D, E, Answer) VALUES (%s, %s, %s, %s, %s, %s, %s)"
                        cursor.execute(sql, (No, A, B, C, D, E, Right))
                    connection.commit()
            time.sleep(0.05)
connection.close()