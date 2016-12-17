import pymysql.cursors

connection = pymysql.connect(user='root', password='root',
                             database='GRE')

cursor = connection.cursor()

cursor.execute("CREATE TABLE IF NOT EXISTS GRES (No int, Sentence VARCHAR(1000));")
connection.commit()

commit = "select count(*) from GREA1"
cursor.execute(commit)
row = cursor.fetchall()[0][0]
No = 0
for i in range(row):
    cursor = connection.cursor()
    commit = "select Answer from GREA1 where No = %d;" % (i + 1)
    cursor.execute(commit)
    right = cursor.fetchall()[0][0]
    commit = "select %s from GREA1 where No = %d;" % (right, (i + 1))
    cursor.execute(commit)
    answer = cursor.fetchall()[0][0]
    commit = "select Former from GREQ1 where No = %d;" % (i + 1)
    cursor.execute(commit)
    former = cursor.fetchall()[0][0]
    commit = "select Later from GREQ1 where No = %d;" % (i + 1)
    cursor.execute(commit)
    later = cursor.fetchall()[0][0]
    sentence = former + answer + later
    No += 1
    with connection.cursor() as cursor:
        # Create a new record
        sql = "INSERT INTO GRES "
        sql += "(No, Sentence) VALUES (%s, %s)"
        cursor.execute(sql, (No, sentence))
    connection.commit()
    print No

cursor = connection.cursor()
commit = "select count(*) from GREA2"
cursor.execute(commit)
row = cursor.fetchall()[0][0]
for j in range(row):
    for i in range(1, 3):
        cursor = connection.cursor()
        commit = "select Answer%d from GREA2 where No = %d;" % (i, (j + 1))
        cursor.execute(commit)
        right = cursor.fetchall()[0][0]
        commit = "select %s from GREA2 where No = %d;" % (right, (j + 1))
        cursor.execute(commit)
        answer = cursor.fetchall()[0][0]
        commit = "select Former from GREQ2 where No = %d;" % (j + 1)
        cursor.execute(commit)
        former = cursor.fetchall()[0][0]
        commit = "select Later from GREQ2 where No = %d;" % (j + 1)
        cursor.execute(commit)
        later = cursor.fetchall()[0][0]
        sentence = former + answer + later
        No += 1
        with connection.cursor() as cursor:
            # Create a new record
            sql = "INSERT INTO GRES "
            sql += "(No, Sentence) VALUES (%s, %s)"
            cursor.execute(sql, (No, sentence))
        connection.commit()
        print No