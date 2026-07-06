import sqlite3

# conn = sqlite3.connect("smartdesk_tickets.db")
# cursor = conn.cursor()
# cursor.execute("SELECT * FROM ticket_links WHERE ticket_key = ?", ("KAN-25",))     # update to last ticket key you created
# print(cursor.fetchall())
# conn.close()


conn = sqlite3.connect("smartdesk_tickets.db")
cursor = conn.cursor()
cursor.execute("SELECT * FROM ticket_links ORDER BY id DESC LIMIT 1")
print(cursor.fetchall())
conn.close()