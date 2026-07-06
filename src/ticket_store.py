from datetime import datetime
import sqlite3

def link_ticket(email: str, ticket_key: str) -> None:
    """
    Records the association between an email and a Jira ticket key in a SQLite database.

    Args:
        email (str): The email address of the user who created the ticket.
        ticket_key (str): The key of the Jira ticket that was created.

    This function opens a connection to a SQLite database named "smartdesk_tickets.db", 
    creates a table named "ticket_links" if it doesn't already exist, and inserts a new record linking 
    the provided email to the ticket key along with the current timestamp. 
    The connection is closed after the operation is complete.
    """

    conn = sqlite3.connect("smartdesk_tickets.db")
    cursor = conn.cursor()

    # Create the ticket_links table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ticket_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            ticket_key TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    # Insert the new link
    cursor.execute("""
        INSERT INTO ticket_links (email, ticket_key, created_at)
        VALUES (?, ?, ?)
    """, (email, ticket_key, datetime.now().isoformat()))

    # Commit and close the connection
    conn.commit()
    conn.close()


def get_ticket_keys_for_email(email: str) -> list[str]:
    """
    Retrieves all Jira ticket keys associated with a given email from the SQLite database.

    Args:
        email (str): The email address to look up.

    Returns:
        list[str]: A list of ticket keys associated with the email. Returns an empty list if none are found.
    This function opens a connection to the "smartdesk_tickets.db" SQLite database, queries the "ticket_links" table for all ticket keys 
    associated with the provided email, and returns them as a list. The connection is closed after the operation is complete.
    """
    conn = sqlite3.connect("smartdesk_tickets.db")
    cursor = conn.cursor()

    # Create the ticket_links table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ticket_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            ticket_key TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    # Query for all ticket keys associated with the given email
    cursor.execute("""
        SELECT ticket_key FROM ticket_links WHERE email = ?
    """, (email,))
    
    rows = cursor.fetchall()
    ticket_keys = [row[0] for row in rows]

    # Close the connection
    conn.close()

    return ticket_keys

