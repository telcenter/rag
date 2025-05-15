import MySQLdb

def update_summary_and_satisfaction(conn: MySQLdb.Connection, row_id: int, summary: str, customer_satisfaction):
    """
    Update 'summary' and 'customer_satisfaction' for a row in the customer_service_chats table.

    Args:
        conn (MySQLdb.Connection): Active MySQLdb connection.
        row_id (int): ID of the row to update.
        summary (str): New summary value.
        customer_satisfaction (int or float): New satisfaction score.
    """
    cursor = None
    try:
        cursor = conn.cursor()
        query = """
            UPDATE customer_service_chats
            SET summary = %s,
                customer_satisfaction = %s
            WHERE id = %s
        """
        cursor.execute(query, (summary, customer_satisfaction, row_id))
        conn.commit()
        print(f"Updated row with id={row_id}")
    except MySQLdb.Error as e:
        print("MySQL Error:", e)
        conn.rollback()
    finally:
        if cursor:
            cursor.close()
