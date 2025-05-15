import MySQLdb
import pandas as pd
import json

def fetch_package_metadata_interpretations(
        db: MySQLdb.Connection,
) -> tuple[str, dict[str, str]]:
    cursor = db.cursor()

    try:
        cursor.execute(f"SELECT field_name, field_local_name, field_interpretation FROM package_metadata_interpretations")
        rows = cursor.fetchall()
        context = ""
        field_name_to_local_name_map = {}
        for row in rows:
            context += f"- {row[1]}: {row[2]}\n"
            field_name_to_local_name_map[row[0]] = row[1]
        return context, field_name_to_local_name_map
    finally:
        cursor.close()

def fetch_interpretations_and_packages(
        db: MySQLdb.Connection,
) -> tuple[str, dict[str, str], pd.DataFrame]:
    metadata_context, field_name_to_local_name_map = fetch_package_metadata_interpretations(db)
    packages_df = pd.DataFrame(columns=[*field_name_to_local_name_map.values()])

    cursor = db.cursor()

    try:
        cursor.execute(f"SELECT name, metadata FROM packages")
        rows = cursor.fetchall()
        for row in rows:
            df_row = {}
            try:
                metadata = row[1]
                m = json.loads(metadata)
                for field_name, field_value in m.items():
                    if field_value:
                        field_local_name = field_name_to_local_name_map.get(field_name, field_name)
                        df_row[field_local_name] = field_value
            except json.JSONDecodeError:
                print(f"Invalid JSON in metadata for package {row[0]}")
                continue
            # packages_df = packages_df.append(df_row, ignore_index=True) # type: ignore
            df_row["Mã dịch vụ"] = row[0]
            packages_df_new_row = pd.DataFrame([df_row])
            packages_df = pd.concat([packages_df, packages_df_new_row], ignore_index=True)
        return metadata_context, field_name_to_local_name_map, packages_df
    finally:
        cursor.close()

def fetch_faqs(
    db: MySQLdb.Connection,
) -> pd.DataFrame:
    cursor = db.cursor()

    try:
        cursor.execute(f"SELECT question, answer FROM faqs")
        rows = cursor.fetchall()
        faqs_df = pd.DataFrame(rows, columns=["Câu hỏi thường gặp", "Câu trả lời"])
        return faqs_df
    finally:
        cursor.close()
