from data_access import get_db_connection
conn = get_db_connection(); cur = conn.cursor()
cur.execute("""
    SELECT conrelid::regclass::text, confrelid::regclass::text,
           CASE confdeltype WHEN 'a' THEN 'NO ACTION' WHEN 'r' THEN 'RESTRICT' WHEN 'c' THEN 'CASCADE'
                            WHEN 'n' THEN 'SET NULL' ELSE 'OTHER' END
    FROM pg_constraint
    WHERE contype = 'f'
      AND (conrelid IN ('schedule'::regclass, 'schedule_runs'::regclass)
           OR confrelid IN ('schedule'::regclass, 'schedule_runs'::regclass, 'rooms'::regclass, 'timeslots'::regclass))
    ORDER BY 1, 2""")
for tbl, ref, rule in cur.fetchall(): print(f"{tbl:<26} -> {ref:<22} ON DELETE {rule}")
cur.execute("SELECT count(*), pg_size_pretty(pg_total_relation_size('schedule')) FROM schedule_runs")
print("מערכות בהיסטוריה / גודל טבלת schedule:", cur.fetchone())
conn.close()
