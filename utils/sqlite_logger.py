"""
SQLite 日志记录功能
"""

import logging
import datetime
import sqlite3
import pandas as pd

def sqlite_log(task_id, step_description, result_type, result):
    """记录任务执行日志到 SQLite 数据库
    
    Args:
        task_id: 任务ID
        step_description: 步骤描述
        result_type: 结果类型
        result: 结果内容
    """
    logging.info(f"记录日志: task_id={task_id}, step={step_description}, result_type={result_type}, result={result}")
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = {
        'timestamp': timestamp,
        'task_id': task_id,
        'step': step_description,
        'result_type': result_type,
        'result': result
    }
    
    try:
        df = pd.DataFrame([log_entry])
        with sqlite3.connect('task_logs.db') as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    timestamp TEXT,
                    task_id TEXT,
                    step TEXT,
                    result_type TEXT,
                    result TEXT
                )
            ''')
            df.to_sql('logs', conn, if_exists='append', index=False)
            conn.commit()
    except Exception as e:
        print(f"写入日志失败: {e}") 