from typing import List

from sqlalchemy import create_engine, inspect


class MySqlDatabaseManager:
    """
    MySql数据库管理类
    """
    def __init__(self, connection_string: str):
        """
        初始化mysql 数据库连接
        Arg :
            connection_string mysql连接字符串
        """
        self.engine = create_engine(connection_string,pool_size=5, pool_recycle=3600)

       # self.engine = create_engine(connection_string,pool_size=5, pool_recycle=3600)

    def get_table_names(self) -> List[str]:
        """
        获取数据库中所有的表名
        """
        try:
            inspector = inspect(self.engine)
            return inspector.get_table_names()
        except Exception as e:
            print(f"Error: {e}")
            raise ValueError(f"获取表名称失败：{str(e)}")


# ... existing code ...

if __name__ == "__main__":
    DB_CONFIG={
        "host": "localhost",
        "port": 3306,
        "user": "root",
        "password": "yzx12345.",
        "database": "lumen_bi",
    }
    connection_string = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?charset=utf8mb4"
    db_manager = MySqlDatabaseManager(connection_string)
    table_names = db_manager.get_table_names()




    print(table_names)

# ... existing code ...

