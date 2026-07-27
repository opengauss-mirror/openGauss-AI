import unittest
from pathlib import Path
from unittest import mock

from psycopg2 import sql as psycopg2_sql

from advisor.index_advisor import postgres_env


MODULE_PATH = Path(__file__).resolve().parents[1] / "postgres_env.py"


class RecordingCursor:
    def __init__(self, table_name="orders", column_name="id", index_size=10):
        self.calls = []
        self.table_name = table_name
        self.column_name = column_name
        self.index_size = index_size
        self.closed = False

    def execute(self, query, params=None):
        self.calls.append((query, params))

    def fetchall(self):
        query = str(self.calls[-1][0])
        if "information_schema.tables" in query:
            return [(self.table_name,)]
        if "information_schema.columns" in query:
            return [(self.column_name,)]
        if "hypopg_create_index" in query:
            return [(42,)]
        if "hypopg_relation_size" in query:
            return [(self.index_size,)]
        raise AssertionError(f"Unexpected fetch for query: {query}")

    def close(self):
        self.closed = True


class RecordingConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.closed = False

    def cursor(self):
        return self._cursor

    def close(self):
        self.closed = True


class PostgresEnvSourceTest(unittest.TestCase):
    def test_module_is_valid_python(self):
        source = MODULE_PATH.read_text(encoding="utf-8")

        try:
            compile(source, str(MODULE_PATH), "exec")
        except SyntaxError as exc:
            self.fail(f"postgres_env.py must be valid Python: {exc}")


class PostgresEnvSqlCompositionTest(unittest.TestCase):
    def test_catalog_queries_bind_schema_and_table_names(self):
        schema_name = "tenant'; DROP SCHEMA public; --"
        table_name = "orders'; DROP TABLE accounts; --"
        cursor = RecordingCursor(table_name=table_name)
        connection = RecordingConnection(cursor)

        with mock.patch.object(postgres_env.config_raw, "get", return_value=""), \
                mock.patch.object(postgres_env.psycopg2, "connect", return_value=connection):
            postgres_env.postgresql_env("tpch", schema_name, 1024)

        self.assertEqual((schema_name,), cursor.calls[0][1])
        self.assertNotIn(schema_name, str(cursor.calls[0][0]))
        self.assertEqual((schema_name, table_name), cursor.calls[1][1])
        self.assertNotIn(table_name, str(cursor.calls[1][0]))

    def test_hypopg_ddl_quotes_identifiers_and_binds_the_result(self):
        schema_name = 'tenant"; DROP SCHEMA public; --'
        table_name = 'orders"; DROP TABLE accounts; --'
        column_name = 'customer"; DROP TABLE accounts; --'
        cursor = RecordingCursor()
        connection = RecordingConnection(cursor)
        env = object.__new__(postgres_env.postgresql_env)
        env.schema_name = schema_name

        self.assertTrue(
            hasattr(env, "_create_hypothetical_index"),
            "postgresql_env must centralize safe HypoPG index creation",
        )

        def quote_identifier(value, _context):
            return '"' + value.replace('"', '""') + '"'

        with mock.patch.object(
                psycopg2_sql.ext, "quote_ident", side_effect=quote_identifier):
            index_id = env._create_hypothetical_index(
                connection, cursor, table_name, column_name
            )

        expected_ddl = (
            'CREATE INDEX ON "tenant""; DROP SCHEMA public; --".'
            '"orders""; DROP TABLE accounts; --" '
            '("customer""; DROP TABLE accounts; --")'
        )
        self.assertEqual(42, index_id)
        self.assertEqual(
            ("SELECT * FROM hypopg_create_index(%s);", (expected_ddl,)),
            cursor.calls[0],
        )

    def test_step_binds_hypothetical_index_id(self):
        malicious_index_id = "42); DROP TABLE accounts; --"
        cursor = RecordingCursor(index_size=10)
        connection = RecordingConnection(cursor)
        env = object.__new__(postgres_env.postgresql_env)
        env.dbname = "tpch"
        env.schema_name = "public"
        env.db_table_list = ["orders"]
        env.db_column_list = [["id"]]
        env.cur_state = [[0]]
        env.cur_cost = 100
        env.budget_size = 1024
        env.cur_size_use = 0
        env.cur_virtual_index = []
        env.termial_wait_num = 0
        env.action_space = [[0, 0]]
        env.action_acum_pos = [0]

        with mock.patch.object(postgres_env.config_raw, "get", return_value=""), \
                mock.patch.object(postgres_env.psycopg2, "connect", return_value=connection), \
                mock.patch.object(
                    env,
                    "_create_hypothetical_index",
                    return_value=malicious_index_id,
                    create=True,
                ) as create_index, \
                mock.patch.object(env, "cal_cur_cost", return_value=90):
            env.step([0, 0])

        create_index.assert_called_once_with(
            connection, cursor, "orders", "id"
        )
        relation_size_calls = [
            call for call in cursor.calls
            if "hypopg_relation_size" in str(call[0])
        ]
        self.assertEqual(
            [("SELECT * FROM hypopg_relation_size(%s);", (malicious_index_id,))],
            relation_size_calls,
        )

    def test_cost_calculation_uses_safe_hypopg_helper(self):
        cursor = RecordingCursor()
        connection = RecordingConnection(cursor)
        env = object.__new__(postgres_env.postgresql_env)
        env.dbname = "tpch"
        env.schema_name = "public"
        env.db_table_list = ["orders'; DROP TABLE accounts; --"]
        env.db_column_list = [["id'; DROP TABLE accounts; --"]]
        env.cur_virtual_index = [[0, 0]]
        env.workload = []

        with mock.patch.object(postgres_env.config_raw, "get", return_value=""), \
                mock.patch.object(postgres_env.psycopg2, "connect", return_value=connection), \
                mock.patch.object(
                    env, "_create_hypothetical_index", create=True
                ) as create_index:
            cost = env.cal_cur_cost()

        self.assertEqual(0, cost)
        create_index.assert_called_once_with(
            connection,
            cursor,
            "orders'; DROP TABLE accounts; --",
            "id'; DROP TABLE accounts; --",
        )


if __name__ == "__main__":
    unittest.main()
