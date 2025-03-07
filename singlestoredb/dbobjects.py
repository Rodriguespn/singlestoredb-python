import datetime
import math
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type

from dateutil import parser
from pydantic import BaseModel
from pydantic import create_model
from pydantic import Field

from .connection import Connection


DB_TYPEMAP: Dict[str, Tuple[Type[Any], Dict[str, Any], Callable[[Any], Any]]] = dict(
    char=(str, {}, str),
    varchar=(str, dict(max_length=21844), str),
    tinyint=(int, dict(ge=-128, le=127), int),
    smallint=(int, dict(ge=-32768, le=32767), int),
    mediumint=(int, dict(ge=-8388608, le=8388607), int),
    int=(int, dict(ge=-2147483648, le=2147483647), int),
    bigint=(int, dict(ge=-2**63, le=(2**63)-1), int),
    binary=(bytes, {}, bytes),
    varbinary=(bytes, dict(max_length=65533), bytes),
    tinyblob=(bytes, dict(max_length=215), bytes),
    mediumblob=(bytes, dict(max_length=16777216), bytes),
    blob=(bytes, dict(max_length=65535), bytes),
    longblob=(bytes, dict(max_length=4194304000), bytes),
    tinytext=(str, dict(max_length=255), str),
    mediumtext=(str, dict(max_length=16777216), str),
    text=(str, dict(max_length=65535), str),
    longtext=(str, dict(max_length=4194304000), str),
    float=(float, {}, float),
    double=(float, {}, float),
    datetime=(datetime.datetime, {}, parser.parse),  # noqa
    time=(datetime.timedelta, {}, lambda x: parser.parse(x).time()),  # noqa
    date=(datetime.date, {}, lambda x: parser.parse(x).date()),  # noqa
    timestamp=(datetime.datetime, {}, parser.parse),  # noqa
    year=(int, dict(ge=1901, le=2155), int),
)


class TableBaseModel(BaseModel):
    pass


class TableRowBaseModel(BaseModel):
    pass


class Table:

    def __init__(self, conn: Connection, database: str, name: str) -> None:
        self.connection = conn
        self.database = database
        self.name = name
        self.schema = self._schema()

    def insert(self, model: TableBaseModel) -> int:
        subs = ', '.join([f'%({x["COLUMN_NAME"]})s' for x in self.schema])
        names = ', '.join([f'`{x["COLUMN_NAME"]}`' for x in self.schema])
        query = f'INSERT INTO `{self.database}`.`{self.name}`({names}) VALUES ({subs})'
        with self.connection.cursor() as cur:
            if isinstance(model, TableBaseModel):
                out = cur.executemany(query, model.model_dump()['rows'])
            elif isinstance(model, TableRowBaseModel):
                out = cur.executemany(query, model.model_dump())
            else:
                raise TypeError('Unrecognized parameter type for insert')
        return out

    def insert_completions(self, client: Any, **kwargs: Any) -> int:
        kwargs['response_model'] = self.table_model()
        kwargs['messages'] = list(kwargs.get('messages', []))

        has_system_msg = False
        has_user_msg = False
        for msg in kwargs['messages']:
            if msg['role'] == 'system':
                has_system_msg = True
            elif msg['role'] == 'user':
                has_user_msg = True

        if not has_system_msg:
            kwargs['messages'].insert(
                0, dict(role='system', content='You are a helpful assistant'),
            )

        kwargs['messages'].insert(
            1, dict(
                role='assistant', content=self._table_info()['TABLE_COMMENT'],
            ),
        )

        if not has_user_msg:
            kwargs['messages'].append(
                dict(role='user', content=self._table_info()['TABLE_COMMENT']),
            )

        return self.insert(client.create(**kwargs))

    def _schema(self) -> List[Dict[str, Any]]:
        query = '''
            SELECT * FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s ORDER BY ORDINAL_POSITION
        '''
        out = []
        with self.connection.cursor() as cur:
            cur.execute(query, (self.database, self.name))
            names = [x.name for x in cur.description or []]
            for row in cur:
                out.append({k: v for k, v in zip(names, row)})
        return out

    def _table_info(self) -> Dict[str, Any]:
        query = '''
            SELECT * FROM information_schema.TABLES
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
        '''
        with self.connection.cursor() as cur:
            cur.execute(query, (self.database, self.name))
            names = [x.name for x in cur.description or []]
            for row in cur:
                return {k: v for k, v in zip(names, row)}
        return {}

    def table_model(self) -> Type[TableBaseModel]:
        tbl_info = self._table_info()
        desc = tbl_info.get('TABLE_COMMENT') or 'rows of data in the table'
        row_model = self.row_model()
        tbl_model = create_model(
            'TableModel',
            rows=(List[row_model], Field(description=desc, default=[])),  # type: ignore
            __base__=(TableBaseModel,),
        )
        return tbl_model

    def row_model(self) -> Type[TableRowBaseModel]:
        row_model = create_model(
            'RowModel',
            **dict([self._get_model_field(x) for x in self.schema]),
            __base__=(TableRowBaseModel,),
        )
        return row_model

    def _get_model_field(
        self,
        info: Dict[str, Any],
    ) -> Tuple[str, Tuple[Type[Any], Field]]:
        is_required = 'N' in info['IS_NULLABLE']

        dtype, dtype_params, dtype_conv = DB_TYPEMAP[info['DATA_TYPE']]
        if not is_required:
            dtype = Optional[dtype]  # type: ignore

        kwargs = dtype_params.copy()

        if info['COLUMN_COMMENT'].strip():
            kwargs['description'] = info['COLUMN_COMMENT'].strip()

        if info['COLUMN_DEFAULT']:
            kwargs['default'] = dtype_conv(info['COLUMN_DEFAULT'])
        elif not is_required:
            kwargs['default'] = None

        max_length = info['CHARACTER_MAXIMUM_LENGTH']
        if max_length is not None and not math.isnan(max_length):
            kwargs['max_length'] = int(max_length)

        if dtype in ['decimal']:
            max_digits = info['NUMERIC_PRECISION']
            if max_digits is not None and not math.isnan(max_digits):
                kwargs['max_digits'] = int(max_digits)

            decimal_places = info['NUMERIC_SCALE']
            if decimal_places is not None and not math.isnan(decimal_places):
                kwargs['decimal_places'] = int(decimal_places)

        return (str(info['COLUMN_NAME']), (dtype, Field(**kwargs)))


class Database:

    def __init__(self, conn: Connection, name: str) -> None:
        self.connection = conn
        self.name = name

    @property
    def tables(self) -> Dict[str, Table]:
        with self.connection.cursor() as cur:
            cur.execute(f'SHOW TABLES IN `{self.name}`')
            return {k[0]: Table(self.connection, self.name, k[0]) for k in cur}


def dbs(self: Connection) -> Dict[str, Database]:
    with self.cursor() as cur:
        cur.execute('SHOW DATABASES')
        return {k[0]: Database(self, k[0]) for k in cur}
