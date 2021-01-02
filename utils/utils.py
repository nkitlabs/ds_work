
import pandas as pd
import random
import string
import numpy as np

def gen_dataframe(n_rows, fix_cols={}, rand_cols={}):
    """return random pandas dataframe
    
    parameters:
    n_rows: Integer
        number of data being generated
    fix_cols: a dictionary; default empty
        a dictionary determine a column name and its value.
    rand_cols: a dictionary; default empty
        a dictionary determine a column name and types of random or random list or function that return a value.
        Possible values of types of random are 'int', 'long', 'string', 'str', 'double', 'float'.
        
    Examples
    --------
    To generate a dataframe
    
    >>> data = gen_dataframe(3, fix_cols={'A': 0}, rand_cols={'B':'int', 'C':[1,2,3,4]})
    >>> data
         A    B    C
    0    0  235    4
    1    0   91    1
    2    0  794    2
    """
    
    if not isinstance(fix_cols, dict):
        raise Exception('fix_cols must be a dictionary type')
    if not isinstance(rand_cols, dict):
        raise Exception('rand_cols must be a dictionary type')
    
    table = []
    column_name = []
    
    
    for k,v in fix_cols.items():
        fix_data = [[v]] * n_rows
        table.append(fix_data)
        column_name.append(k)
    
    for k,v in rand_cols.items():
        if isinstance(v, list):
            rand_data = [[random.choice(v)] for x in range(n_rows)]
        elif callable(v):
            rand_data = [[v()] for x in range(n_rows)]
        elif v in ['int', 'long']:
            rand_data = [[random.randrange(1000)] for x in range(n_rows)]
        elif v in ['float', 'double']:
            rand_data = [[random.random()] for x in range(n_rows)]
        elif v in ['string', 'str']:
            rand_data = [[''.join(random.choice(string.ascii_lowercase + string.digits) for i in range(10))] for x in range(n_rows)]
        else:
            raise Exception(f'type of column {k} is not list, function, \'int\', \'long\', \'float\', \'double\', \'string\', \'str\'')
        table.append(rand_data)
        column_name.append(k)
    
    if len(table) == 0:
        table.append([[0]] *n_rows)
        column_name.append('tmp_column')
    final_table = table[0]
    for _data in table[1:]:
        final_table = [ _x1 + _x2 for _x1, _x2 in zip(final_table, _data)]
    return pd.DataFrame(final_table, columns=column_name)
    
# data = gen_dataframe(20, fix_cols={'column_A': 0, 'column_B': 3.0}, rand_cols={'column_C':'int', 'column_D':[1,2,3,4]})
