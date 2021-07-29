import time


class Query(object):
    def __init__(self, sql="", q_id=None):
        self.sql=sql
        self.id=q_id if q_id else hash(sql+str(time.time()))
        self.execution_time=None
        self.sql_psy_plan=None
        self.serialized_plan=None

class View(object):
    def __init__(self, sql="", v_id=None):
        self.sql=sql
        self.id=v_id if v_id else hash(sql+str(time.time()))
        self.frequency=0
        self.related_queries=[]
        self.execution_time=None    # TODO
        self.sql_psy_plan=None
        self.serialized_plan=None
        self.size=None              # TODO

class Query_view_pair(object):
    def __init__(self, query, view):
        self.query=query
        self.view=view
        self.query_view_execution_time=None
        self.benefit=0
        self.embedding=None