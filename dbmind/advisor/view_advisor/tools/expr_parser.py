#!/usr/bin/python
import re

class Expr_parser(object):
    def __init__(self):
        # 忽略一元减号 -
        self.operation_list={
            ':=':{'pr':0, 'op_num':2},
            "||":{'pr':1, 'op_num':2},
            "OR":{'pr':1, 'op_num':2},
            "XOR":{'pr':1, 'op_num':2},
            "&&":{'pr':2, 'op_num':2},
            
            
            "BETWEEN":{'pr':4, 'op_num':3},

            "AND":{'pr':2, 'op_num':2},
            
            "CASE":{'pr':4, 'op_num':0},
            "WHEN":{'pr':4, 'op_num':1},
            "THEN":{'pr':4, 'op_num':1},
            "ELSE":{'pr':4, 'op_num':1},
            "=":{'pr':5, 'op_num':2},
            
            ">=":{'pr':5, 'op_num':2},
            ">":{'pr':5, 'op_num':2},
            "<=":{'pr':5, 'op_num':2},
            "<":{'pr':5, 'op_num':2},
            "<>":{'pr':5, 'op_num':2},
            "!=":{'pr':5, 'op_num':2},
            "IS":{'pr':5, 'op_num':2},
            "LIKE":{'pr':5, 'op_num':2},
			"~~":{'pr':5, 'op_num':2},  #LIKE
			"!~~":{'pr':5, 'op_num':2},  #NOT LIKE
            "REGEXP":{'pr':5, 'op_num':2},
            "IN":{'pr':5, 'op_num':2},
            "|":{'pr':6, 'op_num':2},
            "&":{'pr':7, 'op_num':2},

            "-":{'pr':9, 'op_num':2},
            "+":{'pr':9, 'op_num':2},
            "*":{'pr':10, 'op_num':2},
            "/":{'pr':10, 'op_num':2},
            "DIV":{'pr':10, 'op_num':2},
            "%":{'pr':10, 'op_num':2},
            "MOD":{'pr':10, 'op_num':2},
            "^":{'pr':11, 'op_num':2},
            "~":{'pr':12, 'op_num':2},
            "!":{'pr':13, 'op_num':1},

            "NOT":{'pr':13, 'op_num':1},
			"ANY":{'pr':13, 'op_num':1},
            
            "BINARY":{'pr':14, 'op_num':1},
            "COLLATE":{'pr':14, 'op_num':1},
        }
        # "<=>":{'pr':5, 'op_num':2},
        # "<":{'pr':8, 'op_num':2},
        # ">>":{'pr':8, 'op_num':2},

        self.reserved_key={
            "NULL"
        }

        self.rcs={
            "text":re.compile(r"'(.*)'::text$"),
            "text_lis":re.compile(r"'\{(.*)\}'::text\[\]$"),
            "extract_text_lis":re.compile(r'(?<=,)(([^ ]*?)|("(.*?)"))(?=,)'),
            "col_string":re.compile(r"([A-Za-z0-9_$#]+)::text$"),
            "col_numeric":re.compile(r"([A-Za-z0-9_$#]+)::numeric$"),
            "num_numeric":re.compile(r"'([0-9]*(\.[0-9]*)?)'::numeric$"),
            "number":re.compile(r"(-?[0-9]*(\.[0-9]*)?)$"),
            "col_with_tb":re.compile(r"([A-Za-z0-9_$#]+)\.([A-Za-z0-9_$#]+)$"),
            "col":re.compile(r"([A-Za-z0-9_$#]+)$")
        }

    def parse_expr(self, expr, table_name=None, alias_table_map=None):
        
        expr=self.preprocess(expr)
        seq=self.op_node2seq(self._parse_expression(expr))
        result_seq=[]
        for e in seq:
            result_seq.extend(self.classify_element(e, table_name, alias_table_map))
        
        return result_seq

    def classify_element(self, e, table_name=None, alias_table_map=None):
        if table_name and alias_table_map and table_name in alias_table_map:
            table_name=alias_table_map[table_name]
        table_prefix=table_name+"." if table_name else ""
        e=e.strip()
        if e in self.operation_list:
            return [(e,'keyword')]
        elif e in self.reserved_key:
            return [(e,'keyword')]
        elif self.rcs["text"].match(e):
            content=self.rcs['text'].search(e).group(1)
            return [(content,"string")]
        elif self.rcs["text_lis"].match(e):
            content=self.rcs['text_lis'].search(e).group(1)
            content=","+content+","
            str_lis=self.rcs["extract_text_lis"].findall(content)
            str_lis=[x[1] or x[3] for x in str_lis]
            return [(x,'string') for x in str_lis]
        elif self.rcs["col_string"].match(e):
            content=self.rcs["col_string"].search(e).group(1)
            return [(table_prefix+content,"identifier")]
        elif self.rcs["col_numeric"].match(e):
            content=self.rcs["col_numeric"].search(e).group(1)
            return [(table_prefix+content,"identifier")]
        elif self.rcs["num_numeric"].match(e):
            content=self.rcs["num_numeric"].search(e).group(1)
            return [(content,"number")]
        elif self.rcs["number"].match(e):
            content=self.rcs["number"].search(e).group(1)
            return [(content,"number")]
        elif self.rcs["col_with_tb"].match(e):
            res=self.rcs["col_with_tb"].search(e)
            tb=res.group(1)
            col=res.group(2)
            if alias_table_map and tb in alias_table_map:
                tb=alias_table_map[tb]
            return [(tb+"."+col, "identifier")]
        elif self.rcs["col"].match(e):
            col=self.rcs["col"].search(e).group(1)
            return [(table_prefix+col, "identifier")]
        else:
            raise Exception("can not classify element "+e)

    def _parse_expression(self, expr):
        e="("+expr+")"
        stk_op=[]
        stk_num=[]
        last_pos=0
        quote=None
        for i in range(1,len(e)):
            c=e[i]
            if c in {'"', "'"}:
                if quote is None:
                    quote=c+"1"
                elif (quote==c or quote==c+"1") and e[i+1]==':':
                    quote=None

            if quote is not None:
                if quote[-1]!='1':continue
                else: quote=quote[:-1]

            if c in {" ","(",")"} or e[i-1]=='(':
                while last_pos<i-1 and e[last_pos] in {" ","(",")"}: last_pos+=1
                obj=e[last_pos:i]
                last_pos=i
            else:
                continue

            if obj in {" ",""}:
                continue
            elif obj=="(":
                stk_op.append(obj)
                continue
            elif obj==")":
                while len(stk_op)>0 and stk_op[-1]!="(":
                    self.pop_op(stk_op, stk_num)
                stk_op.pop()
            elif obj in self.operation_list:
                while len(stk_op)>0 and self.prior_LE(obj, stk_op[-1]):
                    self.pop_op(stk_op, stk_num)
                stk_op.append(obj)
            else:
                stk_num.append(obj)
        
        # the last char should be ')'
        while len(stk_op)>0 and stk_op[-1]!="(":
            self.pop_op(stk_op, stk_num)
        stk_op.pop()

        if len(stk_op)>0 or len(stk_num)>1:
            raise Exception("expr op rest or num rest")

        result=stk_num[0]
        return result

    def op_node2seq(self, op_node):
        seq_lis=[]
        self.traverse_nodetree(op_node, seq_lis)
        return seq_lis

    def traverse_nodetree(self, op_node, result_seq):
        for obj in op_node['num_list']:
            if isinstance(obj, dict):
                self.traverse_nodetree(obj, result_seq)
            else:
                result_seq.append(obj)
        result_seq.append(op_node['op_name'])
            
    def op_node(self, op_name, op_num, num_list):
        result=dict(op_name=op_name,op_num=op_num,num_list=num_list)
        return result

    def pop_op(self, stk_op, stk_num):
        if not stk_op:
            return
        op=stk_op[-1]
        op_num=self.operation_list[op]['op_num']
        if len(stk_num)<op_num:
            raise Exception("stk length small than op num")
        
        new_op_node=self.op_node(op, op_num, stk_num[-op_num:])
        
        stk_op.pop()
        for i in range(op_num):stk_num.pop()
        
        stk_num.append(new_op_node)
        
        return

    def prior_LE(self, op1, op2):
        if op2=="(": return False
        else: return self.operation_list[op1]['pr']<=self.operation_list[op2]['pr']

    def preprocess(self, expr):
        prog=re.compile(r'(\(([^()]+?)\)::)')
        match_list=prog.findall(expr)
        # ('(temp)::', 'temp')
        for x,y in match_list:
            expr=expr.replace(x,y+"::")
        return expr
        

if __name__=="__main__":
    expr="((hours >= '9'::numeric) AND (hours <= '90'::numeric) AND (((tb.hours > '10.56'::numeric) AND (hours < '100'::numeric)) OR (hours > '110'::numeric)))"
    # expr="(info = ANY ('{Sweden,Norway,Germany,Denmark,Swedish,Denish,Norwegian,German}'::text[]))"
    expr="(info = ANY ('{Sweden,Norway,Germany,Denmark,Swedish,Denish,Norwegian,German}'::text[]))"
    expr="((info)::text = 'bottom 10 rank'::text)"
    expr="(note IS NOT NULL)"
    expr="((tb.info IS NOT NULL) AND ((info ~~ 'Japan:%200%'::text) OR (info ~~ 'USA:%200%'::text)))"
    expr="(((note)::text !~~ '%(2011)%'::text) AND (((note)::text ~~ '%(201%)%'::text) OR ((note)::text ~~ '%(worldwide)%'::text)))"
    expr="((note)::text = '(''Alien'' characters)'::text)"
    expr_parser=Expr_parser()
    lis=expr_parser.parse_expr(expr, "work", {"tb":"table_name"})
    for line in lis:
        print(line)
