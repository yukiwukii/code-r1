import ast
from collections import defaultdict

def _is_name(node, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name

def _attr_chain_endswith(node, suffix: str) -> bool:
    if isinstance(node, ast.Attribute):
        return node.attr == suffix
    if isinstance(node, ast.Name):
        return node.id == suffix
    return False

def _full_attr_name(node) -> str | None:
    
    parts = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None

class PatternDetector(ast.NodeVisitor):
    def __init__(self):
        self.found = defaultdict(bool)

        
        self._func_stack = []
        self._calls_in_func = defaultdict(set)

        
        self._list_append_targets = set()
        self._list_pop_targets = set()
        self._list_pop0_targets = set()
        self._deque_append_targets = set()
        self._deque_pop_targets = set()
        self._deque_popleft_targets = set()
        self._vars_assigned_deque = set()
        self._vars_assigned_list = set()

       
        self._class_fields = defaultdict(set)

        
        self._graph_adj_vars = set()          
        self._vars_assigned_defaultdict = set()  
        self._vars_assigned_dict = set()        
        self._vars_assigned_nx_graph = set()   

       
        self._adj_subscript_append = set()     
        self._adj_subscript_add = set()       
        self._adj_subscript_setdefault = set() 

        
        self._edges_container_inserts = set() 

    def visit_If(self, node):
        self.found["if statement"] = True
        self.generic_visit(node)

    def visit_For(self, node):
        self.found["for loop"] = True
        self.generic_visit(node)

    def visit_AsyncFor(self, node):
        self.found["for loop"] = True
        self.generic_visit(node)

    def visit_While(self, node):
        self.found["while loop"] = True
        self.generic_visit(node)

    def visit_Break(self, node):
        self.found["break statement"] = True

    def visit_Continue(self, node):
        self.found["continue statement"] = True

    def visit_Pass(self, node):
        self.found["pass statement"] = True

    def visit_Match(self, node):
        self.found["match statement"] = True
        self.generic_visit(node)

    def visit_Tuple(self, node):
        self.found["tuple"] = True
        self.generic_visit(node)

    def visit_Set(self, node):
        self.found["set"] = True
        self.generic_visit(node)

    def visit_Dict(self, node):
        self.found["dictionary"] = True
        self.generic_visit(node)

    def visit_Call(self, node):
        
        if _is_name(node.func, "tuple"):
            self.found["tuple"] = True
        elif _is_name(node.func, "set"):
            self.found["set"] = True
        elif _is_name(node.func, "dict"):
            self.found["dictionary"] = True

        
        if self._func_stack:
            current = self._func_stack[-1]
            called = None
            if isinstance(node.func, ast.Name):
                called = node.func.id
            elif isinstance(node.func, ast.Attribute):
                called = node.func.attr
            if called:
                self._calls_in_func[current].add(called)


        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            var = node.func.value.id
            meth = node.func.attr

            if meth == "append":
                self._list_append_targets.add(var)
                self._deque_append_targets.add(var)

                if node.args and isinstance(node.args[0], ast.Tuple) and len(node.args[0].elts) == 2:
                    self._edges_container_inserts.add(var)

            elif meth == "pop":
                self._list_pop_targets.add(var)
                self._deque_pop_targets.add(var)
                if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == 0:
                    self._list_pop0_targets.add(var)

            elif meth == "popleft":
                self._deque_popleft_targets.add(var)

            elif meth == "add":
               
                if node.args and isinstance(node.args[0], ast.Tuple) and len(node.args[0].elts) == 2:
                    self._edges_container_inserts.add(var)

            elif meth == "setdefault":
                self._adj_subscript_setdefault.add(var)

        
        fn_full = _full_attr_name(node.func)
        if fn_full in {
            "nx.Graph", "nx.DiGraph", "nx.MultiGraph", "nx.MultiDiGraph",
            "networkx.Graph", "networkx.DiGraph", "networkx.MultiGraph", "networkx.MultiDiGraph",
        }:
            
            pass

        self.generic_visit(node)

    def visit_Assign(self, node):
        targets = [t for t in node.targets if isinstance(t, ast.Name)]
        if targets:
            for t in targets:
                name = t.id

               
                if isinstance(node.value, ast.List):
                    self._vars_assigned_list.add(name)

                elif isinstance(node.value, ast.Call) and _is_name(node.value.func, "list"):
                    self._vars_assigned_list.add(name)

                elif isinstance(node.value, ast.Call) and _attr_chain_endswith(node.value.func, "deque"):
                    self._vars_assigned_deque.add(name)

                elif isinstance(node.value, ast.Dict):
                    self._vars_assigned_dict.add(name)
                    self.found["dictionary"] = True

                elif isinstance(node.value, ast.Call) and _is_name(node.value.func, "dict"):
                    self._vars_assigned_dict.add(name)
                    self.found["dictionary"] = True

                elif isinstance(node.value, ast.Call) and _attr_chain_endswith(node.value.func, "defaultdict"):
                    self._vars_assigned_defaultdict.add(name)
          
                    if node.value.args:
                        arg0 = node.value.args[0]
                        if _is_name(arg0, "list") or _is_name(arg0, "set") or _is_name(arg0, "dict"):
                            self._graph_adj_vars.add(name)

              
                elif isinstance(node.value, ast.Call):
                    fn_full = _full_attr_name(node.value.func)
                    if fn_full in {
                        "nx.Graph", "nx.DiGraph", "nx.MultiGraph", "nx.MultiDiGraph",
                        "networkx.Graph", "networkx.DiGraph", "networkx.MultiGraph", "networkx.MultiDiGraph",
                    }:
                        self._vars_assigned_nx_graph.add(name)

              
                if name.lower() in {"adj", "graph", "neighbors", "neighbours", "edges", "adjacency"}:
                    self._graph_adj_vars.add(name)

        self.generic_visit(node)

    def visit_ClassDef(self, node):
        class_name = node.name
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for sub in ast.walk(item):
                    if isinstance(sub, ast.Assign):
                        for tgt in sub.targets:
                            if (
                                isinstance(tgt, ast.Attribute)
                                and isinstance(tgt.value, ast.Name)
                                and tgt.value.id == "self"
                            ):
                                self._class_fields[class_name].add(tgt.attr)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self._func_stack.append(node.name)
        self.generic_visit(node)
        self._func_stack.pop()

    def visit_AsyncFunctionDef(self, node):
        self._func_stack.append(node.name)
        self.generic_visit(node)
        self._func_stack.pop()


    def visit_Attribute(self, node):
        
       
        if isinstance(node.value, ast.Subscript) and isinstance(node.value.value, ast.Name):
            base = node.value.value.id
            if node.attr == "append":
                self._adj_subscript_append.add(base)
            elif node.attr == "add":
                self._adj_subscript_add.add(base)
        self.generic_visit(node)

    def finalize(self):
        
        for func_name, called in self._calls_in_func.items():
            if func_name in called:
                self.found["recursion"] = True
                break

        
        for v in (self._vars_assigned_list | self._list_append_targets | self._list_pop_targets):
            if v in self._list_append_targets and v in self._list_pop_targets and v not in self._list_pop0_targets:
                self.found["stack"] = True
                break
        if not self.found["stack"]:
            for v in (self._vars_assigned_deque | self._deque_append_targets | self._deque_pop_targets):
                if v in self._deque_append_targets and v in self._deque_pop_targets:
                    self.found["stack"] = True
                    break

      
        for v in (self._vars_assigned_deque | self._deque_append_targets | self._deque_popleft_targets):
            if v in self._deque_append_targets and v in self._deque_popleft_targets:
                self.found["queue"] = True
                break
        if not self.found["queue"]:
            for v in (self._vars_assigned_list | self._list_append_targets | self._list_pop0_targets):
                if v in self._list_append_targets and v in self._list_pop0_targets:
                    self.found["queue"] = True
                    break

       
        for fields in self._class_fields.values():
            if "next" in fields or "prev" in fields:
                self.found["linked list"] = True
                break

     
        for fields in self._class_fields.values():
            if ("left" in fields and "right" in fields) or ("children" in fields):
                self.found["tree"] = True
                break



        strong = False
        moderate_score = 0

        if self._adj_subscript_append or self._adj_subscript_add:
            strong = True


        if self._vars_assigned_nx_graph:
            strong = True


        if self._vars_assigned_defaultdict:
            moderate_score += 1
        if self._graph_adj_vars & self._vars_assigned_defaultdict:
            moderate_score += 1  

     
        if self._adj_subscript_setdefault:
            moderate_score += 1
        if self._graph_adj_vars & self._adj_subscript_setdefault:
            moderate_score += 1

        if self._vars_assigned_dict and (self._graph_adj_vars & self._vars_assigned_dict):
            moderate_score += 1

        
        if self._edges_container_inserts:
            moderate_score += 1
        if {n.lower() for n in self._edges_container_inserts} & {"edges", "edge_list", "edgelist"}:
            moderate_score += 1

        if strong or moderate_score >= 2:
            self.found["graph"] = True

        return dict(self.found)


def detect_patterns(code: str):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {
            "if statement": False,
            "for loop": False,
            "while loop": False,
            "break statement": False,
            "continue statement": False,
            "pass statement": False,
            "match statement": False,
            "recursion": False,
            "stack": False,
            "queue": False,
            "tuple": False,
            "set": False,
            "dictionary": False,
            "linked list": False,
            "tree": False,
            "graph": False,
        }

    d = PatternDetector()
    d.visit(tree)
    found = d.finalize()

    keys = [
        "if statement", "for loop", "while loop", "break statement",
        "continue statement", "pass statement", "match statement", "recursion",
        "stack", "queue", "tuple", "set", "dictionary",
        "linked list", "tree", "graph",
    ]
    return {k: bool(found.get(k, False)) for k in keys}


