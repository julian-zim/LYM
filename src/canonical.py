"""Canonical form of Lua card scripts, built on the abstract syntax tree (AST).

Two scripts can behave identically while differing in their text: independent statements in another order, functions
defined in another order, other names, other comments or formatting. For training this is noise (the model has to guess
an arbitrary order), for evaluation it is unfair (a correct script counts as wrong). The canonical form removes these
differences: the script is parsed into an AST, rewritten into one fixed form, and printed again.

Rewrites (all preserve what the script does):
  1. Comments are removed and the formatting is unified (tab indentation, fixed spacing, short tables on one line).
  2. Effect setter calls (e1:SetType(...), e1:SetCode(...), ...) that follow each other on the same effect are sorted
     into one fixed order, unless the engine makes their order matter (see SETTER_CONFLICTS).
  3. Consecutive top-level function definitions (function s.xxx(...) ... end) are sorted by first use: initial_effect
     first, then each function in the order it is first referenced, depth first.
  4. Only with rename=True (for comparing scripts, not for training labels): all local variables and parameters are
     renamed to v1, v2, ... in order of declaration, and the script's own functions to f1, f2, ... in order of
     definition. So e.g. s.spcon vs s.spcon2 or tc vs sc no longer matter.

Every result is verified: the printed script is parsed again and must give exactly the rewritten AST. This guards
against printer bugs (luaparser for example prints '- -x' as '--x', which Lua reads as a comment).
"""
from textwrap import indent

from luaparser import ast, printers
from luaparser import astnodes as A


class CanonicalizationError(ValueError):
    """The script could not be parsed (syntax error) or not be printed back faithfully."""


# ---------------------------------------------------------------------------------------------------------------------
# Rule 2: Effect setters
# ---------------------------------------------------------------------------------------------------------------------
# The Effect setters, in their canonical order (the order EDOPro scripts conventionally use). Each of them only stores
# its arguments in the effect object (see libeffect.cpp in the EDOPro core), so their order is irrelevant, except for
# the pairs in SETTER_CONFLICTS and repeated calls of the same setter (the last one wins).
EFFECT_SETTERS = (
    "SetDescription", "SetCategory", "SetType", "SetProperty", "SetCode", "SetRange", "SetTargetRange",
    "SetAbsoluteRange", "SetCountLimit", "SetHintTiming", "SetCondition", "SetCost", "SetTarget", "SetOperation",
    "SetValue", "SetLabel", "SetLabelObject", "SetReset", "SetOwnerPlayer",
)
SETTER_RANK = {name: rank for rank, name in enumerate(EFFECT_SETTERS)}
# Setter pairs whose order matters, from the EDOPro core:
#   SetType overwrites the range (for EFFECT_TYPE_ACTIVATE) and the code (for EFFECT_TYPE_FLIP),
#   SetTargetRange and SetAbsoluteRange both write the target range and the EFFECT_FLAG_ABSOLUTE_TARGET flag.
# Not conflicting despite touching the same flag field: SetProperty keeps the internal flag bits 0x4f (INITIAL,
# FUNC_VALUE, COUNT_LIMIT, FIELD_ONLY, ABSOLUTE_TARGET) that SetValue, SetCountLimit and SetAbsoluteRange change.
SETTER_CONFLICTS = {
    frozenset({"SetType", "SetRange"}),
    frozenset({"SetType", "SetCode"}),
    frozenset({"SetTargetRange", "SetAbsoluteRange"}),
}
# Calls that may appear in setter arguments without blocking the reordering, because they have no side effects:
# functions of the script libraries' namespaces (used in setter arguments as factories, e.g. Cost.PayLP(1000)), and
# getters (Get.../Is.../Has.../Check...) on anything but the effect being set up.
PURE_NAMESPACES = {"aux", "Cost", "Fusion", "Ritual", "Synchro", "Xyz", "Link", "Pendulum"}
GETTER_PREFIXES = ("Get", "Is", "Has", "Check")


def _setter_receiver(statement) -> str | None:
    """If the statement is a reorderable Effect setter call like 'e1:SetCode(...)', return the effect variable name."""
    if not (
        isinstance(statement, A.Invoke)
        and isinstance(statement.source, A.Name)
        and isinstance(statement.func, A.Name)
        and statement.func.id in SETTER_RANK
    ):
        return None
    receiver = statement.source.id
    return receiver if all(_is_pure(arg, receiver) for arg in statement.args) else None


def _is_pure(expression, receiver: str) -> bool:
    """True if evaluating the expression has no side effects and does not read the effect being set up.

    Only the parts evaluated right away count: the body of an anonymous function runs later, not when the setter is
    called, so it is not inspected.
    """
    if isinstance(expression, A.AnonymousFunction):
        return True
    if isinstance(expression, A.Name):
        return expression.id != receiver  # the receiver itself would expose its (changing) state
    if isinstance(expression, A.Invoke):  # obj:Method(...)
        if not expression.func.id.startswith(GETTER_PREFIXES):
            return False
    elif isinstance(expression, A.Call):  # Namespace.func(...)
        func = expression.func
        if not (isinstance(func, A.Index) and isinstance(func.value, A.Name) and isinstance(func.idx, A.Name)
                and func.notation == A.IndexNotation.DOT):
            return False
        if not (func.value.id in PURE_NAMESPACES
                or (func.value.id == "Duel" and func.idx.id.startswith(GETTER_PREFIXES))):
            return False
        return all(_is_pure(arg, receiver) for arg in expression.args)
    return all(_is_pure(child, receiver) for child in _children(expression))


def _sort_setter_run(run: list) -> list:
    """Sort consecutive setter calls on one effect into canonical order, keeping conflicting pairs in their order.

    This is a topological sort: a call may only be placed once every earlier call it conflicts with is placed, and
    among the placeable calls the one with the lowest canonical rank goes first.
    """
    methods = [statement.func.id for statement in run]
    must_follow = [
        {i for i in range(j) if methods[i] == methods[j] or frozenset({methods[i], methods[j]}) in SETTER_CONFLICTS}
        for j in range(len(run))
    ]
    placed, result = set(), []
    while len(result) < len(run):
        ready = [k for k in range(len(run)) if k not in placed and must_follow[k] <= placed]
        k = min(ready, key=lambda k: (SETTER_RANK[methods[k]], k))
        placed.add(k)
        result.append(run[k])
    return result


def _sort_setters(block: A.Block) -> None:
    """Apply _sort_setter_run to every maximal run of reorderable setter calls on the same effect in the block."""
    body, out, i = block.body, [], 0
    while i < len(body):
        receiver = _setter_receiver(body[i])
        j = i + 1
        if receiver is not None:
            while j < len(body) and _setter_receiver(body[j]) == receiver:
                j += 1
        out.extend(_sort_setter_run(body[i:j]) if j - i > 1 else body[i:j])
        i = j
    block.body = out


# ---------------------------------------------------------------------------------------------------------------------
# Rule 3: top-level function definitions
# ---------------------------------------------------------------------------------------------------------------------


def _module_name(chunk: A.Chunk) -> str | None:
    """Name of the script's own table, i.e. 's' in 'local s,id=GetID()'."""
    for statement in chunk.body.body:
        if (isinstance(statement, A.LocalAssign) and statement.values and isinstance(statement.values[0], A.Call)
                and isinstance(statement.values[0].func, A.Name) and statement.values[0].func.id == "GetID"):
            return statement.targets[0].id
    return None


def _module_field(node, module: str) -> str | None:
    """'xxx' if the node is the expression 's.xxx', else None."""
    if (isinstance(node, A.Index) and node.notation == A.IndexNotation.DOT and isinstance(node.value, A.Name)
            and node.value.id == module and isinstance(node.idx, A.Name)):
        return node.idx.id
    return None


def _defined_function(statement, module: str) -> str | None:
    """'xxx' if the statement is 'function s.xxx(...) ... end', else None."""
    return _module_field(statement.name, module) if isinstance(statement, A.Function) else None


def _referenced_fields(node, module: str):
    """All 's.xxx' field names referenced inside the node, in traversal order."""
    for child in _walk(node):
        field = _module_field(child, module)
        if field is not None:
            yield field


def _sort_functions(chunk: A.Chunk, module: str) -> dict[str, int]:
    """Sort runs of consecutive top-level 'function s.xxx' definitions by first use. Returns the rank of each name.

    Only consecutive definitions are reordered: moving a definition across another statement (e.g. a 'local function'
    helper) could change which variables its body sees. Within such a run, the order of definition does not matter,
    as each one only assigns a different field of s. Repeated definitions of one name keep their order (the sort is
    stable and they share a rank), since the last one wins.
    """
    statements = chunk.body.body
    definitions: dict[str, list] = {}
    for statement in statements:
        name = _defined_function(statement, module)
        if name is not None:
            definitions.setdefault(name, []).append(statement)

    order: list[str] = []

    def visit(name: str) -> None:  # depth-first, so helpers follow right after the function that first uses them
        if name in definitions and name not in order:
            order.append(name)
            for definition in definitions[name]:
                for field in _referenced_fields(definition.body, module):
                    visit(field)

    visit("initial_effect")
    for statement in statements:  # functions only referenced from other top-level code (e.g. tables of functions)
        if _defined_function(statement, module) is None:
            for field in _referenced_fields(statement, module):
                visit(field)
    for name in definitions:  # never referenced: keep their original relative order at the end
        visit(name)
    rank = {name: i for i, name in enumerate(order)}

    out, i = [], 0
    while i < len(statements):
        j = i
        while j < len(statements) and _defined_function(statements[j], module) is not None:
            j += 1
        if j > i:
            out.extend(sorted(statements[i:j], key=lambda s: rank[_defined_function(s, module)]))
            i = j
        else:
            out.append(statements[i])
            i += 1
    chunk.body.body = out
    return rank


# ---------------------------------------------------------------------------------------------------------------------
# Rule 4: renaming (only for comparisons)
# ---------------------------------------------------------------------------------------------------------------------


class _Renamer:
    """Renames local variables (v1, v2, ...) and the script's functions (f1, f2, ...), respecting Lua's scoping.

    Lua scoping rules that matter here: 'local x = x' evaluates the right side before x is declared; 'local function f'
    is declared before its body (so it can call itself); loop variables and parameters live in the body's scope; the
    condition of 'repeat ... until cond' sees the body's locals. Field names (s.xxx, t.key, obj:method) are no
    variables, so they are left alone, except for the script's own functions (s.xxx defined via 'function s.xxx').
    """

    def __init__(self, module: str | None, function_names: list[str]):
        self.scopes: list[dict[str, str]] = [{}]
        self.n_locals = 0
        self.module_binding = None  # the new name of 's' once declared
        self.module = module
        # initial_effect is called by the engine under this exact name, so it keeps it
        self.functions = {name: f"f{i + 1}" for i, name in enumerate(n for n in function_names if n != "initial_effect")}

    # --- scopes ---
    def declare(self, name_node: A.Name) -> None:
        if name_node.id == "self":  # implicit parameter of methods, cannot be renamed
            self.scopes[-1]["self"] = "self"
            return
        self.n_locals += 1
        new = f"v{self.n_locals}"
        if name_node.id == self.module and self.module_binding is None and len(self.scopes) == 2:
            self.module_binding = new  # 'local s,id=GetID()' at the top level of the script
        self.scopes[-1][name_node.id] = new
        name_node.id = new

    def resolve(self, name_node: A.Name) -> None:
        for scope in reversed(self.scopes):
            if name_node.id in scope:
                name_node.id = scope[name_node.id]
                return  # not found: a global (Duel, aux, constants, ...), which keeps its name

    def in_scope(self, visit_inside) -> None:
        self.scopes.append({})
        visit_inside()
        self.scopes.pop()

    # --- traversal ---
    def block(self, block: A.Block) -> None:
        self.in_scope(lambda: [self.visit(statement) for statement in block.body])

    def function(self, args: list, body: A.Block, implicit_self: bool = False) -> None:
        def inside():
            if implicit_self:
                self.scopes[-1]["self"] = "self"
            for arg in args:
                if isinstance(arg, A.Name):
                    self.declare(arg)
            self.block(body)
        self.in_scope(inside)

    def visit(self, node) -> None:
        if isinstance(node, list):
            for item in node:
                self.visit(item)
        elif not isinstance(node, A.Node):
            return
        elif isinstance(node, A.Chunk):
            self.block(node.body)
        elif isinstance(node, A.Block):
            self.block(node)
        elif isinstance(node, A.LocalAssign):
            self.visit(node.values)  # evaluated before the new locals exist
            for target in node.targets:
                self.declare(target)
        elif isinstance(node, A.LocalFunction):
            self.declare(node.name)  # declared before the body, so the function can call itself
            self.function(node.args, node.body)
        elif isinstance(node, A.Function):
            self.visit(node.name)
            self.function(node.args, node.body)
        elif isinstance(node, A.Method):
            self.visit(node.source)  # node.name is a method name, not a variable
            self.function(node.args, node.body, implicit_self=True)
        elif isinstance(node, A.AnonymousFunction):
            self.function(node.args, node.body)
        elif isinstance(node, A.Fornum):
            self.visit([node.start, node.stop, node.step])
            self.in_scope(lambda: (self.declare(node.target), self.block(node.body)))
        elif isinstance(node, A.Forin):
            self.visit(node.iter)
            self.in_scope(lambda: ([self.declare(t) for t in node.targets], self.block(node.body)))
        elif isinstance(node, A.Repeat):  # 'until' sees the body's locals, so both share one scope
            self.in_scope(lambda: ([self.visit(s) for s in node.body.body], self.visit(node.test)))
        elif isinstance(node, A.Name):
            self.resolve(node)
        elif isinstance(node, A.Index):
            module_field = _module_field(node, self.module) if self.module else None
            self.visit(node.value)
            if node.notation != A.IndexNotation.DOT:
                self.visit(node.idx)  # t[expr]: the index is an expression
            elif (module_field in self.functions and isinstance(node.value, A.Name)
                  and node.value.id == self.module_binding):
                node.idx.id = self.functions[module_field]  # s.xxx, where s is the script's table
        elif isinstance(node, A.Invoke):
            self.visit(node.source)  # node.func is a method name, not a variable
            self.visit(node.args)
        elif isinstance(node, A.Field):
            if node.between_brackets or not isinstance(node.key, A.Name):
                self.visit(node.key)  # {[expr] = v}: the key is an expression; {key = v}: a field name
            self.visit(node.value)
        elif isinstance(node, (A.Goto, A.Label)):
            return  # label names are a separate namespace
        else:
            for child in _children(node):
                self.visit(child)


# ---------------------------------------------------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------------------------------------------------
# Node attributes that hold no child nodes, or that must not count when comparing trees (positions, comments, and the
# call style: f{...} and f({...}) are the same call, and the printer always writes the latter).
_IGNORED_ATTRIBUTES = {"comments", "line", "first_token", "last_token", "start_char", "stop_char", "style"}


def _children(node):
    """The direct child nodes of an AST node, in the order its attributes were set (mostly source order)."""
    for key, value in vars(node).items():
        if key.lstrip("_") in _IGNORED_ATTRIBUTES:
            continue
        if isinstance(value, A.Node):
            yield value
        elif isinstance(value, list):
            yield from (item for item in value if isinstance(item, A.Node))


def _walk(node):
    """All nodes of the tree in depth-first order (unlike luaparser's ast.walk, whose order is not specified)."""
    yield node
    for child in _children(node):
        yield from _walk(child)


def _dump(node):
    """A hashable, comparable representation of a tree, ignoring positions, comments and call style."""
    if isinstance(node, A.Node):
        return (type(node).__name__,) + tuple(
            (key, _dump(value)) for key, value in sorted(vars(node).items()) if key.lstrip("_") not in _IGNORED_ATTRIBUTES
        )
    if isinstance(node, list):
        return tuple(_dump(item) for item in node)
    return repr(node)


def _remove_comments(chunk: A.Chunk) -> None:
    for node in _walk(chunk):
        for value in vars(node).values():
            if isinstance(value, list):
                value[:] = [item for item in value if not isinstance(item, A.Comment)]
        if hasattr(node, "comments"):
            node.comments = []


class _Printer(printers.LuaOutputVisitor):
    """luaparser's printer with a more compact, conventional layout: tab indentation like in the EDOPro scripts, and
    tables on one line ({id, 1} instead of four lines) unless they contain multi-line values such as functions.

    Every child node is printed through do_visit, so intercepting blocks and tables there replaces the only two places
    where the base printer indents. Its comment handling is not needed, as comments are removed before printing.
    """

    def __init__(self):
        super().__init__(indent_size=1)

    def do_visit(self, node) -> str:
        if isinstance(node, A.Block):
            self._level += 1  # level 1 is the script's top level, which is not indented
            output = "\n".join(self.do_visit(statement) for statement in node.body)
            if self._level > 1:
                output = indent(output, "\t")
            self._level -= 1
            return output
        if isinstance(node, A.Table):
            fields = [self.do_visit(field) for field in node.fields]
            if any("\n" in field for field in fields):
                output = "{\n" + "".join(indent(field + ",\n", "\t") for field in fields) + "}"
            else:
                output = "{" + ", ".join(fields) + "}"
            return f"({output})" if node.wrapped else output
        return super().do_visit(node)


def _protect_double_minus(chunk: A.Chunk) -> None:
    """Parenthesize '-(-x)': luaparser would print it as '--x', which Lua reads as the start of a comment."""
    for node in _walk(chunk):
        if isinstance(node, A.UMinusOp) and isinstance(node.operand, A.UMinusOp):
            node.operand.wrapped = True


# ---------------------------------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------------------------------


def canonicalize(code: str, rename: bool = False) -> str:
    """The canonical form of a Lua script (see the module docstring). Raises CanonicalizationError if it fails."""
    try:
        chunk = ast.parse(code)
        _remove_comments(chunk)
        for node in list(_walk(chunk)):
            if isinstance(node, A.Block):
                _sort_setters(node)
        module = _module_name(chunk)
        function_order = list(_sort_functions(chunk, module)) if module else []
        if rename:
            _Renamer(module, function_order).visit(chunk)
        _protect_double_minus(chunk)
        printed = _Printer().do_visit(chunk).strip()
        if _dump(ast.parse(printed)) != _dump(chunk):
            raise CanonicalizationError("printed script does not parse back into the same AST")
    except CanonicalizationError:
        raise
    except (Exception, RecursionError) as error:  # luaparser raises its own, varying exception types
        raise CanonicalizationError(f"{type(error).__name__}: {error}") from None
    return printed
