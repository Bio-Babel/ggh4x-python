"""AST detectors for ggh4x anti-patterns."""

from __future__ import annotations

import ast

from biobabel.detector_api import DetectorMatch

_FACET_NAMES = {
    "facet_grid2",
    "facet_wrap2",
    "facet_nested",
    "facet_nested_wrap",
    "facet_manual",
}

_DISCRETE_DISTRIBUTIONS = {"nbinom", "binom"}


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
    return None


def _flatten_add_chain(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _flatten_add_chain(node.left) + _flatten_add_chain(node.right)
    return [node]


def force_panelsizes_before_facet(tree: ast.AST, args: dict) -> list[DetectorMatch]:
    """Flag `force_panelsizes(...)` appearing before a facet_* call in the same `+` chain."""
    child_ids = set()
    add_roots = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            add_roots.append(node)
            for side in (node.left, node.right):
                if isinstance(side, ast.BinOp) and isinstance(side.op, ast.Add):
                    child_ids.add(id(side))

    matches: list[DetectorMatch] = []
    for root in add_roots:
        if id(root) in child_ids:
            continue
        operands = _flatten_add_chain(root)
        names = [_call_name(operand) for operand in operands]
        facet_idx = next((i for i, name in enumerate(names) if name in _FACET_NAMES), None)
        if facet_idx is None:
            continue
        for i, name in enumerate(names):
            if name == "force_panelsizes" and i < facet_idx:
                matches.append(
                    DetectorMatch(
                        line=getattr(operands[i], "lineno", root.lineno),
                        detail={"call": "force_panelsizes", "facet_index": facet_idx, "call_index": i},
                    )
                )
    return matches


def text_aimed_nudge_position_conflict(tree: ast.AST, args: dict) -> list[DetectorMatch]:
    """Flag geom_text_aimed(...) calls combining a nonzero nudge_x/nudge_y with a non-identity position."""
    matches: list[DetectorMatch] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _call_name(node) == "geom_text_aimed"):
            continue
        has_nudge = False
        has_position_conflict = False
        for kw in node.keywords:
            if kw.arg in ("nudge_x", "nudge_y"):
                value = kw.value
                if isinstance(value, ast.Constant) and isinstance(value.value, (int, float)):
                    if value.value != 0:
                        has_nudge = True
                else:
                    has_nudge = True
            elif kw.arg == "position":
                value = kw.value
                if isinstance(value, ast.Constant):
                    if value.value != "identity":
                        has_position_conflict = True
                else:
                    has_position_conflict = True
        if has_nudge and has_position_conflict:
            matches.append(DetectorMatch(line=node.lineno, detail={"call": "geom_text_aimed"}))
    return matches


def margin_length_not_unit(tree: ast.AST, args: dict) -> list[DetectorMatch]:
    """Flag geom_rectmargin/geom_tilemargin(length=<bare number>) instead of a grid_py.Unit."""
    matches: list[DetectorMatch] = []
    for node in ast.walk(tree):
        name = _call_name(node)
        if not (isinstance(node, ast.Call) and name in ("geom_rectmargin", "geom_tilemargin")):
            continue
        for kw in node.keywords:
            if kw.arg == "length" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, (int, float)):
                matches.append(DetectorMatch(line=node.lineno, detail={"call": name}))
    return matches


def theodensity_discrete_default_line(tree: ast.AST, args: dict) -> list[DetectorMatch]:
    """Flag stat_theodensity(distri=<discrete>) without geom='point'."""
    matches: list[DetectorMatch] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _call_name(node) == "stat_theodensity"):
            continue
        distri_value = None
        geom_value = None
        for kw in node.keywords:
            if kw.arg == "distri" and isinstance(kw.value, ast.Constant):
                distri_value = kw.value.value
            if kw.arg == "geom" and isinstance(kw.value, ast.Constant):
                geom_value = kw.value.value
        if distri_value in _DISCRETE_DISTRIBUTIONS and geom_value != "point":
            matches.append(DetectorMatch(line=node.lineno, detail={"distri": distri_value}))
    return matches
