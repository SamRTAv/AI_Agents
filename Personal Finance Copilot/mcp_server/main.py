"""Expense tracker MCP server.

Runs as its own process. It knows nothing about LangGraph, LangChain or the
FastAPI app that calls it — that independence is the whole point of putting
these tools behind MCP: the same server also works from Claude Desktop or any
other MCP client.

Run standalone:   python mcp_server/main.py
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import date as _date
from typing import Any

from fastmcp import FastMCP

DB_PATH = os.path.join(os.path.dirname(__file__), "expenses.db")
CATEGORIES_PATH = os.path.join(os.path.dirname(__file__), "categories.json")

mcp = FastMCP("ExpenseTracker")


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

# The seed data was written with free-text labels ("Transportation", "transport",
# "Groceries") that do not match categories.json. That splits a single real
# category across several GROUP BY buckets, so summaries under-report. Map the
# known legacy labels onto the taxonomy once, at startup.
_LEGACY_CATEGORY_MAP = {
    "transportation": ("transport", ""),
    "transport": ("transport", ""),
    "groceries": ("food", "groceries"),
    "dining out": ("food", "dining_out"),
    "healthcare": ("health", ""),
    "utilities": ("utilities", ""),
    "entertainment": ("entertainment", ""),
    "shopping": ("shopping", ""),
    "education": ("education", ""),
    "food": ("food", ""),
}


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS expenses(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                amount REAL NOT NULL,
                category TEXT NOT NULL,
                subcategory TEXT DEFAULT '',
                note TEXT DEFAULT ''
            )
            """
        )
        # Range queries filter on date, summaries group on category.
        c.execute("CREATE INDEX IF NOT EXISTS idx_expenses_date ON expenses(date)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_expenses_cat ON expenses(category)")

        for row in c.execute("SELECT id, category, subcategory FROM expenses").fetchall():
            mapped = _LEGACY_CATEGORY_MAP.get(row["category"].strip().lower())
            if not mapped:
                continue
            cat, sub = mapped
            sub = row["subcategory"] or sub
            if cat != row["category"] or sub != (row["subcategory"] or ""):
                c.execute(
                    "UPDATE expenses SET category = ?, subcategory = ? WHERE id = ?",
                    (cat, sub, row["id"]),
                )


def _load_categories() -> dict[str, list[str]]:
    with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_date(value: str, field: str) -> str:
    """Dates are compared as strings in SQL, so the format has to be exact."""
    try:
        return _date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise ValueError(
            f"{field} must be in YYYY-MM-DD format (got {value!r})"
        ) from exc


def _normalise_category(category: str, subcategory: str) -> tuple[str, str]:
    """Snap free text onto the taxonomy so summaries aggregate correctly."""
    taxonomy = _load_categories()
    cat = category.strip().lower().replace(" ", "_")
    sub = (subcategory or "").strip().lower().replace(" ", "_")

    if cat not in taxonomy:
        legacy = _LEGACY_CATEGORY_MAP.get(category.strip().lower())
        if legacy:
            cat, mapped_sub = legacy
            sub = sub or mapped_sub
        else:
            raise ValueError(
                f"Unknown category {category!r}. "
                f"Valid categories: {', '.join(sorted(taxonomy))}"
            )

    if sub and sub not in taxonomy[cat]:
        sub = "other" if "other" in taxonomy[cat] else ""

    return cat, sub


init_db()


# ---------------------------------------------------------------------------
# Tools
#
# These docstrings are not decoration: the LLM reads them (plus the type hints)
# to decide which tool to call and with what arguments. Vague wording here shows
# up as wrong tool selection at runtime.
# ---------------------------------------------------------------------------


@mcp.tool()
def add_expense(
    date: str,
    amount: float,
    category: str,
    subcategory: str = "",
    note: str = "",
) -> dict[str, Any]:
    """Record a new expense in the user's expense database.

    Use this when the user reports money they have spent. This writes to the
    database, so only call it when the user clearly intends to record a spend.

    Args:
        date: Date of the expense, YYYY-MM-DD (e.g. "2025-09-21").
        amount: Amount spent, in rupees. Positive number.
        category: Top-level category, e.g. "food", "transport", "utilities".
            Call list_categories to see valid values.
        subcategory: Optional finer label, e.g. "dining_out", "fuel".
        note: Optional free-text description.
    """
    date = _validate_date(date, "date")
    if amount <= 0:
        raise ValueError(f"amount must be positive (got {amount})")
    category, subcategory = _normalise_category(category, subcategory)

    with _connect() as c:
        cur = c.execute(
            "INSERT INTO expenses(date, amount, category, subcategory, note) "
            "VALUES (?,?,?,?,?)",
            (date, amount, category, subcategory, note),
        )
        return {
            "status": "ok",
            "id": cur.lastrowid,
            "date": date,
            "amount": amount,
            "category": category,
            "subcategory": subcategory,
        }


@mcp.tool()
def list_expenses(start_date: str, end_date: str) -> dict[str, Any]:
    """List every individual expense between two dates, inclusive.

    Use this when the user wants to see specific transactions. If they only
    want totals or a breakdown by category, use summarize instead — it returns
    far less data.

    Args:
        start_date: Start of range, YYYY-MM-DD, inclusive.
        end_date: End of range, YYYY-MM-DD, inclusive.
    """
    start_date = _validate_date(start_date, "start_date")
    end_date = _validate_date(end_date, "end_date")

    with _connect() as c:
        rows = c.execute(
            """
            SELECT id, date, amount, category, subcategory, note
            FROM expenses
            WHERE date BETWEEN ? AND ?
            ORDER BY date ASC, id ASC
            """,
            (start_date, end_date),
        ).fetchall()

    expenses = [dict(r) for r in rows]
    return {
        "start_date": start_date,
        "end_date": end_date,
        "count": len(expenses),
        "total_amount": round(sum(e["amount"] for e in expenses), 2),
        "expenses": expenses,
    }


@mcp.tool()
def summarize(
    start_date: str,
    end_date: str,
    category: str | None = None,
) -> dict[str, Any]:
    """Total the user's spending between two dates, broken down by category.

    This is the right tool for questions like "how much did I spend last
    month", "where does my money go", or any question that needs the user's
    actual spending totals. Returns the overall total plus a per-category
    breakdown with each category's share of the total.

    Args:
        start_date: Start of range, YYYY-MM-DD, inclusive.
        end_date: End of range, YYYY-MM-DD, inclusive.
        category: Optional. Restrict the summary to a single category.
    """
    start_date = _validate_date(start_date, "start_date")
    end_date = _validate_date(end_date, "end_date")

    query = """
        SELECT category, COUNT(*) AS entries, SUM(amount) AS total_amount
        FROM expenses
        WHERE date BETWEEN ? AND ?
    """
    params: list[Any] = [start_date, end_date]

    if category:
        cat, _ = _normalise_category(category, "")
        query += " AND category = ?"
        params.append(cat)

    query += " GROUP BY category ORDER BY total_amount DESC"

    with _connect() as c:
        rows = [dict(r) for r in c.execute(query, params).fetchall()]

    grand_total = round(sum(r["total_amount"] for r in rows), 2)
    for r in rows:
        r["total_amount"] = round(r["total_amount"], 2)
        r["share_percent"] = (
            round(r["total_amount"] / grand_total * 100, 1) if grand_total else 0.0
        )

    return {
        "start_date": start_date,
        "end_date": end_date,
        "total_amount": grand_total,
        "by_category": rows,
    }


@mcp.tool()
def list_categories() -> dict[str, list[str]]:
    """Return the valid expense categories and their subcategories.

    Call this before add_expense if you are unsure which category applies to
    something the user described.
    """
    return _load_categories()


# ---------------------------------------------------------------------------
# Resource
#
# Same taxonomy, exposed as a *resource* rather than a tool. Resources are
# read-only context a client can pull; tools are actions a model can invoke.
# list_categories exists as well because MCP resources are not automatically
# offered to the model as callable tools.
# ---------------------------------------------------------------------------


@mcp.resource("expense://categories", mime_type="application/json")
def categories() -> str:
    # Read fresh each time so the file can be edited without a restart.
    with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    mcp.run()
