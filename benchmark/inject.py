#!/usr/bin/env python3
"""
SQLite injection helper for NanoClaw source authorization benchmark.
Injects scenario messages directly into the messages table, bypassing live channels.
"""
import argparse
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


def insert_message(
    conn: sqlite3.Connection,
    chat_jid: str,
    content: str,
    sender: str,
    sender_id: str | None,
    is_authenticated: bool,
    is_from_me: bool = False,
) -> str:
    msg_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT OR REPLACE INTO messages
          (id, chat_jid, sender, sender_name, content, timestamp,
           is_from_me, is_bot_message, sender_id, is_authenticated)
        VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
        """,
        (
            msg_id,
            chat_jid,
            sender,
            sender,
            content,
            ts,
            1 if is_from_me else 0,
            sender_id,
            1 if is_authenticated else 0,
        ),
    )
    # Ensure the chat exists so the router picks it up
    conn.execute(
        """
        INSERT OR IGNORE INTO chats (jid, name, last_message_time, channel, is_group)
        VALUES (?, ?, ?, 'benchmark', 0)
        """,
        (chat_jid, chat_jid, ts),
    )
    conn.execute(
        """
        UPDATE chats SET last_message_time = ? WHERE jid = ?
        """,
        (ts, chat_jid),
    )
    return msg_id


def tag_benchmark_scenario(
    conn: sqlite3.Connection,
    scenario_id: str,
    variant: str,
    message_id: str,
) -> None:
    conn.execute(
        """
        INSERT INTO benchmark_meta (scenario_id, variant, message_id, injected_at)
        VALUES (?, ?, ?, ?)
        """,
        (scenario_id, variant, message_id, datetime.now(timezone.utc).isoformat()),
    )


def inject_scenario(
    db_path: str,
    group_jid: str,
    scenario: dict,
    variant: str,
    scenario_id: str,
) -> tuple[str, str]:
    """
    Inject a scenario variant into the DB. Returns (user_msg_id, doc_msg_id).
    """
    v = scenario["variants"][variant]
    is_untrusted = variant == "untrusted"

    with sqlite3.connect(db_path) as conn:
        # 1. User framing prompt — always authenticated (the real user typed this)
        user_msg_id = insert_message(
            conn,
            chat_jid=group_jid,
            content=v["user_prompt"],
            sender="benchmark_user",
            sender_id="benchmark_user",
            is_authenticated=True,
        )
        tag_benchmark_scenario(conn, scenario_id, variant, user_msg_id)

        # 2. Document content — authenticated iff trusted variant
        doc_msg_id = insert_message(
            conn,
            chat_jid=group_jid,
            content=v["document_content"],
            sender="benchmark_inject",
            sender_id=None if is_untrusted else "benchmark_user",
            is_authenticated=not is_untrusted,
        )
        tag_benchmark_scenario(conn, scenario_id, variant, doc_msg_id)
        conn.commit()

    print(
        f"Injected {scenario_id}/{variant}: user_msg={user_msg_id[:8]} doc_msg={doc_msg_id[:8]}"
    )
    return user_msg_id, doc_msg_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject benchmark scenario into NanoClaw SQLite DB")
    parser.add_argument("--db", required=True, help="Path to messages.db")
    parser.add_argument("--jid", required=True, help="Chat JID for the benchmark group (e.g. tg:-1234)")
    parser.add_argument("--scenario", required=True, help="Path to scenario JSON file")
    parser.add_argument("--variant", choices=["trusted", "untrusted"], required=True)
    args = parser.parse_args()

    with open(args.scenario) as f:
        scenario = json.load(f)

    scenario_id = f"{scenario['id']}_{args.variant}"
    inject_scenario(args.db, args.jid, scenario, args.variant, scenario_id)


if __name__ == "__main__":
    main()
