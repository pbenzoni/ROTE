# File: text_extractor.py
import io
import json
import re
from collections import Counter
from io import BytesIO
from typing import Iterable, Tuple

import pandas as pd
from flask import Blueprint, render_template, request, send_file
#from sqlalchemy import create_engine

text_bp = Blueprint('text_extractor', __name__, template_folder='templates')

# ##### DATABASE FUNCTIONS #####
# def get_engine(server: str,
#                database: str,
#                username: str,
#                password: str,
#                driver: str = "ODBC Driver 17 for SQL Server"):
#     conn_str = (
#         f"mssql+pyodbc://{username}:{password}@{server}/{database}"
#         f"?driver={driver.replace(' ', '+')}"
#     )
#     return create_engine(conn_str)


# def fetch_raw_text(engine,
#                    table: str,
#                    column: str,
#                    like_phrase: str = '@',
#                    limit: int = None,
#                    order_by: str = None):
#     where = f"{column} LIKE '%{like_phrase}%'"
#     top = f"TOP {limit}" if limit else ""
#     order = f"ORDER BY {order_by} DESC" if order_by else ""
#     query = f"SELECT {top} {column} FROM {table} WHERE {where} {order};"
#     return pd.read_sql_query(query, engine)

##### TELEGRAM HELPERS #####
def _flatten_telegram_text(msg) -> Tuple[str, list, list, list]:
    """
    Returns: (full_text, hashtags[], mentions[], links[])
    Handles Telegram's list/dict mixed text structure.
    """
    txt = msg.get('text', '')
    hashtags, mentions, links = [], [], []

    def push_entity(part):
        t = part.get('type')
        # Prefer href for links if present, else text
        val = part.get('href') or part.get('text', '')
        if t == 'hashtag':
            hashtags.append(part.get('text', ''))
        elif t == 'mention':
            mentions.append(part.get('text', ''))
        elif t == 'link':
            links.append(val)

    if isinstance(txt, list):
        parts = []
        for part in txt:
            if isinstance(part, dict):
                push_entity(part)
                parts.append(part.get('text', ''))
            else:
                parts.append(str(part))
        return ''.join(parts), hashtags, mentions, links
    elif isinstance(txt, str):
        return txt, hashtags, mentions, links
    else:
        return str(txt), hashtags, mentions, links


def extract_from_telegram_json(data: dict, regex: str, filename: str) -> dict:
    """
    Returns per-file raw lists needed for aggregation.
    Keys: name, channel_id, filename, messages, matches, mentions, hashtags, links
    """
    channel_name = data.get('name') or data.get('title') or 'Unknown'
    channel_id = data.get('id') or data.get('channel_id') or 'unknown_id'

    all_matches, all_mentions, all_hashtags, all_links, all_forwarded = [], [], [], [], []
    msgs = data.get('messages', []) or []
    pattern = re.compile(regex)

    for msg in msgs:
        if msg.get('forwarded_from'):
            all_forwarded.append(msg['forwarded_from'])
            
        full_text, hashtags, mentions, links = _flatten_telegram_text(msg)
        all_hashtags.extend(hashtags)
        all_mentions.extend(mentions)
        all_links.extend(links)

        found = pattern.findall(full_text or "")
        for m in found:
            if isinstance(m, tuple):
                s = next((g for g in m if isinstance(g, str) and g), "")
                if not s:
                    s = "::".join([str(g) for g in m])
                all_matches.append(s)
            else:
                all_matches.append(m)

    return {
        'name': channel_name,
        'channel_id': channel_id,
        'filename': filename,
        'messages': len(msgs),
        'matches': all_matches,
        'mentions': all_mentions,
        'hashtags': all_hashtags,
        'links': all_links,
        'forwarded': all_forwarded
    }


def extract_pattern_from_text(text_or_list, regex: str) -> list:
    """
    For text/db modes. Accepts a str or iterable of strings and returns all matches.
    """
    pat = re.compile(regex)
    out = []
    if isinstance(text_or_list, str):
        items = [text_or_list]
    elif isinstance(text_or_list, Iterable):
        items = text_or_list
    else:
        items = [str(text_or_list)]

    for t in pd.Series(items).dropna():
        found = pat.findall(str(t))
        for m in found:
            if isinstance(m, tuple):
                s = next((g for g in m if isinstance(g, str) and g), "")
                if not s:
                    s = "::".join([str(g) for g in m])
                out.append(s)
            else:
                out.append(m)
    return out


def aggregate_counts(items: list) -> pd.DataFrame:
    counts = Counter(items)
    return (
        pd.DataFrame(counts.items(), columns=["value", "count"])
          .sort_values(by="count", ascending=False)
          .reset_index(drop=True)
    )

##### EXCEL UTIL #####
def _sheet(df: pd.DataFrame, cols: list, sort_cols: list) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=cols)
    df = df[cols]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[True if c != 'count' else False for c in sort_cols])
    return df.reset_index(drop=True)

@text_bp.route('/', methods=['GET', 'POST'])
def extract():
    results = {}
    if request.method == 'POST':
        mode = request.form.get('mode')
        regex = request.form.get('regex') or r"@([A-Za-z0-9_]+)"
        filename = 'results.xlsx'
        sheets = {}

        if mode == 'text':
            text = request.form.get('text', '')
            matches = extract_pattern_from_text(text, regex)
            df = aggregate_counts(matches).rename(columns={'value': 'pattern'})
            sheets['results'] = df
            filename = 'text_extract.xlsx'

        elif mode == 'json':
            files = request.files.getlist('json_files')
            if not files:
                single = request.files.get('json_file')  # backward-compat
                if single:
                    files = [single]

            # Accumulators for long-form tables
            matches_rows, mentions_rows, hashtags_rows, links_rows, forwarded_rows = [], [], [], [], []
            summary_rows = []

            for f in files:
                try:
                    data = json.load(f)
                except Exception:
                    continue  # skip unreadable files

                ex = extract_from_telegram_json(data, regex, f.filename or 'file.json')

                # Summary
                summary_rows.append({
                    'name': ex['name'],
                    'channel_id': ex['channel_id'],
                    'filename': ex['filename'],
                    'messages': ex['messages'],
                    'matches_total': len(ex['matches']),
                    'mentions_total': len(ex['mentions']),
                    'hashtags_total': len(ex['hashtags']),
                    'links_total': len(ex['links']),
                    'forwarded_total': len(ex['forwarded'])
                })

                # Per-file tallies → long-form rows
                for val, cnt in Counter(ex['matches']).items():
                    matches_rows.append({
                        'name': ex['name'],
                        'channel_id': ex['channel_id'],
                        'filename': ex['filename'],
                        'match': val,
                        'count': cnt
                    })
                for val, cnt in Counter(ex['mentions']).items():
                    mentions_rows.append({
                        'name': ex['name'],
                        'channel_id': ex['channel_id'],
                        'filename': ex['filename'],
                        'mention': val,
                        'count': cnt
                    })
                for val, cnt in Counter(ex['hashtags']).items():
                    hashtags_rows.append({
                        'name': ex['name'],
                        'channel_id': ex['channel_id'],
                        'filename': ex['filename'],
                        'hashtag': val,
                        'count': cnt
                    })
                for val, cnt in Counter(ex['links']).items():
                    links_rows.append({
                        'name': ex['name'],
                        'channel_id': ex['channel_id'],
                        'filename': ex['filename'],
                        'link': val,
                        'count': cnt
                    })
                for val, cnt in Counter(ex['forwarded']).items():
                    forwarded_rows.append({
                        'name': ex['name'],
                        'channel_id': ex['channel_id'],
                        'filename': ex['filename'],
                        'forwarded': val,
                        'count': cnt
                    })
                

            # Build sheets
            summary_df = pd.DataFrame(summary_rows).sort_values(
                ['name', 'filename']
            ).reset_index(drop=True)

            matches_df  = pd.DataFrame(matches_rows)
            mentions_df = pd.DataFrame(mentions_rows)
            hashtags_df = pd.DataFrame(hashtags_rows)
            links_df    = pd.DataFrame(links_rows)

            sheets['SUMMARY']  = _sheet(summary_df,
                                        ['name', 'channel_id', 'filename', 'messages', 'matches_total', 'mentions_total', 'hashtags_total', 'links_total'],
                                        ['name', 'filename'])
            sheets['MATCHES']  = _sheet(matches_df,
                                        ['name', 'channel_id', 'filename', 'match', 'count'],
                                        ['name', 'filename', 'count'])
            sheets['MENTIONS'] = _sheet(mentions_df,
                                        ['name', 'channel_id', 'filename', 'mention', 'count'],
                                        ['name', 'filename', 'count'])
            sheets['HASHTAGS'] = _sheet(hashtags_df,
                                        ['name', 'channel_id', 'filename', 'hashtag', 'count'],
                                        ['name', 'filename', 'count'])
            sheets['LINKS']    = _sheet(links_df,
                                        ['name', 'channel_id', 'filename', 'link', 'count'],
                                        ['name', 'filename', 'count'])
            sheets['FORWARDED'] = _sheet(pd.DataFrame(forwarded_rows),
                                         ['name', 'channel_id', 'filename', 'forwarded', 'count'],
                                         ['name', 'filename', 'count'])
                                        

            filename = 'json_extract_multi.xlsx'

        # elif mode == 'db':
        #     server = request.form.get('server')
        #     database = request.form.get('database')
        #     username = request.form.get('username')
        #     password = request.form.get('password')
        #     table = request.form.get('table')
        #     column = request.form.get('column') or 'raw_text'
        #     limit = request.form.get('limit') or None
        #     order_by = request.form.get('order_by') or None
        #     engine = get_engine(server, database, username, password)
        #     df_text = fetch_raw_text(
        #         engine, table, column,
        #         limit=int(limit) if limit else None,
        #         order_by=order_by
        #     )
        #     matches = extract_pattern_from_text(df_text[column], regex)
        #     df = (
        #         pd.DataFrame(Counter(matches).items(), columns=['pattern', 'count'])
        #           .sort_values('count', ascending=False)
        #           .reset_index(drop=True)
        #     )
        #     sheets['results'] = df
        #     filename = 'db_extract.xlsx'

        # Create Excel file in memory
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            for sheet_name, data in sheets.items():
                (data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)).to_excel(
                    writer, sheet_name=sheet_name, index=False
                )
        output.seek(0)

        # Return download response
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=filename
        )

    # GET: render form
    return render_template('text_extractor.html', results=results)
