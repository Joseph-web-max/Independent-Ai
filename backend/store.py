"""Simple user-controlled SQLite conversation memory; no network access."""
from __future__ import annotations
import re,sqlite3,time
from pathlib import Path

class ConversationMemory:
    def __init__(self,path:Path|str):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        with self._connect() as con:
            con.execute('''CREATE TABLE IF NOT EXISTS turns (id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL NOT NULL, user_text TEXT NOT NULL, assistant_text TEXT NOT NULL)''')
    def _connect(self):
        con=sqlite3.connect(self.path,timeout=10);con.execute('PRAGMA journal_mode=WAL');return con
    def add(self,user_text:str,assistant_text:str):
        with self._connect() as con:
            con.execute('INSERT INTO turns(created,user_text,assistant_text) VALUES(?,?,?)',(time.time(),user_text[:2000],assistant_text[:3000]))
    def search(self,query:str,limit:int=3):
        terms=set(re.findall(r"[a-zA-Z0-9_]{3,}",query.lower()))
        with self._connect() as con:
            rows=con.execute('SELECT id,created,user_text,assistant_text FROM turns ORDER BY id DESC LIMIT 200').fetchall()
        scored=[]
        for row in rows:
            blob=(row[2]+' '+row[3]).lower(); words=set(re.findall(r"[a-zA-Z0-9_]{3,}",blob))
            overlap=len(terms & words); recency=1/(1+max(0,time.time()-row[1])/86400)
            score=overlap+0.15*recency
            if score>0.15: scored.append((score,row))
        scored.sort(key=lambda x:x[0],reverse=True)
        return [{'id':r[0],'user':r[2],'assistant':r[3]} for _,r in scored[:limit]]
    def recent(self,limit:int=3):
        with self._connect() as con:
            rows=con.execute('SELECT id,user_text,assistant_text FROM turns ORDER BY id DESC LIMIT ?',(limit,)).fetchall()
        return [{'id':r[0],'user':r[1],'assistant':r[2]} for r in reversed(rows)]
    def clear(self):
        with self._connect() as con: con.execute('DELETE FROM turns')
    def count(self):
        with self._connect() as con:return int(con.execute('SELECT COUNT(*) FROM turns').fetchone()[0])
