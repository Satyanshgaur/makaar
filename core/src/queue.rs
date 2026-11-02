//! Resumable page queue backed by SQLite. A crash or closed laptop leaves `running` rows that
//! `recover` returns to `pending`, so batches resume instead of restarting.
use rusqlite::{params, Connection};

#[derive(Debug, thiserror::Error)]
pub enum QueueError {
    #[error("sqlite: {0}")]
    Sql(#[from] rusqlite::Error),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PageJob {
    pub id: i64,
    pub job: String,
    pub path: String,
}

pub struct Queue {
    db: Connection,
}

impl Queue {
    pub fn open(path: &str) -> Result<Self, QueueError> {
        let db = Connection::open(path)?;
        db.execute_batch(
            "CREATE TABLE IF NOT EXISTS pages (
               id INTEGER PRIMARY KEY, job TEXT NOT NULL, path TEXT NOT NULL,
               state TEXT NOT NULL DEFAULT 'pending' CHECK (state IN ('pending','running','done','failed')));
             CREATE INDEX IF NOT EXISTS pages_state ON pages(job, state);",
        )?;
        Ok(Self { db })
    }

    pub fn enqueue(&self, job: &str, path: &str) -> Result<(), QueueError> {
        self.db.execute("INSERT INTO pages (job, path) VALUES (?1, ?2)", params![job, path])?;
        Ok(())
    }

    /// Claims the oldest pending page of `job`, marking it running.
    pub fn claim(&self, job: &str) -> Result<Option<PageJob>, QueueError> {
        let row = self.db.query_row(
            "UPDATE pages SET state='running' WHERE id = (
               SELECT id FROM pages WHERE job=?1 AND state='pending' ORDER BY id LIMIT 1)
             RETURNING id, job, path",
            params![job],
            |r| Ok(PageJob { id: r.get(0)?, job: r.get(1)?, path: r.get(2)? }),
        );
        match row {
            Ok(p) => Ok(Some(p)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    pub fn finish(&self, id: i64, ok: bool) -> Result<(), QueueError> {
        self.db.execute("UPDATE pages SET state=?2 WHERE id=?1", params![id, if ok { "done" } else { "failed" }])?;
        Ok(())
    }

    /// Returns pages left `running` by a crash to `pending`; returns how many.
    pub fn recover(&self) -> Result<usize, QueueError> {
        Ok(self.db.execute("UPDATE pages SET state='pending' WHERE state='running'", [])?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claim_finish_recover() {
        let q = Queue::open(":memory:").unwrap();
        q.enqueue("j", "a.png").unwrap();
        q.enqueue("j", "b.png").unwrap();
        let a = q.claim("j").unwrap().unwrap();
        assert_eq!(a.path, "a.png");
        assert_eq!(q.recover().unwrap(), 1); // simulated crash mid-page
        let a2 = q.claim("j").unwrap().unwrap();
        assert_eq!(a2.path, "a.png");
        q.finish(a2.id, true).unwrap();
        assert_eq!(q.claim("j").unwrap().unwrap().path, "b.png");
        assert!(q.claim("other").unwrap().is_none());
    }
}
