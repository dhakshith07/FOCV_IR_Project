-- Context-Aware Workspace Organizer
-- SQLite Schema v1 — Initial Migration
-- This schema is managed by SQLAlchemy ORM (database/models.py)
-- This file is for reference and manual inspection only.

CREATE TABLE IF NOT EXISTS objects (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id    INTEGER NOT NULL,
    label       TEXT,
    class_name  TEXT,
    first_seen  DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_seen   DATETIME,
    is_active   INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS sessions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time      DATETIME DEFAULT CURRENT_TIMESTAMP,
    end_time        DATETIME,
    session_type    TEXT DEFAULT 'unknown',
    total_objects   INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS usage_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    object_id   INTEGER REFERENCES objects(id),
    session_id  INTEGER REFERENCES sessions(id),
    event_type  TEXT,
    timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP,
    duration_s  REAL
);

CREATE TABLE IF NOT EXISTS workspace_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER REFERENCES sessions(id),
    timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP,
    layout_json TEXT,
    source      TEXT
);

CREATE TABLE IF NOT EXISTS robot_actions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER REFERENCES sessions(id),
    object_id       INTEGER REFERENCES objects(id),
    action_type     TEXT,
    start_pos_json  TEXT,
    end_pos_json    TEXT,
    joint_path_json TEXT,
    duration_ms     INTEGER,
    timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dqn_checkpoints (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    episode     INTEGER,
    reward      REAL,
    epsilon     REAL,
    saved_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    model_path  TEXT
);

CREATE TABLE IF NOT EXISTS priority_scores (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    object_id       INTEGER REFERENCES objects(id),
    session_id      INTEGER REFERENCES sessions(id),
    score           REAL,
    zone_affinity   TEXT,
    computed_at     DATETIME DEFAULT CURRENT_TIMESTAMP
);
