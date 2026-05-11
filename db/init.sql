CREATE EXTENSION IF NOT EXISTS postgis;

-- ─── PLACES ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS places (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    source_url      TEXT UNIQUE NOT NULL,
    place_type      TEXT NOT NULL DEFAULT 'cafe',   -- cafe | restaurant | bar | hotel | ...
    location        GEOMETRY(Point, 4326) NOT NULL, -- required for map display
    lat             FLOAT GENERATED ALWAYS AS (ST_Y(location)) STORED,
    lng             FLOAT GENERATED ALWAYS AS (ST_X(location)) STORED,
    total_reviews   INT         DEFAULT 0,
    scraped_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Harita sorguları için kritik index'ler
CREATE INDEX IF NOT EXISTS places_location_gist ON places USING GIST (location);
CREATE INDEX IF NOT EXISTS places_type_idx      ON places (place_type);
-- Bounding box + type kombinasyonu (haritada filtreli arama)
CREATE INDEX IF NOT EXISTS places_type_location ON places USING GIST (location) WHERE place_type IS NOT NULL;

-- ─── REVIEWS ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS reviews (
    id          SERIAL PRIMARY KEY,
    place_id    INT  NOT NULL REFERENCES places(id) ON DELETE CASCADE,
    content     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS reviews_place_idx ON reviews (place_id);

-- ─── ANALYSIS ────────────────────────────────────────────────────────────────
-- Common scores for all place types.
-- Type-specific scores go into scores_extra JSONB
-- (e.g. cafe → coffee_quality, restaurant → food_quality)
CREATE TABLE IF NOT EXISTS place_analysis (
    id              SERIAL PRIMARY KEY,
    place_id        INT UNIQUE NOT NULL REFERENCES places(id) ON DELETE CASCADE,
    analyzed_at     TIMESTAMPTZ,

    overall_score       FLOAT,
    summary             TEXT,
    ideal_for           TEXT,
    price_level         TEXT,   -- budget | mid | mid-high | expensive | unknown

    -- Scores common to all types
    score_service           FLOAT,
    score_price_performance  FLOAT,
    score_atmosphere        FLOAT,

    -- Type-specific scores
    scores_extra    JSONB   DEFAULT '{}',

    highlights      TEXT[],
    downsides       TEXT[],
    popular_items   TEXT[],
    tags            TEXT[]
);

CREATE INDEX IF NOT EXISTS analysis_overall_idx  ON place_analysis (overall_score DESC);
CREATE INDEX IF NOT EXISTS analysis_tags_gin     ON place_analysis USING GIN (tags);
CREATE INDEX IF NOT EXISTS analysis_extra_gin    ON place_analysis USING GIN (scores_extra);

-- ─── MAP VIEW ────────────────────────────────────────────────────────────────
-- Ready-to-use view for map markers with all needed data in one query
CREATE OR REPLACE VIEW map_places AS
    SELECT
        p.id,
        p.name,
        p.place_type,
        p.source_url,
        p.lat,
        p.lng,
        ST_AsGeoJSON(p.location)::json AS geojson,
        p.total_reviews,
        a.overall_score,
        a.price_level,
        a.tags,
        a.ideal_for,
        a.summary
    FROM places p
    LEFT JOIN place_analysis a ON a.place_id = p.id;