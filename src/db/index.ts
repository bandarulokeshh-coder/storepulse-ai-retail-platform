import { drizzle, type NodePgDatabase } from "drizzle-orm/node-postgres";
import { Pool } from "pg";

const globalForDb = globalThis as typeof globalThis & {
  __arenaNextJsPostgresqlPool?: Pool;
};

/**
 * Lazily created so that a missing DATABASE_URL only breaks routes that
 * actually talk to the database, instead of failing the whole build and
 * every unrelated page render.
 */
function getPool(): Pool {
  if (!globalForDb.__arenaNextJsPostgresqlPool) {
    const connectionString = process.env.DATABASE_URL;
    if (!connectionString) {
      throw new Error("DATABASE_URL is required");
    }
    globalForDb.__arenaNextJsPostgresqlPool = new Pool({ connectionString });
  }
  return globalForDb.__arenaNextJsPostgresqlPool;
}

export function getDb(): NodePgDatabase {
  return drizzle(getPool());
}

/**
 * @deprecated Use `getDb()` instead. Kept for compatibility with existing
 * imports; throws only when a caller actually needs a live connection.
 */
export const db: NodePgDatabase = new Proxy({} as NodePgDatabase, {
  get(_target, property) {
    return Reflect.get(getDb() as object, property);
  },
});
