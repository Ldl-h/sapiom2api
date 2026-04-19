import { Router } from "express";
import { db } from "@workspace/db";
import { apiKeysTable } from "@workspace/db";
import { eq, sql, isNotNull } from "drizzle-orm";
import { z } from "zod";

const router = Router();

router.get("/keys/stats", async (req, res) => {
  try {
    const keys = await db.select().from(apiKeysTable);
    const active = keys.filter((k) => k.isActive).length;
    const inactive = keys.length - active;
    const invalid = keys.filter((k) => k.validationStatus === "invalid").length;
    const valid = keys.filter((k) => k.validationStatus === "valid").length;
    const noBalance = keys.filter((k) => k.validationStatus === "no_balance").length;

    const providerMap: Record<string, number> = {};
    for (const k of keys) {
      const p = k.provider ?? "Unknown";
      providerMap[p] = (providerMap[p] ?? 0) + 1;
    }
    const providers = Object.entries(providerMap).map(([provider, count]) => ({ provider, count }));

    res.json({ total: keys.length, active, inactive, invalid, valid, noBalance, providers });
  } catch (e) {
    req.log.error(e);
    res.status(500).json({ error: "Internal server error" });
  }
});

router.get("/keys", async (req, res) => {
  try {
    const keys = await db.select().from(apiKeysTable).orderBy(sql`${apiKeysTable.createdAt} DESC`);
    res.json(
      keys.map((k) => ({
        ...k,
        createdAt: k.createdAt.toISOString(),
        updatedAt: k.updatedAt.toISOString(),
        validatedAt: k.validatedAt?.toISOString() ?? null,
      }))
    );
  } catch (e) {
    req.log.error(e);
    res.status(500).json({ error: "Internal server error" });
  }
});

const createKeySchema = z.object({
  name: z.string().min(1),
  key: z.string().min(1),
  provider: z.string().optional(),
  note: z.string().optional(),
});

router.post("/keys", async (req, res) => {
  try {
    const parsed = createKeySchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: "Invalid request body" });
      return;
    }
    const { name, key, provider, note } = parsed.data;
    const [created] = await db
      .insert(apiKeysTable)
      .values({ name, key, provider: provider ?? null, note: note ?? null })
      .returning();
    res.status(201).json({
      ...created,
      createdAt: created.createdAt.toISOString(),
      updatedAt: created.updatedAt.toISOString(),
      validatedAt: null,
    });
  } catch (e) {
    req.log.error(e);
    res.status(500).json({ error: "Internal server error" });
  }
});

const importKeysSchema = z.object({
  keys: z.array(
    z.object({
      name: z.string().optional(),
      key: z.string().min(1),
      provider: z.string().optional(),
      note: z.string().optional(),
    })
  ),
});

router.delete("/keys/purge-invalid", async (req, res) => {
  try {
    const deleted = await db
      .delete(apiKeysTable)
      .where(eq(apiKeysTable.validationStatus, "invalid"))
      .returning({ id: apiKeysTable.id });
    res.json({ deleted: deleted.length });
  } catch (e) {
    req.log.error(e);
    res.status(500).json({ error: "Internal server error" });
  }
});

router.post("/keys/import", async (req, res) => {
  try {
    const parsed = importKeysSchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: "Invalid request body" });
      return;
    }

    const existingKeys = await db.select({ key: apiKeysTable.key }).from(apiKeysTable);
    const existingSet = new Set(existingKeys.map((k) => k.key));

    const toInsert = parsed.data.keys.filter((k) => !existingSet.has(k.key));
    const skipped = parsed.data.keys.length - toInsert.length;

    if (toInsert.length === 0) {
      res.status(201).json({ imported: 0, skipped, keys: [] });
      return;
    }

    const created = await db
      .insert(apiKeysTable)
      .values(
        toInsert.map((k, i) => ({
          name: k.name ?? `Imported Key ${i + 1}`,
          key: k.key,
          provider: k.provider ?? null,
          note: k.note ?? null,
        }))
      )
      .returning();

    res.status(201).json({
      imported: created.length,
      skipped,
      keys: created.map((k) => ({
        ...k,
        createdAt: k.createdAt.toISOString(),
        updatedAt: k.updatedAt.toISOString(),
        validatedAt: null,
      })),
    });
  } catch (e) {
    req.log.error(e);
    res.status(500).json({ error: "Internal server error" });
  }
});

router.get("/keys/:id", async (req, res) => {
  try {
    const id = parseInt(req.params.id);
    if (isNaN(id)) {
      res.status(400).json({ error: "Invalid id" });
      return;
    }
    const [key] = await db.select().from(apiKeysTable).where(eq(apiKeysTable.id, id));
    if (!key) {
      res.status(404).json({ error: "Key not found" });
      return;
    }
    res.json({
      ...key,
      createdAt: key.createdAt.toISOString(),
      updatedAt: key.updatedAt.toISOString(),
      validatedAt: key.validatedAt?.toISOString() ?? null,
    });
  } catch (e) {
    req.log.error(e);
    res.status(500).json({ error: "Internal server error" });
  }
});

const updateKeySchema = z.object({
  name: z.string().optional(),
  key: z.string().optional(),
  provider: z.string().optional(),
  note: z.string().optional(),
  isActive: z.boolean().optional(),
});

router.patch("/keys/:id", async (req, res) => {
  try {
    const id = parseInt(req.params.id);
    if (isNaN(id)) {
      res.status(400).json({ error: "Invalid id" });
      return;
    }
    const parsed = updateKeySchema.safeParse(req.body);
    if (!parsed.success) {
      res.status(400).json({ error: "Invalid request body" });
      return;
    }
    const [updated] = await db
      .update(apiKeysTable)
      .set(parsed.data)
      .where(eq(apiKeysTable.id, id))
      .returning();
    if (!updated) {
      res.status(404).json({ error: "Key not found" });
      return;
    }
    res.json({
      ...updated,
      createdAt: updated.createdAt.toISOString(),
      updatedAt: updated.updatedAt.toISOString(),
      validatedAt: updated.validatedAt?.toISOString() ?? null,
    });
  } catch (e) {
    req.log.error(e);
    res.status(500).json({ error: "Internal server error" });
  }
});

router.delete("/keys/:id", async (req, res) => {
  try {
    const id = parseInt(req.params.id);
    if (isNaN(id)) {
      res.status(400).json({ error: "Invalid id" });
      return;
    }
    const [deleted] = await db.delete(apiKeysTable).where(eq(apiKeysTable.id, id)).returning();
    if (!deleted) {
      res.status(404).json({ error: "Key not found" });
      return;
    }
    res.status(204).send();
  } catch (e) {
    req.log.error(e);
    res.status(500).json({ error: "Internal server error" });
  }
});

async function checkKeyValidity(key: string): Promise<{ status: string; message: string; httpStatus: number }> {
  const resp = await fetch("https://api.sapiom.ai/v1/transactions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": key,
    },
    body: JSON.stringify({ metadata: { test: true } }),
    signal: AbortSignal.timeout(8000),
  }).catch((e: Error) => ({ status: 0, _err: e.message }));

  const apiStatus = "status" in resp ? (resp as Response).status : 0;
  let apiBody = "";
  if ("json" in resp) {
    try {
      const j = await (resp as Response).json() as Record<string, unknown>;
      apiBody = String(j.message || j.error || JSON.stringify(j).slice(0, 100));
    } catch {}
  }

  const valid = apiStatus === 201 || apiStatus === 400 || apiStatus === 422;
  const noBalance = apiStatus === 402;
  const invalidKey = apiStatus === 403 || apiStatus === 401;

  if (valid) return { status: "valid", message: "Key authenticated successfully with Sapiom API", httpStatus: apiStatus };
  if (noBalance) return { status: "no_balance", message: "Key is valid but account has insufficient balance", httpStatus: apiStatus };
  if (invalidKey) return { status: "invalid", message: `Key rejected by Sapiom API: ${apiBody}`, httpStatus: apiStatus };
  if (apiStatus === 0) return { status: "unreachable", message: "Could not reach Sapiom API", httpStatus: 0 };
  return { status: "invalid", message: `Unexpected response ${apiStatus}: ${apiBody}`, httpStatus: apiStatus };
}

router.post("/keys/:id/validate", async (req, res) => {
  try {
    const id = parseInt(req.params.id);
    if (isNaN(id)) {
      res.status(400).json({ error: "Invalid id" });
      return;
    }
    const [keyRow] = await db.select().from(apiKeysTable).where(eq(apiKeysTable.id, id));
    if (!keyRow) {
      res.status(404).json({ error: "Key not found" });
      return;
    }

    const result = await checkKeyValidity(keyRow.key);

    await db
      .update(apiKeysTable)
      .set({
        validationStatus: result.status,
        validationMessage: result.message,
        validatedAt: new Date(),
      })
      .where(eq(apiKeysTable.id, id));

    res.json({ id, ...result });
  } catch (e) {
    req.log.error(e);
    res.status(500).json({ error: "Internal server error" });
  }
});

const validateAllSchema = z.object({
  autoBan: z.boolean().default(true),
  onlyActive: z.boolean().default(true),
  concurrency: z.number().int().min(1).max(20).default(5),
});

router.post("/keys/validate-all", async (req, res) => {
  try {
    const parsed = validateAllSchema.safeParse(req.body ?? {});
    if (!parsed.success) {
      res.status(400).json({ error: "Invalid request body" });
      return;
    }
    const { autoBan, onlyActive, concurrency } = parsed.data;

    const allKeys = await db.select().from(apiKeysTable);
    const keysToCheck = onlyActive ? allKeys.filter((k) => k.isActive) : allKeys;

    let checked = 0;
    let banned = 0;
    let validCount = 0;
    let invalidCount = 0;
    let noBalanceCount = 0;

    const chunks: typeof keysToCheck[] = [];
    for (let i = 0; i < keysToCheck.length; i += concurrency) {
      chunks.push(keysToCheck.slice(i, i + concurrency));
    }

    for (const chunk of chunks) {
      await Promise.all(
        chunk.map(async (keyRow) => {
          const result = await checkKeyValidity(keyRow.key);
          checked++;

          const updates: Record<string, unknown> = {
            validationStatus: result.status,
            validationMessage: result.message,
            validatedAt: new Date(),
          };

          if (autoBan && result.status === "invalid" && keyRow.isActive) {
            updates.isActive = false;
            banned++;
          }

          await db.update(apiKeysTable).set(updates).where(eq(apiKeysTable.id, keyRow.id));

          if (result.status === "valid") validCount++;
          else if (result.status === "invalid") invalidCount++;
          else if (result.status === "no_balance") noBalanceCount++;
        })
      );
    }

    res.json({
      checked,
      banned,
      valid: validCount,
      invalid: invalidCount,
      noBalance: noBalanceCount,
      autoBan,
    });
  } catch (e) {
    req.log.error(e);
    res.status(500).json({ error: "Internal server error" });
  }
});

export default router;
