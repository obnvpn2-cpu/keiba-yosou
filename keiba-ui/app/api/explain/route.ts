import { NextRequest, NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

/**
 * Default explain JSON path (relative to project root)
 */
const DEFAULT_EXPLAIN_PATH = "../models/explain_target_win_v4.json";

/**
 * GET /api/explain
 *
 * Query params:
 * - target: target name (default: target_win)
 * - version: model version (default: v4)
 */
export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const target = searchParams.get("target") || "target_win";
    const version = searchParams.get("version") || "v4";

    // Build file path
    const filename = `explain_${target}_${version}.json`;
    const filePath = path.resolve(process.cwd(), "..", "models", filename);

    // Check if file exists
    try {
      await fs.access(filePath);
    } catch {
      return NextResponse.json(
        {
          error: "Explain JSON not found",
          message: `File not found: ${filename}`,
          hint: `Run: python -m src.features_v4.explain_runner --model-dir models --target ${target}`,
        },
        { status: 404 }
      );
    }

    // Read and parse JSON
    let content = await fs.readFile(filePath, "utf-8");

    // Sanitize legacy files with NaN values (invalid JSON)
    if (content.includes(": NaN")) {
      content = content.replace(/: NaN([,\n\r\}])/g, ": null$1");
    }

    let data;
    try {
      data = JSON.parse(content);
    } catch (parseError) {
      const parseMessage = parseError instanceof Error ? parseError.message : "Unknown parse error";
      return NextResponse.json(
        {
          error: "Invalid JSON",
          message: `Failed to parse ${filename}: ${parseMessage}`,
          hint: "Regenerate with: python -m src.features_v4.explain_runner --model-dir models --target " + target,
        },
        { status: 500 }
      );
    }

    return NextResponse.json(data);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json(
      {
        error: "Failed to load explain data",
        message,
      },
      { status: 500 }
    );
  }
}
