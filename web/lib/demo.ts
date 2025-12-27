export const DEMO_MODE =
  process.env.DEMO_MODE === "true" ||
  process.env.NEXT_PUBLIC_DEMO_MODE === "true";

export const DEMO_PUBLIC_READONLY =
  process.env.NEXT_PUBLIC_DEMO_PUBLIC_READONLY === "true" ||
  process.env.DEMO_PUBLIC_READONLY === "true" ||
  DEMO_MODE;

export const DEMO_READONLY_MESSAGE =
  process.env.NEXT_PUBLIC_DEMO_READONLY_MESSAGE ||
  "Demo mode: read-only experience. Some actions are disabled.";

export const DEMO_READONLY_BANNER =
  process.env.NEXT_PUBLIC_DEMO_READONLY_BANNER ||
  "Demo Mode (Read-only)";
