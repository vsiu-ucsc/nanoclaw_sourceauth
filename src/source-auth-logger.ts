import fs from 'fs';
import path from 'path';

import { BENCHMARK_MODE, DATA_DIR, GROUPS_DIR } from './config.js';
import { MessageProvenance } from './types.js';

export interface SourceAuthEvent {
  timestamp: string;
  scenarioId?: string;
  groupName: string;
  operationType: string;
  decision: 'blocked' | 'allowed' | 'warned';
  unauthenticatedSources: MessageProvenance[];
  authenticatedSources: MessageProvenance[];
  reason: string;
}

export function logSourceAuthEvent(event: SourceAuthEvent): void {
  const line = JSON.stringify(event) + '\n';

  // Per-group log
  const groupLogsDir = path.join(GROUPS_DIR, event.groupName, 'logs');
  fs.mkdirSync(groupLogsDir, { recursive: true });
  fs.appendFileSync(
    path.join(groupLogsDir, 'source-auth-violations.jsonl'),
    line,
  );

  // Benchmark-wide log (only when benchmark mode active)
  if (BENCHMARK_MODE) {
    const benchmarkDir = path.join(DATA_DIR, 'benchmark');
    fs.mkdirSync(benchmarkDir, { recursive: true });
    fs.appendFileSync(path.join(benchmarkDir, 'results.jsonl'), line);
  }
}
