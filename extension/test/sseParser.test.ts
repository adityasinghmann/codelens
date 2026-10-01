/**
 * The SSE parser, exercised at every chunk boundary.
 *
 * The bug being guarded against: the previous implementation parsed each TCP
 * chunk in isolation with split('\n\n'), so any event straddling a chunk
 * boundary produced invalid JSON and was silently dropped.
 */
import { SseParser, parseJsonEvent, SseEvent } from '../src/sseParser';

/** A representative index stream: five progress events plus completion. */
function buildStream() {
    const events: unknown[] = [];
    for (let i = 1; i <= 5; i++) {
        events.push({
            type: 'progress',
            file: `src/very/long/path/module_${i}.py`,
            processed_files: i,
            total_files: 5,
            processed_chunks: i * 3,
        });
    }
    events.push({ type: 'complete', total_files: 5, processed_files: 5, total_chunks: 15, stored: 15, skipped: 0, failed: 0, duration_ms: 1234 });
    return { events, text: events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join('') };
}

function collect(chunks: string[]): SseEvent[] {
    const parser = new SseParser();
    const out: SseEvent[] = [];
    for (const chunk of chunks) {
        out.push(...parser.push(chunk));
    }
    out.push(...parser.flush());
    return out;
}

describe('SseParser', () => {
    it('parses a whole stream delivered in one chunk', () => {
        const { text, events } = buildStream();
        expect(collect([text])).toHaveLength(events.length);
    });

    it('parses correctly at EVERY possible chunk boundary', () => {
        const { text, events } = buildStream();
        const failures: number[] = [];

        for (let split = 1; split < text.length; split++) {
            const got = collect([text.slice(0, split), text.slice(split)]);
            const parsed = got.map((e) => parseJsonEvent(e));
            if (got.length !== events.length || !parsed.every((p) => p.ok)) {
                failures.push(split);
            }
        }

        expect(failures).toEqual([]);
    });

    it('parses a stream delivered one byte at a time', () => {
        const { text, events } = buildStream();
        const got = collect(text.split(''));
        expect(got).toHaveLength(events.length);
        expect(got.every((e) => parseJsonEvent(e).ok)).toBe(true);
    });

    it('preserves event payloads across a split', () => {
        const { text } = buildStream();
        const mid = Math.floor(text.length / 2);
        const got = collect([text.slice(0, mid), text.slice(mid)]);
        const last = parseJsonEvent<any>(got[got.length - 1]);
        expect(last.ok).toBe(true);
        const value = (last as { value: any }).value;
        expect(value.type).toBe('complete');
        expect(value.total_chunks).toBe(15);
    });

    it('demonstrates the old naive parser losing events (regression proof)', () => {
        const { text } = buildStream();
        const naive = (chunks: string[]) => {
            let kept = 0;
            for (const chunk of chunks) {
                for (const block of chunk.split('\n\n')) {
                    if (block.startsWith('data: ')) {
                        try { JSON.parse(block.substring(6)); kept++; } catch { /* dropped */ }
                    }
                }
            }
            return kept;
        };
        const mid = Math.floor(text.length / 2);
        expect(naive([text.slice(0, mid), text.slice(mid)])).toBeLessThan(6);
        expect(naive(text.split(''))).toBe(0);
    });

    it('recovers a final event that was not terminated by a blank line', () => {
        const parser = new SseParser();
        const partial = 'data: {"type":"complete","total_chunks":1}';
        expect(parser.push(partial)).toHaveLength(0);
        expect(parser.pending).toBe(true);

        const flushed = parser.flush();
        expect(flushed).toHaveLength(1);
        expect(parseJsonEvent<any>(flushed[0]).ok).toBe(true);
    });

    it('reports malformed JSON instead of swallowing it', () => {
        const parser = new SseParser();
        const [event] = parser.push('data: {not json}\n\n');
        const parsed = parseJsonEvent(event);
        expect(parsed.ok).toBe(false);
        expect('failure' in parsed).toBe(true);
        const failure = (parsed as { failure: { data: string; error: string } }).failure;
        expect(failure.data).toBe('{not json}');
        expect(failure.error).toBeTruthy();
    });

    it('handles CRLF separators, comments, multi-line data and named events', () => {
        const parser = new SseParser();
        const events = parser.push(': keep-alive\r\n\r\nevent: progress\r\ndata: {"a":\r\ndata: 1}\r\n\r\n');
        expect(events).toHaveLength(1);
        expect(events[0].event).toBe('progress');
        const parsed = parseJsonEvent<any>(events[0]);
        expect(parsed.ok).toBe(true);
        expect((parsed as { value: any }).value.a).toBe(1);
    });

    it('ignores keep-alive comments entirely', () => {
        const parser = new SseParser();
        expect(parser.push(': ping\n\n: ping\n\n')).toHaveLength(0);
    });

    it('flushes nothing when the buffer is empty', () => {
        const parser = new SseParser();
        parser.push('data: {"a":1}\n\n');
        expect(parser.flush()).toHaveLength(0);
        expect(parser.pending).toBe(false);
    });

    it('accepts Buffer input as well as strings', () => {
        const parser = new SseParser();
        const events = parser.push(Buffer.from('data: {"a":1}\n\n', 'utf-8'));
        expect(events).toHaveLength(1);
    });
});

describe('SseParser with raw bytes', () => {
    it('keeps a multi-byte character intact when it is split across chunks', () => {
        const bytes = Buffer.from('data: {"file":"café.py"}\n\n', 'utf-8');
        const cut = bytes.indexOf(0xc3) + 1; // inside the two-byte "é"
        const parser = new SseParser();

        const events = [...parser.push(bytes.subarray(0, cut)), ...parser.push(bytes.subarray(cut))];

        expect(events.map((e) => e.data)).toEqual(['{"file":"café.py"}']);
    });

    it('decodes byte-by-byte delivery correctly', () => {
        const bytes = Buffer.from('data: ✓ 日本\n\n', 'utf-8');
        const parser = new SseParser();
        const events: SseEvent[] = [];
        for (let i = 0; i < bytes.length; i++) {
            events.push(...parser.push(bytes.subarray(i, i + 1)));
        }
        expect(events.map((e) => e.data)).toEqual(['✓ 日本']);
    });
});
