/**
 * Incremental Server-Sent Events parser.
 *
 * The previous implementation did `chunk.toString().split('\n\n')` on each
 * 'data' callback. One TCP chunk is not one SSE event: a large event arrives
 * split across several chunks, and several small events arrive in one. Every
 * event that straddled a chunk boundary produced invalid JSON, was swallowed by
 * a bare `catch {}`, and was lost - which is why progress could stall or skip.
 *
 * This parser keeps a persistent buffer, emits only complete events, and
 * retains the trailing partial for the next chunk. It is deliberately a pure
 * string-in/events-out class with no I/O so it can be unit tested against
 * arbitrary chunk splits.
 */
import { StringDecoder } from 'string_decoder';

export interface SseEvent {
    /** The SSE `event:` field, or undefined when the stream omits it. */
    event?: string;
    /** Concatenated `data:` lines, joined with newlines per the SSE spec. */
    data: string;
}

/** Records a data payload that was not valid JSON, so callers can log it. */
export interface SseParseFailure {
    data: string;
    error: string;
}

export class SseParser {
    private buffer = '';
    /**
     * Streaming UTF-8 decoder for Buffer input. Decoding each Buffer on its
     * own garbled any multi-byte character split across two chunks; the
     * decoder holds the incomplete bytes back until the rest arrive.
     */
    private readonly decoder = new StringDecoder('utf8');

    /**
     * Feed raw bytes/text. Returns every event completed by this chunk.
     * A partial trailing event is retained for the next call.
     */
    public push(chunk: string | Buffer): SseEvent[] {
        this.buffer += typeof chunk === 'string' ? chunk : this.decoder.write(chunk);

        const events: SseEvent[] = [];
        // Normalise CRLF so a \r\n\r\n separator is recognised too.
        this.buffer = this.buffer.replace(/\r\n/g, '\n');

        let separator = this.buffer.indexOf('\n\n');
        while (separator !== -1) {
            const raw = this.buffer.slice(0, separator);
            this.buffer = this.buffer.slice(separator + 2);
            const parsed = parseEventBlock(raw);
            if (parsed) {
                events.push(parsed);
            }
            separator = this.buffer.indexOf('\n\n');
        }
        return events;
    }

    /**
     * Called when the stream ends. A well-behaved server terminates the last
     * event with a blank line, but if the connection closed without one, the
     * remaining buffer still holds a complete event that must not be dropped.
     */
    public flush(): SseEvent[] {
        const remainder = (this.buffer + this.decoder.end()).trim();
        this.buffer = '';
        if (!remainder) {
            return [];
        }
        const parsed = parseEventBlock(remainder);
        return parsed ? [parsed] : [];
    }

    /** True when a partial event is still buffered (used in tests/diagnostics). */
    public get pending(): boolean {
        return this.buffer.length > 0;
    }
}

/** Parse one `\n\n`-delimited block into an event, or null if it carries no data. */
function parseEventBlock(block: string): SseEvent | null {
    let event: string | undefined;
    const dataLines: string[] = [];

    for (const line of block.split('\n')) {
        if (!line || line.startsWith(':')) {
            continue; // blank line or comment/keep-alive
        }
        const colon = line.indexOf(':');
        const field = colon === -1 ? line : line.slice(0, colon);
        // Per the SSE spec a single leading space after the colon is stripped.
        let value = colon === -1 ? '' : line.slice(colon + 1);
        if (value.startsWith(' ')) {
            value = value.slice(1);
        }

        if (field === 'data') {
            dataLines.push(value);
        } else if (field === 'event') {
            event = value;
        }
    }

    if (dataLines.length === 0) {
        return null;
    }
    return { event, data: dataLines.join('\n') };
}

/**
 * Parse an event's data as JSON.
 *
 * Returns either the value or a failure describing why - never silently
 * discards it, so callers can log malformed payloads instead of hiding them.
 */
export function parseJsonEvent<T = any>(event: SseEvent): { ok: true; value: T } | { ok: false; failure: SseParseFailure } {
    try {
        return { ok: true, value: JSON.parse(event.data) as T };
    } catch (err) {
        return {
            ok: false,
            failure: { data: event.data, error: err instanceof Error ? err.message : String(err) },
        };
    }
}
