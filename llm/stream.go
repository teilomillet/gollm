package llm

import (
	"bufio"
	"bytes"
	"context"
	"io"
)

// StreamToken represents a single token from the streaming response.
type StreamToken struct {
	// Text is the actual token text
	Text string

	// Type indicates the type of token (e.g., "text", "function_call", "error")
	Type string

	// Index is the position of this token in the sequence
	Index int

	// Number of input tokens processed before this token
	InputTokens int64

	// Number of output tokens generated up to this token
	OutputTokens int64

	// Metadata contains provider-specific metadata
	Metadata map[string]any
}

// TokenStream represents a stream of tokens from the LLM.
// It follows Go's io.ReadCloser pattern but with token-level granularity.
type TokenStream interface {
	// Next returns the next token in the stream.
	// When the stream is finished, it returns io.EOF.
	Next(context.Context) (*StreamToken, error)

	// Closer Close releases any resources associated with the stream.
	io.Closer
}

// SSEDecoder handles Server-Sent Events (SSE) streaming
type SSEDecoder struct {
	reader  *bufio.Scanner
	current Event
	err     error
}

type Event struct {
	Type string
	Data []byte
}

func NewSSEDecoder(reader io.Reader) *SSEDecoder {
	return &SSEDecoder{
		reader: bufio.NewScanner(reader),
	}
}

func (d *SSEDecoder) Next() bool {
	if d.err != nil {
		return false
	}

	event := ""
	data := bytes.NewBuffer(nil)

	for d.reader.Scan() {
		line := d.reader.Bytes()

		// Dispatch event on empty line
		if len(line) == 0 {
			d.current = Event{
				Type: event,
				Data: data.Bytes(),
			}
			return true
		}

		// Split "event: value" into parts
		name, value, _ := bytes.Cut(line, []byte(":"))

		// Remove optional space after colon
		if len(value) > 0 && value[0] == ' ' {
			value = value[1:]
		}

		switch string(name) {
		case "":
			continue // Skip comments
		case "event":
			event = string(value)
		case "data":
			data.Write(value)
			data.WriteRune('\n')
		}
	}

	return false
}

func (d *SSEDecoder) Event() Event {
	return d.current
}

func (d *SSEDecoder) Err() error {
	return d.err
}
