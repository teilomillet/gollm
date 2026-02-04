package llm

import (
	"strings"
	"testing"
)

func TestSSEDecoder(t *testing.T) {
	sseData := `event: message
data: Hello

event: message
data: World

`
	reader := strings.NewReader(sseData)
	decoder := NewSSEDecoder(reader)

	// First event
	if !decoder.Next() {
		t.Fatal("Expected first event")
	}
	event := decoder.Event()
	if event.Type != "message" {
		t.Errorf("Expected type 'message', got '%s'", event.Type)
	}
	if string(event.Data) != "Hello\n" {
		t.Errorf("Expected data 'Hello\\n', got '%s'", string(event.Data))
	}

	// Second event
	if !decoder.Next() {
		t.Fatal("Expected second event")
	}
	event = decoder.Event()
	if event.Type != "message" {
		t.Errorf("Expected type 'message', got '%s'", event.Type)
	}
	if string(event.Data) != "World\n" {
		t.Errorf("Expected data 'World\\n', got '%s'", string(event.Data))
	}

	// No more events
	if decoder.Next() {
		t.Error("Expected no more events")
	}
}

func TestNDJSONDecoder(t *testing.T) {
	ndjsonData := `{"response": "Hello", "done": false}
{"response": "World", "done": true}
`
	reader := strings.NewReader(ndjsonData)
	decoder := NewNDJSONDecoder(reader)

	// First line
	if !decoder.Next() {
		t.Fatal("Expected first line")
	}
	event := decoder.Event()
	if event.Type != "text" {
		t.Errorf("Expected type 'text', got '%s'", event.Type)
	}
	expected := `{"response": "Hello", "done": false}`
	if string(event.Data) != expected {
		t.Errorf("Expected data '%s', got '%s'", expected, string(event.Data))
	}

	// Second line
	if !decoder.Next() {
		t.Fatal("Expected second line")
	}
	event = decoder.Event()
	if event.Type != "text" {
		t.Errorf("Expected type 'text', got '%s'", event.Type)
	}
	expected = `{"response": "World", "done": true}`
	if string(event.Data) != expected {
		t.Errorf("Expected data '%s', got '%s'", expected, string(event.Data))
	}

	// No more lines
	if decoder.Next() {
		t.Error("Expected no more lines")
	}
}

func TestNDJSONDecoderSkipsEmptyLines(t *testing.T) {
	ndjsonData := `{"response": "Hello", "done": false}

{"response": "World", "done": true}
`
	reader := strings.NewReader(ndjsonData)
	decoder := NewNDJSONDecoder(reader)

	// First line
	if !decoder.Next() {
		t.Fatal("Expected first line")
	}
	event := decoder.Event()
	expected := `{"response": "Hello", "done": false}`
	if string(event.Data) != expected {
		t.Errorf("Expected data '%s', got '%s'", expected, string(event.Data))
	}

	// Second line (should skip empty line)
	if !decoder.Next() {
		t.Fatal("Expected second line")
	}
	event = decoder.Event()
	expected = `{"response": "World", "done": true}`
	if string(event.Data) != expected {
		t.Errorf("Expected data '%s', got '%s'", expected, string(event.Data))
	}
}

func TestNDJSONDecoderManyEmptyLines(t *testing.T) {
	// Test that many empty lines don't cause stack overflow (was recursive before)
	var sb strings.Builder
	sb.WriteString(`{"response": "Start", "done": false}`)
	sb.WriteString("\n")
	// Add 10000 empty lines - would cause stack overflow with recursive implementation
	for i := 0; i < 10000; i++ {
		sb.WriteString("\n")
	}
	sb.WriteString(`{"response": "End", "done": true}`)
	sb.WriteString("\n")

	reader := strings.NewReader(sb.String())
	decoder := NewNDJSONDecoder(reader)

	// First line
	if !decoder.Next() {
		t.Fatal("Expected first line")
	}
	event := decoder.Event()
	if string(event.Data) != `{"response": "Start", "done": false}` {
		t.Errorf("Unexpected first data: %s", string(event.Data))
	}

	// Second line (should skip all 10000 empty lines without stack overflow)
	if !decoder.Next() {
		t.Fatal("Expected second line after skipping empty lines")
	}
	event = decoder.Event()
	if string(event.Data) != `{"response": "End", "done": true}` {
		t.Errorf("Unexpected second data: %s", string(event.Data))
	}

	// No more lines
	if decoder.Next() {
		t.Error("Expected no more lines")
	}

	if decoder.Err() != nil {
		t.Errorf("Unexpected error: %v", decoder.Err())
	}
}
