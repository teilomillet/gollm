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
