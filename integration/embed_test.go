//go:build integration

package integration

import (
	"context"
	"math"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func dotProduct(v1, v2 []float64) float64 {
	var result float64 = 0
	for i := 0; i < len(v1); i++ {
		result += v1[i] * v2[i]
	}
	return result
}

func magnitude(v []float64) float64 {
	var result float64 = 0
	for _, val := range v {
		result += val * val
	}
	return math.Sqrt(result)
}

func cosineSimilarity(v1, v2 []float64) float64 {
	return dotProduct(v1, v2) / (magnitude(v1) * magnitude(v2))
}

func TestAllMiniLMEmbeddings(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	req := api.EmbeddingRequest{
		Model:  "all-minilm",
		Prompt: "why is the sky blue?",
	}

	res, err := embeddingTestHelper(ctx, t, req)

	if err != nil {
		t.Fatalf("error: %v", err)
	}

	if len(res.Embedding) != 384 {
		t.Fatalf("expected 384 floats, got %d", len(res.Embedding))
	}

	expected := 0.06642947345972061
	sim := cosineSimilarity([]float64{res.Embedding[0]}, []float64{expected})
	if sim < 0.99 {
		t.Fatalf("expected %.16f, got %.16f (similarity: %f)", expected, res.Embedding[0], sim)
	}
}

func TestAllMiniLMEmbed(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	req := api.EmbedRequest{
		Model: "all-minilm",
		Input: "why is the sky blue?",
	}

	res, err := embedTestHelper(ctx, t, req)

	if err != nil {
		t.Fatalf("error: %v", err)
	}

	if len(res.Embeddings) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(res.Embeddings))
	}

	if len(res.Embeddings[0]) != 384 {
		t.Fatalf("expected 384 floats, got %d", len(res.Embeddings[0]))
	}

	expected := 0.010071031
	sim := cosineSimilarity([]float64{float64(res.Embeddings[0][0])}, []float64{expected})
	if sim < 0.99 {
		t.Fatalf("expected %.8f, got %.8f (similarity: %f)", expected, res.Embeddings[0][0], sim)
	}

	if res.PromptEvalCount != 6 {
		t.Fatalf("expected 6 prompt tokens, got %d", res.PromptEvalCount)
	}
}

func TestAllMiniLMBatchEmbed(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	req := api.EmbedRequest{
		Model: "all-minilm",
		Input: []string{"why is the sky blue?", "why is the grass green?"},
	}

	res, err := embedTestHelper(ctx, t, req)

	if err != nil {
		t.Fatalf("error: %v", err)
	}

	if len(res.Embeddings) != 2 {
		t.Fatalf("expected 2 embeddings, got %d", len(res.Embeddings))
	}

	if len(res.Embeddings[0]) != 384 {
		t.Fatalf("expected 384 floats, got %d", len(res.Embeddings[0]))
	}

	expected := []float64{0.010071031, -0.009802706}
	sim := cosineSimilarity([]float64{float64(res.Embeddings[0][0]), float64(res.Embeddings[1][0])}, expected)
	if sim < 0.99 {
		t.Fatalf("expected %v, got %.8f and %.8f (similarity: %f)", expected, res.Embeddings[0][0], res.Embeddings[1][0], sim)
	}

	if res.PromptEvalCount != 12 {
		t.Fatalf("expected 12 prompt tokens, got %d", res.PromptEvalCount)
	}
}

func TestAllMiniLMEmbedTruncate(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	truncTrue, truncFalse := true, false

	type testReq struct {
		Name    string
		Request api.EmbedRequest
	}

	reqs := []testReq{
		{
			Name: "Target Truncation",
			Request: api.EmbedRequest{
				Model: "all-minilm",
				Input: "why",
			},
		},
		{
			Name: "Default Truncate",
			Request: api.EmbedRequest{
				Model:   "all-minilm",
				Input:   "why is the sky blue?",
				Options: map[string]any{"num_ctx": 1},
			},
		},
		{
			Name: "Explicit Truncate",
			Request: api.EmbedRequest{
				Model:    "all-minilm",
				Input:    "why is the sky blue?",
				Truncate: &truncTrue,
				Options:  map[string]any{"num_ctx": 1},
			},
		},
	}

	res := make(map[string]*api.EmbedResponse)

	for _, req := range reqs {
		response, err := embedTestHelper(ctx, t, req.Request)
		if err != nil {
			t.Fatalf("error: %v", err)
		}
		res[req.Name] = response
	}

	if res["Target Truncation"].Embeddings[0][0] != res["Default Truncate"].Embeddings[0][0] {
		t.Fatal("expected default request to truncate correctly")
	}

	if res["Default Truncate"].Embeddings[0][0] != res["Explicit Truncate"].Embeddings[0][0] {
		t.Fatal("expected default request and truncate true request to be the same")
	}

	// check that truncate set to false returns an error if context length is exceeded
	_, err := embedTestHelper(ctx, t, api.EmbedRequest{
		Model:    "all-minilm",
		Input:    "why is the sky blue?",
		Truncate: &truncFalse,
		Options:  map[string]any{"num_ctx": 1},
	})

	if err == nil {
		t.Fatal("expected error, got nil")
	}
}

func embeddingTestHelper(ctx context.Context, t *testing.T, req api.EmbeddingRequest) (*api.EmbeddingResponse, error) {
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	if err := PullIfMissing(ctx, client, req.Model); err != nil {
		t.Fatalf("failed to pull model %s: %v", req.Model, err)
	}

	response, err := client.Embeddings(ctx, &req)

	if err != nil {
		return nil, err
	}

	return response, nil
}

func embedTestHelper(ctx context.Context, t *testing.T, req api.EmbedRequest) (*api.EmbedResponse, error) {
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	if err := PullIfMissing(ctx, client, req.Model); err != nil {
		t.Fatalf("failed to pull model %s: %v", req.Model, err)
	}

	response, err := client.Embed(ctx, &req)

	if err != nil {
		return nil, err
	}

	return response, nil
}
