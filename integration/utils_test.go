//go:build integration

package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math/rand"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/app/lifecycle"
	"github.com/stretchr/testify/assert"
)

func FindPort() string {
	port := 0
	if a, err := net.ResolveTCPAddr("tcp", "localhost:0"); err == nil {
		var l *net.TCPListener
		if l, err = net.ListenTCP("tcp", a); err == nil {
			port = l.Addr().(*net.TCPAddr).Port
			l.Close()
		}
	}
	if port == 0 {
		port = rand.Intn(65535-49152) + 49152 // get a random port in the ephemeral range
	}
	return strconv.Itoa(port)
}

func GetTestEndpoint() (string, string) {
	defaultPort := "11434"
	ollamaHost := os.Getenv("OLLAMA_HOST")

	scheme, hostport, ok := strings.Cut(ollamaHost, "://")
	if !ok {
		scheme, hostport = "http", ollamaHost
	}

	// trim trailing slashes
	hostport = strings.TrimRight(hostport, "/")

	host, port, err := net.SplitHostPort(hostport)
	if err != nil {
		host, port = "127.0.0.1", defaultPort
		if ip := net.ParseIP(strings.Trim(hostport, "[]")); ip != nil {
			host = ip.String()
		} else if hostport != "" {
			host = hostport
		}
	}

	if os.Getenv("OLLAMA_TEST_EXISTING") == "" && port == defaultPort {
		port = FindPort()
	}

	url := fmt.Sprintf("%s:%s", host, port)
	slog.Info("server connection", "url", url)
	return scheme, url
}

// TODO make fanicier, grab logs, etc.
var serverMutex sync.Mutex
var serverReady bool

func StartServer(ctx context.Context, ollamaHost string) error {
	// Make sure the server has been built
	CLIName, err := filepath.Abs("../ollama")
	if err != nil {
		return err
	}

	if runtime.GOOS == "windows" {
		CLIName += ".exe"
	}
	_, err = os.Stat(CLIName)
	if err != nil {
		return fmt.Errorf("CLI missing, did you forget to build first?  %w", err)
	}
	serverMutex.Lock()
	defer serverMutex.Unlock()
	if serverReady {
		return nil
	}

	if tmp := os.Getenv("OLLAMA_HOST"); tmp != ollamaHost {
		slog.Info("setting env", "OLLAMA_HOST", ollamaHost)
		os.Setenv("OLLAMA_HOST", ollamaHost)
	}

	slog.Info("starting server", "url", ollamaHost)
	done, err := lifecycle.SpawnServer(ctx, "../ollama")
	if err != nil {
		return fmt.Errorf("failed to start server: %w", err)
	}

	go func() {
		<-ctx.Done()
		serverMutex.Lock()
		defer serverMutex.Unlock()
		exitCode := <-done
		if exitCode > 0 {
			slog.Warn("server failure", "exit", exitCode)
		}
		serverReady = false
	}()

	// TODO wait only long enough for the server to be responsive...
	time.Sleep(500 * time.Millisecond)

	serverReady = true
	return nil
}

func GenerateTestHelper(ctx context.Context, t *testing.T, client *http.Client, genReq api.GenerateRequest, anyResp []string) {
	requestJSON, err := json.Marshal(genReq)
	if err != nil {
		t.Fatalf("Error serializing request: %v", err)
	}
	defer func() {
		if t.Failed() && os.Getenv("OLLAMA_TEST_EXISTING") == "" {
			// TODO
			fp, err := os.Open(lifecycle.ServerLogFile)
			if err != nil {
				slog.Error("failed to open server log", "logfile", lifecycle.ServerLogFile, "error", err)
				return
			}
			data, err := io.ReadAll(fp)
			if err != nil {
				slog.Error("failed to read server log", "logfile", lifecycle.ServerLogFile, "error", err)
				return
			}
			slog.Warn("SERVER LOG FOLLOWS")
			os.Stderr.Write(data)
			slog.Warn("END OF SERVER")
		}
		err = os.Remove(lifecycle.ServerLogFile)
		if err != nil && !os.IsNotExist(err) {
			slog.Warn("failed to cleanup", "logfile", lifecycle.ServerLogFile, "error", err)
		}
	}()
	scheme, testEndpoint := GetTestEndpoint()

	if os.Getenv("OLLAMA_TEST_EXISTING") == "" {
		assert.NoError(t, StartServer(ctx, testEndpoint))
	}

	// Make the request and get the response
	req, err := http.NewRequest("POST", scheme+"://"+testEndpoint+"/api/generate", bytes.NewReader(requestJSON))
	if err != nil {
		t.Fatalf("Error creating request: %v", err)
	}

	// Set the content type for the request
	req.Header.Set("Content-Type", "application/json")

	// Make the request with the HTTP client
	response, err := client.Do(req.WithContext(ctx))
	if err != nil {
		t.Fatalf("Error making request: %v", err)
	}
	body, err := io.ReadAll(response.Body)
	assert.NoError(t, err)
	assert.Equal(t, response.StatusCode, 200, string(body))

	// Verify the response is valid JSON
	var payload api.GenerateResponse
	err = json.Unmarshal(body, &payload)
	if err != nil {
		assert.NoError(t, err, body)
	}

	// Verify the response contains the expected data
	for _, resp := range anyResp {
		assert.Contains(t, strings.ToLower(payload.Response), resp)
	}
}
