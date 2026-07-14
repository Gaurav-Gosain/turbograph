package redact

import (
	"slices"
	"strings"
	"testing"
)

// TestRedactsRealCredentials: each of these is a live-format credential that an agent
// could plausibly scrape out of a shell session or a config file and index.
func TestRedactsRealCredentials(t *testing.T) {
	// The fixtures are assembled from parts rather than written as literals. They are
	// structurally real credential formats -- that is the whole point of the test -- and
	// as literals they trip GitHub's push protection and every other secret scanner,
	// which would make this file unpushable and teach the next person to weaken it.
	j := func(parts ...string) string { return strings.Join(parts, "") }
	aws := j("AKIA", "IOSFODNN7EXAMPLE")
	gh := j("ghp", "_", "aBcD1234567890aBcD1234567890aBcD1234")
	anth := j("sk", "-ant-", "api03-AbCdEfGhIjKlMnOpQrStUvWxYz123456")
	oai := j("sk", "-proj-", "AbCdEfGhIjKlMnOpQrStUvWxYz0123456789")
	slack := j("xox", "b-", "123456789012-abcdefghijklmnop")
	stripe := j("sk", "_live_", "51HxYzAbCdEfGhIjKlMnOp")
	goog := j("AIza", "SyA1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q")
	jwt := j("eyJhbGciOiJIUzI1NiJ9", ".", "eyJzdWIiOiIxMjM0NTY3ODkwIn0", ".", "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U")

	cases := []struct {
		name   string
		text   string
		secret string // must NOT survive
		kind   string
	}{
		{"aws access key", "export AWS_ACCESS_KEY_ID=" + aws, aws, "aws-access-key-id"},
		{"github token", "git remote set-url origin https://" + gh + "@github.com/x/y", gh, "github-token"},
		{"anthropic key", "ANTHROPIC_API_KEY=" + anth, anth, "anthropic-api-key"},
		{"openai key", "the client uses " + oai + " for auth", oai, "openai-api-key"},
		{"slack token", "SLACK_BOT_TOKEN=" + slack, slack, "slack-token"},
		{"stripe key", "billing uses " + stripe, stripe, "stripe-key"},
		{"google key", "maps key " + goog, goog, "google-api-key"},
		{"jwt", "Authorization: Bearer " + jwt, jwt, "jwt"},
		{"db url password", "DATABASE_URL=postgres://svc:s3cr3tP4ssw0rd@db.internal:5432/app", "s3cr3tP4ssw0rd", "url-credentials"},
		{"assigned secret", `config: {"client_secret": "abc123def456ghi789jkl"}`, "abc123def456ghi789jkl", "assigned-secret"},
		{"private key", "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA1234\n-----END RSA PRIVATE KEY-----", "MIIEowIBAAKCAQEA1234", "private-key"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, found := Text(c.text)
			if strings.Contains(got, c.secret) {
				t.Errorf("the secret survived redaction:\n  in:  %s\n  out: %s", c.text, got)
			}
			if len(found) == 0 {
				t.Fatalf("nothing reported as redacted for %q", c.text)
			}
			var kinds []string
			for _, f := range found {
				kinds = append(kinds, f.Kind)
			}
			if !slices.Contains(kinds, c.kind) {
				t.Errorf("reported %v, want it to include %q", kinds, c.kind)
			}
		})
	}
}

// TestKeepsTheContextReadable: a redacted passage must still be worth retrieving. If
// redaction destroyed the sentence, an agent's knowledge base would lose the fact
// along with the secret.
func TestKeepsTheContextReadable(t *testing.T) {
	got, _ := Text("DATABASE_URL=postgres://svc:s3cr3tP4ssw0rd@db.internal:5432/app")
	for _, want := range []string{"postgres://", "db.internal", "5432", "app"} {
		if !strings.Contains(got, want) {
			t.Errorf("redaction destroyed the context: %q is gone from %q", want, got)
		}
	}
	if !strings.Contains(got, "[redacted:") {
		t.Errorf("the redaction is not visible to a reader: %q", got)
	}
}

// TestDoesNotFireOnProse is the property that decides whether anyone leaves this on.
// A redactor that mangles ordinary technical writing gets disabled, and then it
// protects nothing.
func TestDoesNotFireOnProse(t *testing.T) {
	clean := []string{
		"Refresh tokens rotate on every use; the old token is revoked immediately.",
		"The retry queue is capped at 3 attempts because the provider treats a 4th request as a duplicate.",
		"Set the api_key environment variable before running the test suite.",
		"We store the password hash with bcrypt, never the password itself.",
		"See https://github.com/Gaurav-Gosain/turbograph for the source.",
		"The access token expires after 30 minutes of idle time.",
		"Call client.Generate(ctx, model, system, prompt) to get a completion.",
		"turbograph query --store store.tg --q \"what is the retry cap\"",
	}
	for _, s := range clean {
		got, found := Text(s)
		if len(found) != 0 {
			t.Errorf("false positive on prose:\n  %q\n  reported: %+v\n  became:   %q", s, found, got)
		}
		if got != s {
			t.Errorf("prose was modified:\n  in:  %q\n  out: %q", s, got)
		}
	}
}

// TestPlaceholdersAreNotSecrets: documentation showing where a key goes must not be
// reported as a leak, or the warning becomes noise and gets ignored.
func TestPlaceholdersAreNotSecrets(t *testing.T) {
	docs := []string{
		`api_key = "YOUR_API_KEY_HERE"`,
		`export ANTHROPIC_API_KEY=<your-key>`,
		`password: changeme123`,
		`client_secret: "xxxxxxxxxxxxxxxx"`,
		`api_key=${OPENAI_API_KEY}`,
	}
	for _, s := range docs {
		_, found := Text(s)
		if len(found) != 0 {
			t.Errorf("a placeholder was reported as a secret: %q -> %+v", s, found)
		}
	}
}

// TestRedactionIsIdempotent: re-ingesting already-redacted text must not redact the
// markers again, or repeated ingestion would nest them.
func TestRedactionIsIdempotent(t *testing.T) {
	once, _ := Text("AWS_ACCESS_KEY_ID=" + strings.Join([]string{"AKIA", "IOSFODNN7EXAMPLE"}, ""))
	twice, found := Text(once)
	if once != twice {
		t.Errorf("redacting twice changed the text:\n  once:  %q\n  twice: %q", once, twice)
	}
	if len(found) != 0 {
		t.Errorf("already-redacted text reported new secrets: %+v", found)
	}
}

func TestSummary(t *testing.T) {
	j := func(parts ...string) string { return strings.Join(parts, "") }
	_, f := Text("AWS_ACCESS_KEY_ID=" + j("AKIA", "IOSFODNN7EXAMPLE") +
		" and " + j("sk", "-ant-", "api03-AbCdEfGhIjKlMnOpQrStUvWxYz123456"))
	s := Summary(f)
	if !strings.Contains(s, "2 secrets") {
		t.Errorf("summary should count both: %q", s)
	}
}
