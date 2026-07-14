package main

import (
	_ "embed"
	"flag"
	"fmt"
	"os"
	"path/filepath"
)

// agentSkill is the instruction file that teaches an agent to use turbograph as a
// durable knowledge base. It is embedded in the binary rather than being a file the
// user has to find: the whole point of driving turbograph from a shell is that an
// agent needs nothing but the binary, and a skill it cannot locate is a skill it does
// not have.
//
//go:embed skill/SKILL.md
var agentSkill string

// cmdSkill prints the agent skill, or installs it where a harness will find it.
// Claude Code and several other harnesses read skills from a directory of markdown
// files with front matter; the same file is plain markdown, so a harness without a
// skill mechanism can be handed it as instructions directly, or a repo can commit it
// alongside its AGENTS.md.
func cmdSkill(args []string) error {
	fs := flag.NewFlagSet("skill", flag.ExitOnError)
	install := fs.Bool("install", false, "write the skill to a skills directory instead of printing it")
	dir := fs.String("dir", "", "where to install (default: ~/.claude/skills/turbograph)")
	force := fs.Bool("force", false, "overwrite an existing skill file")
	fs.Parse(args)

	if !*install {
		fmt.Print(agentSkill)
		return nil
	}

	target := *dir
	if target == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return fmt.Errorf("cannot find your home directory; pass --dir: %w", err)
		}
		target = filepath.Join(home, ".claude", "skills", "turbograph")
	}
	if err := os.MkdirAll(target, 0o755); err != nil {
		return err
	}
	path := filepath.Join(target, "SKILL.md")
	if !*force {
		if _, err := os.Stat(path); err == nil {
			return fmt.Errorf("%s already exists; pass --force to overwrite it", path)
		}
	}
	if err := os.WriteFile(path, []byte(agentSkill), 0o644); err != nil {
		return err
	}
	fmt.Printf("installed the turbograph skill to %s\n", path)
	fmt.Println("set TURBOGRAPH_STORE (and TURBOGRAPH_MODEL for `ask`) and an agent can drive it from a shell.")
	return nil
}
